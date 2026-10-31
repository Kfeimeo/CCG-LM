import importlib.util
from importlib.machinery import SourceFileLoader
import itertools
from pathlib import Path
import sys
import pytest
import torch
from models.composer import Composer
from models.forest import SemanticForest
from validate_cky_v2 import forest_case, model_gradient_case, training_case


@pytest.mark.parametrize("batch,words,k,r,vector", list(itertools.product([1, 2, 4], [4, 8, 16, 24], [1, 2, 4], [4, 12], [False, True])))
def test_v2_forward_matrix(batch, words, k, r, vector):
    forest_case(batch, words, k, r, vector, policy="explore" if vector else "normal")


@pytest.mark.parametrize("policy", ["prune", "explore", "ties", "unreachable", "empty", "partial_inf"])
@pytest.mark.parametrize("vector", [False, True])
def test_v2_search_edges(policy, vector):
    forest_case(2, 8, 4, 12, vector, policy=policy)


@pytest.mark.parametrize("vector", [False, True])
def test_v2_single_word(vector):
    forest_case(2, 1, 4, 12, vector)


@pytest.mark.parametrize("mode", ["forest_scalar4", "forest_vector4"])
def test_v2_lm_gradients(mode):
    model_gradient_case(mode)


@pytest.mark.parametrize("mode", ["forest_scalar4", "forest_vector4"])
def test_v2_eight_step_training(mode):
    training_case(mode)


@pytest.mark.parametrize("lengths", [[8, 8], [8, 4], [8, 0]])
def test_reference_finite_behavior_matches_frozen_v0(lengths):
    path = Path("reports/cky_v2/v0_forest_source.py.txt")
    loader = SourceFileLoader("_frozen_cky_v0", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    old = module.SemanticForest(Composer(8, 16), exploration=0.5).train()
    ref = SemanticForest(Composer(8, 16), exploration=0.5, implementation="reference").train()
    ref.load_state_dict(old.state_dict())
    leaves, scores, coeff = torch.randn(2, 8, 8), torch.randn(2, 8, 8), torch.randn(2, 8, 8, 3)
    mask = torch.arange(8)[None, :] < torch.tensor(lengths)[:, None]
    state = torch.get_rng_state()
    a = old(leaves, scores, coeff, mask, True)
    torch.set_rng_state(state)
    b = ref(leaves, scores, coeff, mask, True)
    assert a["trees"] == b["trees"]
    for key in ("semantic", "scores", "vector_coeff", "log_prior", "candidate_mask"):
        x, y = a[key], b[key]
        if key == "scores":
            # V0 left arbitrary values in masked leaf scores. The explicit
            # invalid-state repair canonicalizes these unused fields to zero.
            x, y = x.masked_fill(~a["candidate_mask"], 0), y.masked_fill(~b["candidate_mask"], 0)
        torch.testing.assert_close(x, y, atol=0, rtol=0)


def test_materialization_is_delayed_and_reduced():
    ref = SemanticForest(Composer(16, 32), exploration=0, implementation="reference")
    v2 = SemanticForest(Composer(16, 32), exploration=0, implementation="score_first")
    v2.load_state_dict(ref.state_dict())
    leaves, scores, coeff = torch.randn(2, 16, 16), torch.randn(2, 16, 16), torch.randn(2, 16, 16, 8)
    mask = torch.ones(2, 16, dtype=torch.bool)
    a = ref(leaves, scores, coeff, mask, return_audit=True)["materialization_audit"]
    b = v2(leaves, scores, coeff, mask, return_audit=True)["materialization_audit"]
    assert b["before_selection_semantic"] == b["before_selection_coeff"] == 0
    assert b["total_materialized_semantic"] < a["total_materialized_semantic"]
    assert b["total_materialized_coeff"] < a["total_materialized_coeff"]
    assert b["composer_input_candidate_rows"] == a["composer_input_candidate_rows"]
    assert b["composer_calls"] == a["composer_calls"] == 15
