"""Independent tree enumeration checks both multiplicity and local-potential sums."""
from functools import lru_cache
import pytest
import torch
from models.composer import Composer
from models.forest import SemanticForest, catalan, masked_log_softmax


@lru_cache(None)
def enumerate_trees(i, j):
    if j - i == 1:
        return (i,)
    return tuple((left, right) for k in range(i + 1, j)
                 for left in enumerate_trees(i, k) for right in enumerate_trees(k, j))


def enumerate_score(tree, scores):
    if isinstance(tree, int):
        return scores[tree, tree], tree, tree + 1
    left, i, k = enumerate_score(tree[0], scores)
    right, _, j = enumerate_score(tree[1], scores)
    return left + right + scores[i, j - 1], i, j


@pytest.mark.parametrize("length", [3, 4, 5, 6])
def test_exact_cky_matches_all_trees(length):
    comp = Composer(8, 16)
    forest = SemanticForest(comp, k=None, max_compositions=None, delta=None, exploration=0).eval()
    leaves = torch.randn(1, length, 8)
    scores = torch.randn(1, length, length)
    coeff = torch.randn(1, length, length, 3)
    result = forest(leaves, scores, coeff, torch.ones(1, length, dtype=torch.bool), return_trees=True)
    expected_trees = enumerate_trees(0, length)
    assert result["candidate_mask"].sum() == len(expected_trees) == catalan(length - 1)
    assert set(result["trees"][0]) == set(expected_trees)
    expected = torch.stack([enumerate_score(tree, scores[0])[0] for tree in expected_trees]).sort().values
    actual = result["scores"][result["candidate_mask"]].sort().values
    torch.testing.assert_close(expected, actual, atol=2e-6, rtol=1e-5)
    # Coefficient sums use exactly the same selected derivation.
    for slot, tree in enumerate(result["trees"][0]):
        target = torch.stack([enumerate_score(tree, coeff[0, :, :, c])[0] for c in range(3)])
        torch.testing.assert_close(result["vector_coeff"][0, slot], target)


def test_asymmetry():
    comp = Composer(16, 32)
    a, b = torch.randn(4, 16), torch.randn(4, 16)
    assert not torch.allclose(comp(a, b), comp(b, a))


def test_pruning_normalization_empty_and_variable_lengths():
    forest = SemanticForest(Composer(8, 16), k=4, max_compositions=3, delta=0.01, exploration=0.2).train()
    leaves = torch.randn(3, 6, 8)
    scores = torch.randn(3, 6, 6) * 20
    mask = torch.arange(6)[None, :] < torch.tensor([6, 3, 0])[:, None]
    result = forest(leaves, scores, torch.zeros(3, 6, 6, 0), mask)
    assert result["candidate_mask"][:2].any(-1).all()
    assert not result["candidate_mask"][2].any()
    assert torch.isfinite(result["semantic"]).all() and torch.isfinite(result["log_prior"]).all()
    probabilities = result["log_prior"].exp() * result["candidate_mask"]
    torch.testing.assert_close(probabilities.sum(-1), torch.tensor([1.0, 1.0, 0.0]))


def test_hard_topk_has_no_ordinary_gradient_to_discarded_scores():
    scores = torch.tensor([1.0, 2.0, -1.0], requires_grad=True)
    chosen = scores.topk(2).values
    loss = (chosen.softmax(-1) * torch.tensor([0.3, 0.8])).sum()
    loss.backward()
    assert scores.grad[2] == 0
    assert scores.grad[:2].abs().min() > 0
    one = torch.tensor([[3.0]], requires_grad=True)
    masked_log_softmax(one, torch.ones_like(one, dtype=torch.bool)).sum().backward()
    assert one.grad.item() == 0  # Softmax singleton cannot train the syntax scorer.


def test_padding_does_not_change_forest():
    forest = SemanticForest(Composer(8, 16), exploration=0).eval()
    leaves, scores, coeff = torch.randn(1, 4, 8), torch.randn(1, 4, 4), torch.randn(1, 4, 4, 3)
    a = forest(leaves, scores, coeff, torch.ones(1, 4, dtype=torch.bool))
    b = forest(torch.nn.functional.pad(leaves, (0, 0, 0, 2)), torch.nn.functional.pad(scores, (0, 2, 0, 2)),
               torch.nn.functional.pad(coeff, (0, 0, 0, 2, 0, 2)), torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool))
    for key in ("semantic", "log_prior", "vector_coeff", "candidate_mask"):
        torch.testing.assert_close(a[key], b[key])


def test_exploration_reaches_retained_beam_and_respects_expansion_budget():
    forest = SemanticForest(Composer(8, 16), k=4, max_compositions=12, delta=8.0, exploration=1.0).train()
    leaves, scores = torch.randn(1, 6, 8), torch.randn(1, 6, 6) * 0.1
    observed = []
    hook = forest.composer.register_forward_pre_hook(lambda module, args: observed.append(args[0].shape[2]))
    selections = set()
    for seed in range(6):
        torch.manual_seed(seed)
        result = forest(leaves, scores, torch.zeros(1, 6, 6, 0), torch.ones(1, 6, dtype=torch.bool), True)
        selections.add(frozenset(result["trees"][0]))
    hook.remove()
    assert max(observed) <= 12
    assert len(selections) > 1
