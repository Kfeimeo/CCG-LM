"""V2.5 invariants: exact discrete search and differentiable shared closures."""
import pytest
import torch
from models.composer import Composer
from models.forest import SemanticForest
from models.forest_lazy import ScoreCell
from utils import rng_state,restore_rng
from validate_cky_v25 import forest_case,model_gradient_case

@pytest.mark.parametrize("words,k,r",[(1,1,4),(4,4,12),(8,2,4),(16,4,12),(24,4,12)])
@pytest.mark.parametrize("policy",["normal","explore","ties","prune","empty","partial_inf","unreachable"])
def test_search_and_roots(words,k,r,policy):
    if words == 1 and policy == "partial_inf":
        pytest.skip("No two-word subspan in a one-word sentence")
    forest_case(2,words,k,r,True,policy=policy)

@pytest.mark.parametrize("mode",["forest_scalar4","forest_vector4"])
def test_full_model_gradients(mode):
    model_gradient_case(mode)

@pytest.mark.parametrize("empty",[False,True])
def test_shared_dag_gradients_and_zero_reachability(empty):
    f=SemanticForest(Composer(8,16),exploration=0).train()
    h=torch.randn(2,8,8,requires_grad=True)
    s=(torch.randn(2,8,8)*.1).requires_grad_()
    u=torch.randn(2,8,8,3,requires_grad=True)
    mask=torch.ones(2,8,dtype=torch.bool)
    if empty: mask.zero_()
    grads=[];outputs=[]
    for impl in ("score_first","lazy_semantic"):
        f.implementation=impl;f.zero_grad(set_to_none=True)
        for x in (h,s,u): x.grad=None
        out=f(h,s,u,mask,return_spans=True)
        (out["semantic"].square().sum()+out["vector_coeff"].square().sum()+out["log_prior"].square().sum()).backward()
        grads.append([None if p.grad is None else p.grad.clone() for p in [h,s,u,*f.parameters()]])
        outputs.append(out)
    for a,b in zip(*grads):
        assert (a is None)==(b is None)
        if a is not None: torch.testing.assert_close(a,b,atol=5e-5,rtol=5e-4)
    work=f.last_workload
    assert work["v25_actual_composer_rows"] <= work["v25_required_node_occurrences"]
    assert work["v25_composer_calls"] <= 7
    if empty: assert work["v25_actual_composer_rows"]==0
    else: assert work["sharing_ratio"]>0
    assert "semantic" not in ScoreCell.__annotations__
    assert all("semantic" not in v and "vector_coeff" not in v for v in outputs[1]["spans"].values())


def test_guard_stochastic_composer_and_state_dict():
    f=SemanticForest(Composer(8,16),implementation="lazy_semantic")
    original=set(f.state_dict())
    f.implementation="score_first"
    assert set(f.state_dict())==original
    f.implementation="lazy_semantic"
    f.composer.mlp.append(torch.nn.Dropout(.2))
    with pytest.raises(ValueError,match="deterministic"):
        f(torch.randn(1,4,8),torch.randn(1,4,4),torch.randn(1,4,4,0),torch.ones(1,4,dtype=torch.bool))


@pytest.mark.parametrize("batch",[1,4])
def test_candidate_ids_across_batch_rows(batch):
    forest_case(batch,24,4,12,True,policy="explore")


def test_exact_unbounded_lazy():
    f=SemanticForest(Composer(8,16),k=None,max_compositions=None,delta=None,exploration=0).eval()
    h=torch.randn(1,5,8);s=torch.randn(1,5,5);u=torch.randn(1,5,5,2);m=torch.ones(1,5,dtype=torch.bool)
    a=f(h,s,u,m,return_trees=True)
    f.implementation="lazy_semantic"; b=f(h,s,u,m,return_trees=True)
    assert len(b["trees"][0])==14
    assert a["trees"]==b["trees"]
    for key in ("scores","semantic","vector_coeff","log_prior"):
        torch.testing.assert_close(a[key],b[key],atol=3e-6,rtol=3e-5)
