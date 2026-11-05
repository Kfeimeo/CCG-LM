"""V3-C score reconstruction: identical search, bitwise roots, strict gradients, no-grad search."""
import copy
import pytest
import torch
from models.composer import Composer
from models.forest import SemanticForest
from models.forest_tensorized import SCORE_RECONSTRUCT
from models.model import LatentForestLM
from models.precision_policy import POLICIES,configure_model
from legacy.cky_v3.v3_study import forest_pair,model_pair,compare_search,assert_rng_equal
from legacy.cky_v25.validate_cky_v25 import small_config,model_batch
from utils import rng_state,restore_rng,set_seed

CUDA=torch.cuda.is_available()


@pytest.mark.parametrize('words,k,r',[(1,1,4),(2,1,4),(4,4,12),(8,2,4),(16,4,12),(24,4,12)])
@pytest.mark.parametrize('policy',['normal','explore','ties','near_ties','prune','empty','partial_inf','all_inf'])
def test_search_identity_against_legacy(words,k,r,policy):
    forest_pair(2,words,k,r,True,policy=policy,implementation=SCORE_RECONSTRUCT)


@pytest.mark.skipif(not CUDA,reason='CUDA required')
@pytest.mark.parametrize('precision',['fp32','bf16'])
@pytest.mark.parametrize('policy',['normal','explore','ties','near_ties','prune','all_inf'])
def test_search_identity_cuda(precision,policy):
    forest_pair(4,24,4,12,True,'cuda',precision,policy,SCORE_RECONSTRUCT)


def _forest_inputs(n=4,words=9,dim=8,c=3,device='cpu',dtype=torch.float32,seed=5):
    g=torch.Generator().manual_seed(seed)
    h=torch.randn(n,words,dim,generator=g).to(device).requires_grad_()
    s=(torch.randn(n,words,words,generator=g)*.3).to(device).to(dtype).requires_grad_()
    u=(torch.randn(n,words,words,c,generator=g)*.1).to(device).to(dtype).requires_grad_()
    lengths=[words,words-2,1,0][:n];mask=torch.arange(words)[None,:]<torch.tensor(lengths)[:,None]
    return h,s,u,mask.to(device)


def _run(f,impl,inputs,mode='dag',weights=None):
    h,s,u,m=inputs;f.implementation=impl;f.reconstruction_mode=mode
    for t in (h,s,u):t.grad=None
    f.zero_grad(set_to_none=True)
    out=f(h,s,u,m,return_spans=True,return_trees=True);end=rng_state()
    weights=torch.randn_like(out['log_prior']) if weights is None else weights
    # Upstream weights emulate Reader gradient: prior, semantic and coefficients all receive signal.
    (out['log_prior']*weights).sum().backward(retain_graph=True)
    (out['semantic'].square().sum()+out['vector_coeff'].square().sum()).backward()
    grads=[None if t.grad is None else t.grad.clone() for t in (h,s,u,*f.parameters())]
    return out,grads,weights,end


@pytest.mark.parametrize('explore',[0.,.5])
def test_dag_matches_v3_outputs_and_gradients(explore):
    set_seed(11);f=SemanticForest(Composer(8,16),4,12,8.,explore,'lazy_tensorized').train()
    inputs=_forest_inputs();state=rng_state()
    restore_rng(state);ref,ref_grads,weights,end_ref=_run(f,'lazy_tensorized',inputs)
    restore_rng(state);new,new_grads,_,end_new=_run(f,SCORE_RECONSTRUCT,inputs,weights=weights)
    assert_rng_equal(end_ref,end_new);compare_search(ref,new)
    assert new['scores'].grad_fn is not None and new['log_prior'].grad_fn is not None
    assert ref['trees']==new['trees']
    for a,b in zip(ref_grads,new_grads):
        assert (a is None)==(b is None)
        if a is not None:torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-4)
    assert new_grads[1].norm()>0  # span potentials receive the prior gradient
    assert f.last_workload['score_reconstruction_nodes']==f.last_workload['v25_unique_dag_nodes']


def test_tree_reference_matches_dag():
    set_seed(12);f=SemanticForest(Composer(8,16),4,12,8.,0.,'lazy_tensorized').train();inputs=_forest_inputs(words=7)
    dag,dag_grads,weights,_=_run(f,SCORE_RECONSTRUCT,inputs,'dag')
    tree,tree_grads,_,_=_run(f,SCORE_RECONSTRUCT,inputs,'tree',weights)
    compare_search(dag,tree)
    for a,b in zip(dag_grads,tree_grads):
        if a is not None:torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-4)
    # Occurrence expansion never has fewer nodes than the shared DAG.
    assert f.last_workload['score_reconstruction_nodes']>=f.last_workload['v25_unique_dag_nodes']


def test_linear_mode_is_flagged_not_bitwise():
    set_seed(13);f=SemanticForest(Composer(8,16),4,12,8.,0.,'lazy_tensorized').train();inputs=_forest_inputs(words=8)
    dag,dag_grads,weights,_=_run(f,SCORE_RECONSTRUCT,inputs,'dag')
    lin,lin_grads,_,_=_run(f,SCORE_RECONSTRUCT,inputs,'linear',weights)
    assert torch.equal(dag['candidate_mask'],lin['candidate_mask'])
    torch.testing.assert_close(dag['scores'],lin['scores'],atol=1e-5,rtol=1e-5)  # same sum, different order
    torch.testing.assert_close(dag_grads[1],lin_grads[1],atol=1e-5,rtol=1e-4)
    f.reconstruction_mode='dag'


def test_search_runs_without_autograd_graph():
    set_seed(14);f=SemanticForest(Composer(8,16),4,12,8.,0.,SCORE_RECONSTRUCT).train();h,s,u,m=_forest_inputs(words=6)
    captured={}
    original=f._search_scores
    def spy(*a,**k):
        out=original(*a,**k);captured['root_s']=out['root_s'];captured['bands']=out['bands'];return out
    f._search_scores=spy
    try:out=f(h,s,u,m)
    finally:del f._search_scores
    assert captured['root_s'].grad_fn is None and not captured['root_s'].requires_grad
    assert out['scores'].requires_grad and out['log_prior'].requires_grad
    assert torch.equal(out['scores'],captured['root_s'])


def test_bf16_width2_leaf_addition_is_preserved():
    f=SemanticForest(Composer(8,16),implementation=SCORE_RECONSTRUCT,exploration=0)
    s=torch.zeros(1,2,2,dtype=torch.bfloat16);s[0,0,0]=1;s[0,1,1]=.0039;s.requires_grad_()
    out=f(torch.randn(1,2,8),s,torch.empty(1,2,2,0),torch.ones(1,2,dtype=torch.bool))
    expected=(s[0,0,0]+s[0,1,1]).float()+s[0,0,1]
    assert out['scores'][0,0]==expected and expected!=(s[0,0,0].float()+s[0,1,1].float())
    out['scores'][0,0].backward();assert s.grad.dtype==torch.bfloat16 and s.grad[0,0,0]==1 and s.grad[0,1,1]==1 and s.grad[0,0,1]==1


def test_shared_subtree_gradient_accumulates():
    # Two root candidates sharing a child: the shared span potential must receive both contributions.
    f=SemanticForest(Composer(4,8),k=2,max_compositions=12,delta=None,exploration=0,implementation=SCORE_RECONSTRUCT)
    s=torch.zeros(1,3,3,requires_grad=True);h=torch.randn(1,3,4);u=torch.empty(1,3,3,0)
    with torch.no_grad():s[0,0,1]=2.;s[0,1,2]=1.
    out=f(h,s,u,torch.ones(1,3,dtype=torch.bool))
    assert out['candidate_mask'][0].sum()==2
    out['scores'][0].sum().backward()
    # Both derivations contain the root span (0,3) and all three leaves; each contains one width-2 span.
    assert s.grad[0,0,2]==2 and s.grad[0,0,0]==2 and s.grad[0,1,1]==2 and s.grad[0,2,2]==2
    assert s.grad[0,0,1]==1 and s.grad[0,1,2]==1


@pytest.mark.parametrize('mode',['forest_scalar4','forest_vector4'])
def test_full_lm_gradients_cpu(mode):
    z=model_pair(mode,implementation=SCORE_RECONSTRUCT)
    assert z['gradient']['strict_failed_parameter_count']==0,z['gradient']['strict_failed_parameters']
    assert z['gradient']['modules']['span_scorer']['gradient_norm_lazy']>0


@pytest.mark.skipif(not CUDA,reason='CUDA required')
@pytest.mark.parametrize('precision',['fp32','bf16'])
def test_full_lm_gradients_cuda(precision):
    z=model_pair(device='cuda',precision=precision,implementation=SCORE_RECONSTRUCT)
    if precision=='fp32':assert z['gradient']['strict_failed_parameter_count']==0,z['gradient']['strict_failed_parameters']
    assert z['gradient']['modules']['span_scorer']['gradient_norm_lazy']>0


def test_unbounded_exact_mode_falls_back_to_legacy_search():
    f=SemanticForest(Composer(8,16),k=None,max_compositions=None,delta=None,exploration=0,implementation=SCORE_RECONSTRUCT).train()
    h=torch.randn(2,5,8);s=torch.randn(2,5,5,requires_grad=True);u=torch.randn(2,5,5,3);m=torch.ones(2,5,dtype=torch.bool)
    out=f(h,s,u,m,return_trees=True);assert len(out['trees'][0])==14 and out['scores'].requires_grad
    assert f.last_workload.get('search_compatibility_fallback')=='exact_unbounded'


def test_state_dict_and_checkpoint_field(tiny_config,batch):
    cfg=copy.deepcopy(tiny_config['model']);cfg['forest_implementation']=SCORE_RECONSTRUCT
    model=LatentForestLM(cfg);keys=list(model.state_dict())
    reference=LatentForestLM(dict(cfg,forest_implementation='lazy_tensorized'));assert keys==list(reference.state_dict())
    assert model.forest.implementation==SCORE_RECONSTRUCT
    out=model(**batch);out['loss'].backward();assert all(p.grad is not None for n,p in model.syntax.span_scorer.named_parameters())
