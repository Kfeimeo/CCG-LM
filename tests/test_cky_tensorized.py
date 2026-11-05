import pytest
import torch
from models.cky_index_cache import CKYIndexCache
from models.forest import SemanticForest
from models.composer import Composer
from legacy.cky_v3.v3_study import forest_pair,model_pair

@pytest.mark.parametrize('words,k,r',[(1,1,4),(4,4,12),(8,2,4),(16,4,12),(24,4,12)])
@pytest.mark.parametrize('policy',['normal','explore','ties','near_ties','prune','empty','partial_inf','all_inf'])
def test_stage_a_search(words,k,r,policy):
    forest_pair(2,words,k,r,True,policy=policy)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('precision',['fp32','bf16'])
@pytest.mark.parametrize('policy',['normal','explore','ties','near_ties','all_inf'])
def test_cuda_stage_a(precision,policy):
    forest_pair(4,24,4,12,True,'cuda',precision,policy)

@pytest.mark.parametrize('mode',['forest_scalar4','forest_vector4'])
def test_stage_a_full_gradient(mode):
    result=model_pair(mode)
    assert result['gradient']['strict_failed_parameter_count']==0

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('precision',['fp32','bf16'])
def test_stage_a_cuda_full_gradient(precision):
    result=model_pair(device='cuda',precision=precision)
    if precision=='fp32':assert result['gradient']['strict_failed_parameter_count']==0

@pytest.mark.parametrize('k,r',[(1,4),(2,4),(4,12),(None,None)])
def test_ragged_index_layout_and_cache(k,r):
    cache=CKYIndexCache();p,hit,_=cache.get(5,2,k,r,'cpu');assert not hit
    assert cache.get(5,2,k,r,'cpu')[0] is p and cache.hits==1
    assert p.slots[1]==1
    for width in range(2,6):
        q=p.widths[width-1]
        assert q.candidates==sum(p.slots[o]*p.slots[width-o] for o in range(1,width))
        assert list(q.metadata)==[(o,l,r) for o in range(1,width) for l in range(p.slots[o]) for r in range(p.slots[width-o])]
        assert q.left.max()<sum((6-o)*p.slots[o] for o in range(1,width))
    cache.clear();assert not cache.entries

def test_score_prior_gradient_and_empty_anchor():
    f=SemanticForest(Composer(8,16),exploration=0,implementation='lazy_tensorized_a')
    h=torch.randn(2,8,8,requires_grad=True);s=torch.randn(2,8,8,requires_grad=True);u=torch.randn(2,8,8,3,requires_grad=True)
    out=f(h,s,u,torch.ones(2,8,dtype=torch.bool));assert out['scores'].grad_fn is not None
    out['log_prior'][:,0].sum().backward();assert s.grad is not None and s.grad.norm()>0
    f.zero_grad(set_to_none=True)
    out=f(h,s,u,torch.zeros(2,8,dtype=torch.bool));out['semantic'].sum().backward()
    assert all(p.grad is not None and not p.grad.any() for p in f.composer.parameters())

def test_bf16_leaf_add_before_float():
    f=SemanticForest(Composer(8,16),implementation='lazy_tensorized_a',exploration=0)
    s=torch.zeros(1,2,2,dtype=torch.bfloat16);s[0,0,0]=1;s[0,1,1]=.0039
    out=f(torch.randn(1,2,8),s,torch.empty(1,2,2,0),torch.ones(1,2,dtype=torch.bool))
    expected=(s[0,0,0]+s[0,1,1]).float()+s[0,0,1]
    assert out['scores'][0,0]==expected
    assert expected!=(s[0,0,0].float()+s[0,1,1].float())


@pytest.mark.parametrize('device,precision',[('cpu','fp32'),('cuda','fp32'),('cuda','bf16')])
def test_native_legacy_width_probe_is_read_only(device,precision):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    from legacy.cky_v3.profile_cky_v3_widths import WidthProbe,legacy_source_probe
    from models.precision_policy import POLICIES
    from utils import set_seed,rng_state,restore_rng
    from legacy.cky_v3.v3_study import assert_rng_equal
    set_seed(42,True);pp=POLICIES['B' if precision=='bf16' else 'E'];pp.configure_backend()
    f=SemanticForest(Composer(8,16),exploration=.5,implementation='lazy_semantic').to(device).train();f.precision_policy=pp
    h=torch.randn(2,8,8,device=device,requires_grad=True);s=torch.randn(2,8,8,device=device,requires_grad=True);u=torch.randn(2,8,8,3,device=device,requires_grad=True)
    if precision=='bf16':s=s.detach().bfloat16().requires_grad_();u=u.detach().bfloat16().requires_grad_()
    mask=torch.ones(2,8,device=device,dtype=torch.bool);initial=rng_state();outs=[];grads=[];ends=[]
    from contextlib import nullcontext
    for active in (False,True):
        f.zero_grad(set_to_none=True)
        for t in (h,s,u):t.grad=None
        restore_rng(initial)
        with (legacy_source_probe(f,WidthProbe(profile=True)) if active else nullcontext()),pp.outer_context(device):out=f(h,s,u,mask)
        (out['semantic'].square().mean()+out['vector_coeff'].square().mean()+out['log_prior'].square().mean()).backward()
        outs.append({k:v.detach().clone() for k,v in out.items() if isinstance(v,torch.Tensor)});ends.append(rng_state())
        grads.append([None if t.grad is None else t.grad.clone() for t in (h,s,u,*f.parameters())])
    assert_rng_equal(*ends)
    assert all(torch.equal(outs[0][k],outs[1][k]) for k in outs[0])
    for x,y in zip(*grads):
        assert (x is None)==(y is None)
        if x is not None:assert torch.equal(x,y)
