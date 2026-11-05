"""Function/compiled width step must reproduce the eager V3 search exactly (identities, RNG, scores)."""
import pytest
import torch
from models.composer import Composer
from models.forest import SemanticForest
from models.cky_compile import width_step,COMPILE_BACKENDS
from legacy.cky_v3.v3_study import compare_search,assert_rng_equal
from utils import rng_state,restore_rng,set_seed

CUDA=torch.cuda.is_available()


def _inputs(device,dtype=torch.float32,n=4,words=9,dim=8,c=3):
    g=torch.Generator().manual_seed(3)
    h=torch.randn(n,words,dim,generator=g).to(device)
    s=(torch.randn(n,words,words,generator=g)*.3).to(device).to(dtype).requires_grad_()
    u=(torch.randn(n,words,words,c,generator=g)*.1).to(device).to(dtype)
    mask=torch.arange(words)[None,:]<torch.tensor([words,words-3,1,0])[:,None]
    return h,s,u,mask.to(device)


def _pair(device,backend,dtype=torch.float32,explore=.5):
    set_seed(21);f=SemanticForest(Composer(8,16),4,12,8.,explore,'lazy_tensorized').to(device).train()
    h,s,u,m=_inputs(device,dtype);state=rng_state();outs=[];grads=[];ends=[]
    for b in ('eager',backend):
        f.search_backend=b;restore_rng(state);s.grad=None
        out=f(h,s,u,m,return_spans=True,return_trees=True);ends.append(rng_state())
        out['log_prior'].sum().backward();grads.append(s.grad.clone());outs.append(out)
    f.search_backend='eager'
    assert_rng_equal(*ends);compare_search(*outs)
    return grads


@pytest.mark.parametrize('explore',[0.,.5])
def test_function_backend_matches_eager_cpu(explore):
    a,b=_pair('cpu','function',explore=explore)
    torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-4)


def test_unknown_backend_rejected():
    f=SemanticForest(Composer(8,16),implementation='lazy_tensorized');f.search_backend='nope'
    h,s,u,m=_inputs('cpu')
    with pytest.raises(ValueError):f(h,s,u,m)
    f.implementation='lazy_tensorized_a';f.search_backend='function'
    with pytest.raises(ValueError):f(h,s,u,m)


@pytest.mark.skipif(not CUDA,reason='CUDA required')
@pytest.mark.parametrize('backend',['function','compile_default','compile_reduce_overhead'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_backends_match_eager_cuda(backend,dtype):
    from models.cky_compile import reset_compile_state
    if backend.startswith('compile'):reset_compile_state()
    a,b=_pair('cuda',backend,dtype)
    atol,rtol=(1.5e-4,.04) if dtype==torch.bfloat16 else (3e-6,3e-4)
    torch.testing.assert_close(a.float(),b.float(),atol=atol,rtol=rtol)
    assert a.dtype==b.dtype==dtype


@pytest.mark.skipif(not CUDA,reason='CUDA required')
def test_inductor_settings_forced():
    from models.cky_compile import get_width_step,reset_compile_state
    import torch._inductor.config as ic,torch._dynamo
    reset_compile_state();get_width_step('compile_default')
    assert ic.fallback_random and ic.emulate_precision_casts and torch._dynamo.config.recompile_limit>=64


@pytest.mark.skipif(not CUDA,reason='CUDA required')
def test_dynamic_rows_do_not_recompile_and_match_eager():
    """Real training changes the row count every update (truncated last micro-batch): the row
    dimension is symbolic, so new row counts reuse the (words, width) graphs and stay identical."""
    from models.cky_compile import reset_compile_state,compile_counters
    reset_compile_state();set_seed(21);f=SemanticForest(Composer(8,16),4,12,8.,.5,'lazy_tensorized').cuda().train()
    graphs=[]
    for n in (4,6,3):
        g=torch.Generator().manual_seed(3+n);words=9
        h=torch.randn(n,words,8,generator=g).cuda();s=(torch.randn(n,words,words,generator=g)*.3).cuda().to(torch.bfloat16).requires_grad_()
        u=(torch.randn(n,words,words,3,generator=g)*.1).cuda().to(torch.bfloat16)
        m=(torch.arange(words)[None,:]<torch.tensor([words,words-3,1,5,2,words][:n])[:,None]).cuda()
        state=rng_state();outs=[];grads=[];ends=[]
        for b in ('eager','compile_default'):
            f.search_backend=b;restore_rng(state);s.grad=None
            out=f(h,s,u,m,return_spans=True,return_trees=True);ends.append(rng_state());out['log_prior'].sum().backward();grads.append(s.grad.clone());outs.append(out)
        f.search_backend='eager';assert_rng_equal(*ends);compare_search(*outs)
        torch.testing.assert_close(grads[0].float(),grads[1].float(),atol=1.5e-4,rtol=.04)
        graphs.append(compile_counters()['stats']['unique_graphs'])
    assert graphs[0]==words-1 and graphs[1]==graphs[0] and graphs[2]==graphs[0],graphs
