import copy
import pytest
import torch
from models.precision_policy import POLICIES,configure_model
from models.model import LatentForestLM
from train import make_optimizer,make_schedule,train_update,save_checkpoint,load_checkpoint
from utils import rng_state,restore_rng
from legacy.precision_v25.precision_study import search_equal

@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
@pytest.mark.parametrize("group",list("ABCDES"))
def test_fixed_weights_search_identity(group,tiny_config,batch):
    previous=(torch.backends.fp32_precision,torch.backends.cuda.matmul.fp32_precision)
    try:
        p=POLICIES[group];p.configure_backend();m=configure_model(LatentForestLM(tiny_config["model"]).cuda().train(),p)
        m.forest.capture_spans=True;batch={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in batch.items()};state=rng_state();outs=[]
        for impl in ("score_first","lazy_semantic"):
            m.forest.implementation=impl;restore_rng(state)
            with p.outer_context("cuda"):out=m(**batch)
            outs.append(out)
        assert search_equal(outs[0]["memory"],outs[1]["memory"])["identical"]
        torch.testing.assert_close(outs[0]["loss"],outs[1]["loss"],atol=.001,rtol=.001)
    finally:
        torch.backends.fp32_precision,torch.backends.cuda.matmul.fp32_precision=previous

@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
def test_gradscaler_resume_and_policy_guard(tiny_config,batch,tmp_path):
    previous=(torch.backends.fp32_precision,torch.backends.cuda.matmul.fp32_precision)
    try:
        policy=POLICIES["D"];policy.configure_backend();cfg=copy.deepcopy(tiny_config);cfg["precision"]=policy.state_dict()
        batch={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in batch.items()};tokens=int(batch["labels"].ne(-100).sum())
        class Sampler:
            def batch(self,n,t):assert t==tokens;return batch,tokens
            def state_dict(self):return {}
            def load_state_dict(self,s):pass
        sampler=Sampler();a=configure_model(LatentForestLM(cfg["model"]).cuda(),policy)
        oa=make_optimizer(a,cfg["train"]);sa=make_schedule(oa,cfg["train"]);ga=policy.make_scaler()
        first=train_update(a,sampler,oa,sa,tokens,2,torch.device("cuda"),"bf16",1.,scaler=ga)
        assert not first["skipped_step"]
        path=tmp_path/"checkpoint.pt";save_checkpoint(path,a,oa,sa,sampler,cfg,{"step":1},"fixed",scaler=ga)
        second=train_update(a,sampler,oa,sa,tokens,2,torch.device("cuda"),"bf16",1.,scaler=ga)
        b=configure_model(LatentForestLM(cfg["model"]).cuda(),policy);ob=make_optimizer(b,cfg["train"]);sb=make_schedule(ob,cfg["train"]);gb=policy.make_scaler()
        load_checkpoint(path,b,ob,sb,sampler,cfg,"fixed",scaler=gb)
        replay=train_update(b,sampler,ob,sb,tokens,2,torch.device("cuda"),"bf16",1.,scaler=gb)
        assert ga.state_dict()==gb.state_dict()
        for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-5)
        bad=copy.deepcopy(cfg);bad["precision"]=POLICIES["B"].state_dict()
        with pytest.raises(ValueError,match="precision policy"):load_checkpoint(path,b,cfg=bad)
    finally:
        torch.backends.fp32_precision,torch.backends.cuda.matmul.fp32_precision=previous


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
@pytest.mark.parametrize("group",list("ABCDE"))
@pytest.mark.parametrize("edge",["empty","unreachable","partial","ties"])
def test_precision_backward_edge_reachability(group,edge):
    from models.composer import Composer
    from models.forest import SemanticForest
    from legacy.precision_v25.precision_study import errors
    previous=(torch.backends.fp32_precision,torch.backends.cuda.matmul.fp32_precision)
    try:
        policy=POLICIES[group];policy.configure_backend()
        f=SemanticForest(Composer(8,16),k=4,max_compositions=12,delta=0 if edge=="ties" else 8,exploration=.1).cuda().train()
        f.precision_policy=policy
        h=torch.randn(2,8,8,device="cuda",requires_grad=True)
        potential=torch.randn(2,8,8,device="cuda")*.1
        if edge=="unreachable":potential.fill_(-torch.inf)
        if edge=="partial":potential[:,0,1]=-torch.inf
        if edge=="ties":potential.zero_()
        scores=potential.requires_grad_();u=torch.randn(2,8,8,2,device="cuda",requires_grad=True)
        mask=torch.ones(2,8,device="cuda",dtype=torch.bool);mask[1,1:]=False
        if edge=="empty":mask.zero_()
        state=rng_state();snapshots=[];outs=[]
        scale=policy.scaler_init_scale if group=="D" else 1.
        for impl in ("score_first","lazy_semantic"):
            f.implementation=impl;f.zero_grad(set_to_none=True)
            for x in (h,scores,u):x.grad=None
            restore_rng(state)
            with policy.outer_context("cuda"):
                out=f(h,scores,u,mask,return_spans=True)
                loss=out["semantic"].square().mean()+out["vector_coeff"].square().mean()+out["log_prior"].square().mean()
            (loss*scale).backward()
            snapshots.append([None if p.grad is None else p.grad.detach().float().cpu()/scale for p in [h,scores,u,*f.parameters()]])
            outs.append(out)
        assert search_equal(outs[0],outs[1])["identical"]
        for a,b in zip(*snapshots):
            assert (a is None)==(b is None)
            if a is not None:
                assert torch.isfinite(a).all() and torch.isfinite(b).all()
                if group=="E":torch.testing.assert_close(a,b,atol=3e-6,rtol=.0003)
        # Do not relax original V2 elementwise tolerances here: mixed-precision
        # errors are measured by the study; this test checks finite reachability.
    finally:
        torch.backends.fp32_precision,torch.backends.cuda.matmul.fp32_precision=previous
