import copy
import pytest
import torch
from models.reader import SemanticReader
from models.model import LatentForestLM
from models.precision_policy import PrecisionPolicy,configure_model
from legacy.reader_v25.reader_precision_study import READER_POLICIES,ReaderObserver
from utils import rng_state,restore_rng

@pytest.fixture(autouse=True)
def restore_backend():
    before=(torch.backends.fp32_precision,torch.backends.cuda.matmul.fp32_precision)
    yield
    torch.backends.fp32_precision,torch.backends.cuda.matmul.fp32_precision=before


def test_policy_and_parameter_compatibility(tiny_config):
    with pytest.raises(ValueError):PrecisionPolicy(reader="fp32_ieee",composer="fp32_tf32",matmul_precision="tf32")
    with pytest.raises(ValueError):PrecisionPolicy(reader="bad")
    a=LatentForestLM(tiny_config["model"]);b=LatentForestLM(tiny_config["model"])
    b.load_state_dict(a.state_dict());configure_model(b,READER_POLICIES["R"])
    assert list(a.state_dict())==list(b.state_dict())
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())

@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
@pytest.mark.parametrize("family",["B","R","E"])
def test_native_attention_probe_is_read_only(family,tiny_config):
    policy=READER_POLICIES[family];policy.configure_backend();cfg=tiny_config["model"];cfg["dropout"]=.1
    r=SemanticReader(cfg,True).cuda().train();r.precision_mode=policy.reader
    hidden=torch.randn(2,5,32,device="cuda",requires_grad=True)
    semantic=torch.randn(2,2,4,16,device="cuda",requires_grad=True)
    coeff=torch.randn(2,2,4,8,device="cuda",dtype=torch.bfloat16).requires_grad_()
    prior=torch.randn(2,2,4,device="cuda",requires_grad=True)
    mask=torch.ones(2,2,4,device="cuda",dtype=torch.bool);mask[-1].zero_()
    memory=dict(semantic=semantic,vector_coeff=coeff,log_prior=prior,candidate_mask=mask,
                history_mask=torch.ones(2,2,device="cuda",dtype=torch.bool))
    initial=rng_state();snapshots=[];outputs=[];ends=[]
    for enabled in (False,True):
        r.zero_grad(set_to_none=True)
        for t in [hidden,semantic,coeff,prior]:t.grad=None
        restore_rng(initial);obs=ReaderObserver();r.precision_observer=obs if enabled else None
        with policy.outer_context("cuda"):out=r(hidden,memory)[0]
        out.square().mean().backward();outputs.append(out.detach().clone());ends.append(rng_state())
        snapshots.append([None if t.grad is None else t.grad.detach().clone() for t in [hidden,semantic,coeff,prior,*r.parameters()]])
    assert torch.equal(outputs[0],outputs[1])
    assert all(torch.equal(a,b) for a,b in zip(ends[0]["cuda"],ends[1]["cuda"]))
    for a,b in zip(*snapshots):
        assert (a is None)==(b is None)
        if a is not None:assert torch.equal(a,b)
    assert obs.specs["attention_logits"]["actual_dtype"]=="torch.float32"
    assert obs.specs["attention_probabilities"]["actual_dtype"]=="torch.float32"
    expected="torch.bfloat16" if family=="B" else "torch.float32"
    for name in ["query_projection","key_projection","value_projection","attention_weighted_values","output_projection"]:
        assert obs.specs[name]["actual_dtype"]==expected
    assert out.dtype==hidden.dtype
    torch.testing.assert_close(out[-1],hidden[-1],atol=0,rtol=0)

@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
def test_boundary_cast_is_differentiable(tiny_config):
    p=READER_POLICIES["R"];p.configure_backend();r=SemanticReader(tiny_config["model"],False).cuda();r.precision_mode=p.reader
    h=torch.randn(1,3,32,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    sem=torch.randn(1,1,4,16,device="cuda",requires_grad=True)
    mem=dict(semantic=sem,vector_coeff=sem.new_empty(1,1,4,0),log_prior=sem.new_zeros(1,1,4),candidate_mask=torch.ones(1,1,4,device="cuda",dtype=torch.bool),history_mask=torch.ones(1,1,device="cuda",dtype=torch.bool))
    with p.outer_context("cuda"):out=r(h,mem)[0]
    assert out.dtype==torch.bfloat16
    out.float().square().sum().backward()
    assert h.grad is not None and sem.grad is not None and torch.isfinite(sem.grad).all()



def test_legacy_precision_checkpoint_normalizes_reader_default(tiny_config,tmp_path):
    from train import make_optimizer,make_schedule,save_checkpoint,load_checkpoint
    class Sampler:
        def state_dict(self):return {}
    cfg=copy.deepcopy(tiny_config);cfg["precision"]=READER_POLICIES["B"].state_dict()
    m=configure_model(LatentForestLM(cfg["model"]),READER_POLICIES["B"])
    opt=make_optimizer(m,cfg["train"]);schedule=make_schedule(opt,cfg["train"])
    path=tmp_path/"legacy.pt"
    save_checkpoint(path,m,opt,schedule,Sampler(),cfg,{"step":0},"fixture")
    data=torch.load(path,map_location="cpu",weights_only=False)
    data["config"]["precision"].pop("reader");data["precision_policy"].pop("reader")
    torch.save(data,path)
    replay=configure_model(LatentForestLM(cfg["model"]),READER_POLICIES["B"])
    load_checkpoint(path,replay,cfg=data["config"],fingerprint="fixture",restore_random=False)
    for a,b in zip(m.parameters(),replay.parameters()):torch.testing.assert_close(a,b,atol=0,rtol=0)
    changed=copy.deepcopy(cfg);changed["precision"]=READER_POLICIES["R"].state_dict()
    with pytest.raises(ValueError,match="precision policy"):
        load_checkpoint(path,replay,cfg=changed,restore_random=False)
