import pytest
import torch
from models.composer import Composer
from models.forest import SemanticForest
from models.precision_policy import PrecisionPolicy,POLICIES,configure_model
from models.model import LatentForestLM
from utils import restore_rng,rng_state

@pytest.fixture(autouse=True)
def restore_backend():
    generic=torch.backends.fp32_precision;matmul=torch.backends.cuda.matmul.fp32_precision
    yield
    torch.backends.fp32_precision=generic;torch.backends.cuda.matmul.fp32_precision=matmul


def test_policy_validation():
    with pytest.raises(ValueError):PrecisionPolicy(composer="fp32_tf32")
    with pytest.raises(ValueError):PrecisionPolicy(score_dtype="fp64")
    with pytest.raises(ValueError):PrecisionPolicy(composer="unknown")


def test_fp32_policy_keeps_weights_and_math(tiny_config,batch):
    a=LatentForestLM(tiny_config["model"]).eval();b=LatentForestLM(tiny_config["model"]).eval();b.load_state_dict(a.state_dict())
    policy=POLICIES["B"];policy.configure_backend();configure_model(b,policy)
    assert set(a.state_dict())==set(b.state_dict())
    x,y=a(**batch),b(**batch)
    torch.testing.assert_close(x["logits"],y["logits"],atol=1e-6,rtol=1e-5)
    assert all(p.dtype==torch.float32 for p in b.parameters())
    with pytest.raises(ValueError):configure_model(b,POLICIES["D"])

@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
@pytest.mark.parametrize("group",list("ABCDE"))
def test_real_cuda_dtypes(group,tiny_config,batch):
    policy=POLICIES[group];policy.configure_backend()
    model=configure_model(LatentForestLM(tiny_config["model"]).cuda(),policy)
    model.forest.capture_spans=True;model.forest.implementation="lazy_semantic"
    traces=[];states=[]
    hook=model.composer.mlp[0].register_forward_hook(lambda m,i,o:traces.append(o.dtype))
    model.composer.precision_state_trace=lambda m,o:states.append(o.dtype)
    batch={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
    with policy.outer_context("cuda"):out=model(**batch)
    expected={"A":torch.bfloat16,"B":torch.float32,"C":torch.float32,"D":torch.float16,"E":torch.float32}[group]
    assert set(traces)=={expected}
    assert set(states)=={torch.float16 if group=="D" else torch.float32}
    assert out["memory"]["semantic"].dtype==torch.float32
    scaler=policy.make_scaler();scaler.scale(out["loss"]).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    hook.remove()


def test_master_weight_guard(tiny_config):
    model=LatentForestLM(tiny_config["model"]).half()
    with pytest.raises(ValueError,match="master"):configure_model(model,POLICIES["B"])
