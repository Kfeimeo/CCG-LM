import pytest
import torch
from models.model import LatentForestLM
from utils import autocast_context, grad_norm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA-enabled PyTorch installation and GPU")
def test_cuda_sdpa_mixed_precision_forward_backward(tiny_config, batch):
    model = LatentForestLM(tiny_config["model"]).cuda()
    device = torch.device("cuda")
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    with autocast_context(device, "bf16"):
        result = model(**batch)
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    for module in (model.syntax.span_scorer.scalar, model.composer, model.readers):
        assert grad_norm(module) > 0
