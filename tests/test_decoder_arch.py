"""Decoder architecture presets (gpt2 / llama2 / llama3 / qwen3) behind one interface."""
import pytest
import torch

from models.decoder import Decoder
from models.decoder_llama import DECODER_ARCHS, LlamaDecoder, build_decoder, resolve_arch
from models.model import LatentForestLM, parameter_report
from utils import grad_norm

ARCHS = [a for a in DECODER_ARCHS if a != "gpt2"]


def _cfg(tiny_config, arch, **extra):
    cfg = dict(tiny_config["model"], decoder_arch=arch, ffn_size=48, **extra)
    return cfg


def test_default_arch_is_unchanged_gpt_decoder(tiny_config):
    model = LatentForestLM(tiny_config["model"])
    assert isinstance(model.decoder, Decoder)
    assert resolve_arch(tiny_config["model"]) == {"arch": "gpt2"}
    assert any(k.startswith("decoder.position.") for k in model.state_dict())


def test_presets_and_overrides(tiny_config):
    base = tiny_config["model"]
    assert resolve_arch(dict(base, decoder_arch="llama2"))["kv_heads"] == base["heads"]
    assert resolve_arch(dict(base, decoder_arch="llama3"))["kv_heads"] == base["heads"] // 4
    q = resolve_arch(dict(base, decoder_arch="qwen3"))
    assert q["kv_heads"] == base["heads"] // 2 and q["qk_norm"] and q["tie_embeddings"]
    o = resolve_arch(dict(base, decoder_arch="qwen3", kv_heads=1, rope_theta=123.0, qk_norm=False, tie_embeddings=False))
    assert (o["kv_heads"], o["rope_theta"], o["qk_norm"], o["tie_embeddings"]) == (1, 123.0, False, False)
    with pytest.raises(ValueError):
        resolve_arch(dict(base, decoder_arch="mamba"))
    with pytest.raises(ValueError):
        resolve_arch(dict(base, decoder_arch="llama3", kv_heads=3))  # 4 heads not divisible by 3


@pytest.mark.parametrize("arch", ARCHS)
def test_forward_backward_step_eval_with_memory(arch, tiny_config, batch):
    model = LatentForestLM(_cfg(tiny_config, arch))
    assert isinstance(model.decoder, LlamaDecoder)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    result = model(**batch)
    assert result["logits"].shape == (*batch["input_ids"].shape, tiny_config["model"]["vocab_size"])
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    for module in (model.decoder, model.syntax.span_scorer.scalar, model.composer, model.readers):
        assert grad_norm(module) > 0
    opt.step()
    model.eval()
    with torch.no_grad():
        assert torch.isfinite(model(**batch)["loss"])
        assert torch.isfinite(model(**batch, ablation="forest_top1_eval")["loss"])


@pytest.mark.parametrize("arch", ARCHS)
def test_causality_and_memory_effect(arch, tiny_config, batch):
    model = LatentForestLM(_cfg(tiny_config, arch)).eval()
    original = model(**batch)["logits"]
    changed = dict(batch, input_ids=batch["input_ids"].clone())
    changed["input_ids"][:, 2:] = 32
    newer = model(**changed)["logits"]
    torch.testing.assert_close(original[:, :2], newer[:, :2], rtol=0, atol=0)
    assert not torch.allclose(original[:, 2:], newer[:, 2:])
    off = model(**batch, memory_off=True)["logits"]
    assert not torch.allclose(off, original)


@pytest.mark.parametrize("arch", ARCHS)
def test_rope_is_position_dependent(arch, tiny_config, batch):
    # Shifting a token to another position must change its logits even with memory off.
    cfg = _cfg(tiny_config, arch)
    decoder = build_decoder(cfg).eval()
    ids = torch.tensor([[5, 6, 7, 8]])
    a = decoder(ids)[0]
    b = decoder(torch.tensor([[6, 7, 8, 5]]))[0]
    assert not torch.allclose(a[0, 1], b[0, 0])


def test_state_dict_keys_follow_arch(tiny_config):
    llama2 = build_decoder(_cfg(tiny_config, "llama2")).state_dict()
    qwen3 = build_decoder(_cfg(tiny_config, "qwen3")).state_dict()
    assert "lm_head.weight" in llama2 and "lm_head.weight" not in qwen3
    assert "blocks.0.attention.q_norm.weight" in qwen3 and "blocks.0.attention.q_norm.weight" not in llama2
    assert not any(k.startswith("position.") or ".cos" in k or ".sin" in k for k in llama2)
    assert all(not k.endswith("bias") for k in llama2)
    assert llama2["blocks.0.attention.k_proj.weight"].shape[0] == tiny_config["model"]["hidden_size"]
    llama3 = build_decoder(_cfg(tiny_config, "llama3")).state_dict()
    assert llama3["blocks.0.attention.k_proj.weight"].shape[0] == tiny_config["model"]["hidden_size"] // 4


@pytest.mark.parametrize("arch", ARCHS)
def test_parameter_report_and_optimizer_groups(arch, tiny_config):
    from train import make_optimizer
    model = LatentForestLM(_cfg(tiny_config, arch))
    report = parameter_report(model)
    assert report["modules"]["decoder"] == sum(p.numel() for p in model.decoder.parameters())
    opt = make_optimizer(model, tiny_config["train"])
    decoder_ids = {id(p) for p in model.decoder.parameters()}
    for group in opt.param_groups:
        in_decoder = [id(p) in decoder_ids for p in group["params"]]
        assert all(in_decoder) or not any(in_decoder)
        if all(in_decoder):
            assert group["lr"] == tiny_config["train"]["decoder_lr"]


@pytest.mark.parametrize("arch", ARCHS)
def test_target_length_limit(arch, tiny_config):
    decoder = build_decoder(_cfg(tiny_config, arch)).eval()
    too_long = torch.zeros(1, tiny_config["model"]["max_target_length"] + 1, dtype=torch.long)
    with pytest.raises(ValueError):
        decoder(too_long)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("arch", ARCHS)
def test_cuda_bf16_autocast(arch, tiny_config, batch):
    from utils import to_device
    model = LatentForestLM(_cfg(tiny_config, arch)).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = model(**to_device(batch, torch.device("cuda")))
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert grad_norm(model.decoder) > 0
