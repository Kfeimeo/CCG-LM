import pytest
import torch
from models.model import LatentForestLM, MODES
from utils import grad_norm


@pytest.mark.parametrize("mode", MODES)
def test_all_modes_forward_backward_step_eval(mode, tiny_config, batch):
    tiny_config["model"]["mode"] = mode
    model = LatentForestLM(tiny_config["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    result = model(**batch)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    opt.step()
    model.eval()
    with torch.no_grad():
        assert torch.isfinite(model(**batch)["loss"])
        if mode != "no_memory":
            assert model(**batch)["memory"]["semantic"].shape[:3] == (2, 2, 4)
        if mode.startswith("forest"):
            assert torch.isfinite(model(**batch, ablation="forest_top1_eval")["loss"])


def test_lm_gradient_scorer_composer_reader(tiny_config, batch):
    model = LatentForestLM(tiny_config["model"])
    result = model(**batch)
    assert result["memory"]["candidate_mask"][0, 0].sum() >= 2
    result["memory"]["scores"].retain_grad()
    result["loss"].backward()
    for module in (model.syntax.span_scorer.scalar, model.composer, model.readers):
        norm = grad_norm(module)
        assert norm > 0 and torch.isfinite(torch.tensor(norm))


def test_multiple_root_prior_path_trains_scalar_without_coeff(tiny_config, batch):
    tiny_config["model"]["mode"] = "forest_scalar4"
    model = LatentForestLM(tiny_config["model"])
    model(**batch)["loss"].backward()
    assert grad_norm(model.syntax.span_scorer.scalar) > 1e-10


def test_memory_mask_padding_and_off(tiny_config, batch):
    model = LatentForestLM(tiny_config["model"]).eval()
    original = model(**batch)["logits"]
    changed = dict(batch)
    changed["history_ids"] = batch["history_ids"].clone()
    changed["history_ids"][~batch["word_mask"]] = 63
    torch.testing.assert_close(original, model(**changed)["logits"])
    padded = dict(batch)
    padded["history_ids"] = torch.nn.functional.pad(batch["history_ids"], (0, 2, 0, 1), value=31)
    padded["word_mask"] = torch.nn.functional.pad(batch["word_mask"], (0, 2, 0, 1))
    padded["history_mask"] = torch.nn.functional.pad(batch["history_mask"], (0, 1))
    torch.testing.assert_close(original, model(**padded)["logits"], atol=2e-6, rtol=2e-5)
    off = model(**batch, memory_off=True)["logits"]
    changed["history_ids"] = torch.randint(4, 60, batch["history_ids"].shape)
    torch.testing.assert_close(off, model(**changed, memory_off=True)["logits"], rtol=0, atol=0)
    assert not torch.allclose(off, original)


def test_full_model_causality_and_history_encoder_boundary(tiny_config, batch):
    model = LatentForestLM(tiny_config["model"]).eval()
    seen = []
    hook = model.syntax.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    original = model(**batch)["logits"]
    changed = dict(batch, input_ids=batch["input_ids"].clone())
    changed["input_ids"][:, 2:] = 32
    newer = model(**changed)["logits"]
    hook.remove()
    torch.testing.assert_close(original[:, :2], newer[:, :2], rtol=0, atol=0)
    torch.testing.assert_close(seen[0], seen[1], rtol=0, atol=0)
    assert seen[0].shape == (4, 5)


@pytest.mark.parametrize("mode", MODES)
def test_empty_history(mode, tiny_config, batch):
    tiny_config["model"]["mode"] = mode
    model = LatentForestLM(tiny_config["model"])
    batch = dict(batch, history_ids=torch.zeros(2, 1, 1, dtype=torch.long),
                 word_mask=torch.zeros(2, 1, 1, dtype=torch.bool), history_mask=torch.zeros(2, 1, dtype=torch.bool))
    result = model(**batch, return_attention=True)
    assert torch.isfinite(result["logits"]).all()
    result["loss"].backward()
    model.eval()
    torch.testing.assert_close(model(**batch)["logits"], model(**batch, memory_off=True)["logits"], rtol=0, atol=0)


def test_top1_cannot_train_and_is_eval_ablation(tiny_config, batch):
    model = LatentForestLM(tiny_config["model"])
    with pytest.raises(ValueError, match="evaluation-only"):
        model(**batch, ablation="forest_top1_eval")
    model.eval()
    result = model(**batch, ablation="forest_top1_eval")
    result["loss"].backward()
    assert grad_norm(model.syntax.span_scorer.scalar) == 0
    tiny_config["model"]["mode"] = "forest_top1_eval"
    with pytest.raises(ValueError, match="never train"):
        LatentForestLM(tiny_config["model"])


def test_literal_zero_history_dimension(tiny_config, batch):
    model = LatentForestLM(tiny_config["model"]).eval()
    empty = dict(batch, history_ids=torch.zeros(2, 0, 0, dtype=torch.long),
                 word_mask=torch.zeros(2, 0, 0, dtype=torch.bool), history_mask=torch.zeros(2, 0, dtype=torch.bool))
    torch.testing.assert_close(model(**empty)["logits"], model(**empty, memory_off=True)["logits"], rtol=0, atol=0)
