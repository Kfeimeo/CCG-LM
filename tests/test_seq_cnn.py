import pytest
import torch
from torch.nn import functional as F

from models.model import LatentForestLM
from models.seq_cnn import SequentialCNN
from utils import grad_norm


def test_local_receptive_field_and_order():
    encoder = SequentialCNN(8, kernel_size=3, layers=2).eval()
    x = torch.randn(1, 11, 8, requires_grad=True)
    mask = torch.ones(1, 11, dtype=torch.bool)
    output = encoder(x, mask)
    gradient = torch.autograd.grad(output[0, 5, 0], x)[0].abs().sum(-1)[0]
    assert encoder.receptive_field == 5
    assert torch.count_nonzero(gradient[:3]) == 0
    assert torch.count_nonzero(gradient[8:]) == 0
    assert (gradient[3:8] > 0).all()
    flipped = encoder(x.flip(1), mask).flip(1)
    assert not torch.allclose(output, flipped)


@pytest.mark.parametrize("kernel", [1, 3])
def test_padding_invariance_and_empty_rows(kernel):
    encoder = SequentialCNN(8, kernel, 3, hidden_size=12).eval()
    x = torch.randn(2, 7, 8)
    mask = torch.tensor([[True, True, True, False, False, False, False], [False] * 7])
    original = encoder(x, mask)
    padded = encoder(F.pad(x, (0, 0, 0, 4), value=999), F.pad(mask, (0, 4)))
    torch.testing.assert_close(original, padded[:, :7], rtol=0, atol=1e-6)
    assert torch.count_nonzero(original[~mask]) == 0


def test_kernel_one_does_not_mix_neighbors():
    encoder = SequentialCNN(8, 1, 2)
    x = torch.randn(1, 5, 8, requires_grad=True)
    gradient = torch.autograd.grad(encoder(x, torch.ones(1, 5, dtype=torch.bool))[0, 2, 0], x)[0]
    assert torch.count_nonzero(gradient[:, [0, 1, 3, 4]]) == 0
    assert gradient[:, 2].abs().sum() > 0


def test_kernel_one_memory_is_permutation_invariant(tiny_config, batch):
    cfg = dict(tiny_config["model"], mode="seq_cnn4", cnn_kernel_size=1)
    model = LatentForestLM(cfg).eval()
    history = batch["history_ids"].clone()
    history[0, 0, :5] = history[0, 0, :5].flip(0)
    history[0, 1, :3] = history[0, 1, :3].flip(0)
    history[1, 0, :4] = history[1, 0, :4].flip(0)
    original = model.encode_history(batch["history_ids"], batch["word_mask"], batch["history_mask"])
    shuffled = model.encode_history(history, batch["word_mask"], batch["history_mask"])
    torch.testing.assert_close(original["semantic"], shuffled["semantic"], atol=1e-6, rtol=1e-5)


def test_cnn_full_lm_padding_gradients_causality_and_roundtrip(tiny_config, batch):
    cfg = dict(tiny_config["model"], mode="seq_cnn4")
    model = LatentForestLM(cfg).eval()
    result = model(**batch)
    result["loss"].backward()
    assert grad_norm(model.seq_cnn) > 0
    assert grad_norm(model.pooling) > 0
    assert grad_norm(model.semantic_embedding) > 0
    assert grad_norm(model.readers) > 0
    assert model.syntax is None and model.forest is None and model.composer is None
    changed = dict(batch)
    changed["history_ids"] = F.pad(batch["history_ids"], (0, 3, 0, 1), value=79)
    changed["word_mask"] = F.pad(batch["word_mask"], (0, 3, 0, 1))
    changed["history_mask"] = F.pad(batch["history_mask"], (0, 1))
    torch.testing.assert_close(result["logits"], model(**changed)["logits"], atol=2e-6, rtol=2e-5)
    future = dict(batch, input_ids=batch["input_ids"].clone())
    future["input_ids"][:, 2:] = 32
    torch.testing.assert_close(result["logits"][:, :2], model(**future)["logits"][:, :2], rtol=0, atol=0)
    restored = LatentForestLM(cfg).eval()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(result["logits"], restored(**batch)["logits"], rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [dict(kernel_size=2), dict(layers=0), dict(hidden_size=0)])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        SequentialCNN(8, **kwargs)
