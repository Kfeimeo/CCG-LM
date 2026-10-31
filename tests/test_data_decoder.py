import torch
from data.prepare import split_sentences, word_tokenize, BOS, EOS
from data.dataset import Collator
from models.model import LatentForestLM


def test_word_splitter():
    assert word_tokenize("Dr. Amy can't go, U.S. friends!") == ["dr.", "amy", "can't", "go", ",", "u.s.", "friends", "!"]
    assert split_sentences("Dr. Amy can't go. Is Tom here? Yes!") == ["Dr. Amy can't go.", "Is Tom here?", "Yes!"]


def test_teacher_forcing_padding():
    ex = dict(history=[], input_ids=[BOS, 10, 11], labels=[10, 11, EOS], story_id="a", sample_id=0)
    batch = Collator()([ex, dict(ex, input_ids=[BOS], labels=[EOS])])
    assert batch["labels"].tolist() == [[10, 11, EOS], [EOS, -100, -100]]
    assert batch["input_ids"][0, 1:].tolist() == batch["labels"][0, :-1].tolist()
    assert not batch["history_mask"].any()


def test_baseline_learns(tiny_config, batch):
    tiny_config["model"]["mode"] = "no_memory"
    model = LatentForestLM(tiny_config["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    before = float(model(**batch)["loss"].detach())
    for _ in range(25):
        optimizer.zero_grad()
        result = model(**batch)
        result["loss"].backward()
        optimizer.step()
    after = float(model(**batch)["loss"].detach())
    assert after < before * 0.4


def test_decoder_causality(tiny_config, batch):
    tiny_config["model"]["mode"] = "no_memory"
    model = LatentForestLM(tiny_config["model"]).eval()
    original = model(**batch)["logits"]
    changed = dict(batch, input_ids=batch["input_ids"].clone())
    changed["input_ids"][:, 2:] = 31
    torch.testing.assert_close(original[:, :2], model(**changed)["logits"][:, :2], rtol=0, atol=0)
