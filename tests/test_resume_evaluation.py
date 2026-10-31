import copy
import json
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader
from data.dataset import Collator, StatefulSampler, StoryDataset
from data.prepare import prepare, BOS, EOS
from evaluate import evaluate_loader, learning_efficiency, paired_bootstrap
from models.model import LatentForestLM
from train import make_optimizer, make_schedule, save_checkpoint, load_checkpoint, train_update
from utils import set_seed


@pytest.fixture
def cache(tmp_path, tiny_config):
    cfg = dict(tiny_config["data"], cache_dir=str(tmp_path / "cache"), vocab_size=512,
               word_vocab_size=80, max_words=8, max_target_length=16)
    prepare(cfg, demo=True, max_train_stories=12, max_validation_stories=8)
    return cfg


def test_story_boundaries_cache_and_independent_ids(cache):
    data = StoryDataset(cache["cache_dir"])
    assert len(data) == 48
    for i in range(len(data)):
        ex = data[i]
        assert ex["input_ids"][0] == BOS and ex["labels"][-1] == EOS
        assert ex["input_ids"][1:] == ex["labels"][:-1]
        for previous in ex["history_sentence_ids"]:
            assert previous < ex["target_sentence_id"]
            assert data[previous]["story_id"] == ex["story_id"]
        if i % 4 == 0:
            assert not ex["history"]
    dev, test = StoryDataset(cache["cache_dir"], "dev"), StoryDataset(cache["cache_dir"], "test")
    assert set(dev.story_ids.tolist()).isdisjoint(test.story_ids.tolist())
    manifest = prepare(cache, demo=True, max_train_stories=12, max_validation_stories=8)
    assert manifest["fingerprint"] == data.manifest["fingerprint"]
    with pytest.raises(ValueError, match="differs"):
        prepare(dict(cache, max_words=7), demo=True, max_train_stories=12, max_validation_stories=8)


def test_sampler_exact_ce_tokens_and_cursor(cache):
    data = StoryDataset(cache["cache_dir"])
    sampler = StatefulSampler(data, 42)
    first, count = sampler.batch(2, 3)
    assert count == 3 == first["labels"].ne(-100).sum()
    saved = copy.deepcopy(sampler.state_dict())
    expected, _ = sampler.batch(3, 29)
    newer = StatefulSampler(data, 42)
    newer.load_state_dict(saved)
    actual, _ = newer.batch(3, 29)
    for key in ("input_ids", "labels", "history_ids"):
        torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
    assert newer.state_dict() == sampler.state_dict()


def test_resume_full_rng_optimizer_scheduler_data(tiny_config, cache, tmp_path):
    cfg = copy.deepcopy(tiny_config)
    cfg["data"] = cache
    cfg["model"].update(vocab_size=512, dropout=0.15, exploration=0.2)
    set_seed(42, True)
    data = StoryDataset(cache["cache_dir"])
    model = LatentForestLM(cfg["model"])
    optimizer = make_optimizer(model, cfg["train"])
    scheduler, sampler = make_schedule(optimizer, cfg["train"]), StatefulSampler(data, 42)
    args = dict(tokens=19, micro_batch_size=2, device=torch.device("cpu"), precision="fp32", clip=1.0)
    train_update(model, sampler, optimizer, scheduler, **args)
    path = tmp_path / "checkpoint.pt"
    state = dict(step=1, target_tokens=19)
    save_checkpoint(path, model, optimizer, scheduler, sampler, cfg, state, data.manifest["fingerprint"])
    python_before = __import__("random").random()
    numpy_before = float(np.random.rand())
    expected_metrics = train_update(model, sampler, optimizer, scheduler, **args)
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    expected_rng = torch.get_rng_state().clone()
    restored = LatentForestLM(cfg["model"])
    opt = make_optimizer(restored, cfg["train"])
    sch, sam = make_schedule(opt, cfg["train"]), StatefulSampler(data, 42)
    load_checkpoint(path, restored, opt, sch, sam, cfg, data.manifest["fingerprint"])
    assert __import__("random").random() == python_before
    assert float(np.random.rand()) == numpy_before
    actual_metrics = train_update(restored, sam, opt, sch, **args)
    assert actual_metrics == expected_metrics
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    assert sam.state_dict() == sampler.state_dict()
    assert sch.state_dict() == scheduler.state_dict()
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_evaluation_and_bootstrap(tiny_config, cache):
    tiny_config["model"]["vocab_size"] = 512
    model = LatentForestLM(tiny_config["model"])
    data = StoryDataset(cache["cache_dir"], "dev")
    metrics, rows = evaluate_loader(model, DataLoader(data, batch_size=3, collate_fn=Collator()),
                                    torch.device("cpu"), "fp32", diagnostics=True, top1=True)
    assert metrics["target_tokens"] == data.manifest["stats"]["dev"]["target_tokens"]
    for key in ("nll", "memory_off_minus_normal_nll", "shuffled_minus_normal_nll", "forest_top1_eval_nll", "root_candidate_entropy"):
        assert np.isfinite(metrics[key])
    assert model.training  # Evaluation restores caller's mode.
    a = pd.DataFrame(rows)
    b = a.copy()
    b["loss_sum"] += 0.25 * b["tokens"]
    report = paired_bootstrap(a, b, iterations=100)
    assert report["difference_a_minus_b"] == pytest.approx(-0.25)
    assert report["ci_low"] == pytest.approx(-0.25)
    assert report["ci_high"] == pytest.approx(-0.25)
    with pytest.raises(ValueError, match="identical"):
        paired_bootstrap(a, b.iloc[:-1], iterations=10)


def test_schedule_endpoint_not_short_run_budget(tiny_config):
    p = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([p], lr=0.001)
    scheduler = make_schedule(optimizer, tiny_config["train"])
    scheduler.step(5_000_000)
    scheduler.step(20_000_000)
    assert optimizer.param_groups[0]["lr"] > 0.0009
    frame = pd.DataFrame(dict(target_tokens=[5_000_000, 25_000_000, 100_000_000],
                              val_nll=[3.0, 2.0, 1.0], elapsed_seconds=[1.0, 5.0, 20.0]))
    result = learning_efficiency(frame, nll_threshold=2.5)
    assert result["threshold_target_tokens"] == 25_000_000
    assert result["aulc_5m_100m"] == pytest.approx(162_500_000)
    assert learning_efficiency(frame.iloc[:2])["aulc_5m_100m"] is None


def test_complete_training_resume_in_new_directory(tiny_config, cache, tmp_path):
    from train import run
    cfg = copy.deepcopy(tiny_config)
    cfg["data"] = cache
    cfg["model"].update(vocab_size=512, dropout=0.1)
    cfg["train"].update(budget_tokens=32, checkpoint_tokens=[16, 32], eval_every_steps=1,
                         eval_batches=1, eval_batch_size=2, output_dir=str(tmp_path / "continuous"), device="cpu")
    assert run(cfg)["target_tokens"] == 32
    full = torch.load(tmp_path / "continuous/tokens_32.pt", weights_only=False)
    cfg["train"].update(budget_tokens=16, output_dir=str(tmp_path / "partial"))
    run(cfg)
    cfg["train"].update(budget_tokens=32, output_dir=str(tmp_path / "restored_elsewhere"))
    state = run(cfg, str(tmp_path / "partial/tokens_16.pt"))
    restored = torch.load(tmp_path / "restored_elsewhere/tokens_32.pt", weights_only=False)
    assert state["target_tokens"] == 32
    for name, value in full["model"].items():
        torch.testing.assert_close(value, restored["model"][name], rtol=0, atol=0)
    assert len(restored["log_rows"]) == 2
    assert pd.read_csv(tmp_path / "restored_elsewhere/metrics.csv").target_tokens.tolist() == [16, 32]
