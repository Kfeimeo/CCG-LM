import pytest
import torch
from utils import load_config, set_seed


@pytest.fixture(autouse=True)
def deterministic_test():
    torch.set_num_threads(1)
    set_seed(42)


@pytest.fixture
def tiny_config():
    cfg = load_config("configs/smoke.yaml")
    cfg["model"].update(vocab_size=64, word_vocab_size=80, hidden_size=32, layers=2, heads=4,
                        ffn_size=64, dropout=0.0, cross_layers=[1, 2], syntax_embedding=12,
                        semantic_size=16, syntax_hidden=24, syntax_layers=1, syntax_heads=4,
                        composer_hidden=32, coefficient_size=8, reader_heads=4, reader_width=32,
                        max_words=8, max_target_length=16, exploration=0.0)
    cfg["train"].update(warmup_steps=2, effective_tokens=16, micro_batch_size=2, deterministic=True)
    return cfg


@pytest.fixture
def batch():
    from data.dataset import Collator
    return Collator()([
        dict(history=[[4, 5, 6, 7, 8], [9, 10, 11]], input_ids=[2, 7, 8, 9], labels=[7, 8, 9, 3], story_id="a", sample_id=0),
        dict(history=[[11, 12, 13, 14]], input_ids=[2, 10, 11], labels=[10, 11, 3], story_id="b", sample_id=1),
    ])
