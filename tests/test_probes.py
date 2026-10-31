from collections import Counter
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from data.prepare import SPECIAL, demo_stories, word_tokenize
from semantic_tests import generate_probes


def test_semantic_templates_length_balance_and_reversals():
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.train_from_iterator(demo_stories(96), trainers.BpeTrainer(vocab_size=512, special_tokens=SPECIAL,
                                   initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    rows, _ = generate_probes(tokenizer)
    assert {row["category"] for row in rows} == {"pp_attachment", "context_switch", "cross_sentence_reference", "compositional_generalization"}
    pairs = {}
    for row in rows:
        pairs.setdefault(row["pair_id"], []).append(row)
        assert len(tokenizer.encode(row["correct"]).ids) == len(tokenizer.encode(row["wrong"]).ids)
    for a, b in pairs.values():
        assert a["context"] != b["context"]
        assert a["correct"] == b["wrong"] and a["wrong"] == b["correct"]
    assert Counter(w for row in rows for w in word_tokenize(row["correct"])) == Counter(w for row in rows for w in word_tokenize(row["wrong"]))
