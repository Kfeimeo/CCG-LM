"""Download once, fit train-only vocabularies, build memory-mapped sentence caches.

No sentence or story concatenation. Validation is split at the story level before
sample construction. The --demo source is synthetic and ONLY for software tests.
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter
import itertools
import json
import os
from pathlib import Path
import re
import numpy as np

from utils import atomic_json, load_config, sha256

SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>"]
PAD, UNK, BOS, EOS = range(4)
CACHE_VERSION = 2
# Keep contractions, possessives and common abbreviations as one word token.
WORD_RE = re.compile(r"(?:[A-Za-z]\.){2,}|(?:Mr|Mrs|Ms|Dr|Prof|Jr|Sr)\.|[^\W_]+(?:['’][^\W_]+)*|[^\w\s]", re.UNICODE | re.IGNORECASE)
ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "jr.", "sr.", "e.g.", "i.e."}


def word_tokenize(text):
    return [w.lower().replace("’", "'") for w in WORD_RE.findall(text)]


def split_sentences(text):
    """Deterministic lightweight English splitter; no downloaded NLP model.

    Quoted closing punctuation stays in its sentence. Abbreviations and decimals
    are protected. This remains a heuristic; segmentation is not gold syntax.
    """
    text = re.sub(r"\s+", " ", text).strip()
    result, start = [], 0
    for match in re.finditer(r'[.!?]+["\x27”’)]*(?=\s|$)', text):
        prefix = text[start:match.end()]
        last = prefix.split()[-1].lower().strip('"\x27“‘(')
        if last in ABBREVIATIONS or re.fullmatch(r"(?:[a-z]\.){2,}", last):
            continue
        sentence = text[start:match.end()].strip()
        if sentence:
            result.append(sentence)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        result.append(tail)
    return result


def read_raw(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)["text"]


def demo_stories(count, offset=0):
    names = ["Mia", "Tom", "Lily", "Sam", "Ben", "Amy"]
    objects = ["ball", "book", "kite", "doll", "box", "hat"]
    for i in range(offset, offset + count):
        name, obj = names[i % len(names)], objects[(i // len(names)) % len(objects)]
        yield f"{name} saw a {obj}. The {obj} was red. {name} took it home. It made {name} happy!"


def cache_raw(root, split, limit, demo, hf_token=None):
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{'demo' if demo else 'tinystories'}_{split}_{limit or 'all'}.jsonl"
    if path.exists():
        return path
    temporary = path.with_suffix(".tmp")
    if demo:
        iterator = demo_stories(limit or (96 if split == "train" else 24), 0 if split == "train" else 1000)
    else:
        from datasets import load_dataset
        token = hf_token or os.environ.get("HF_TOKEN") or None
        token_path = Path(__file__).resolve().parents[1] / ".hf_token"
        if not token and token_path.is_file():
            token = token_path.read_text(encoding="utf-8").strip() or None
        dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True, token=token)
        iterator = (row["text"] for row in dataset)
        if limit:
            iterator = itertools.islice(iterator, limit)
    with temporary.open("w", encoding="utf-8") as handle:
        for text in iterator:
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def build_split(texts, story_ids, root, tokenizer, words, cfg):
    root.mkdir(parents=True, exist_ok=True)
    # Small offset tables stay in RAM; token arrays are streamed to disk.
    word_offsets, bpe_offsets, story_offsets = array("q", [0]), array("q", [0]), array("q", [0])
    n_word = n_bpe = n_sent = 0
    ids = array("q")
    with (root / "words.bin").open("wb") as wf, (root / "targets.bin").open("wb") as bf:
        for text, story_id in zip(texts, story_ids):
            sentences = split_sentences(text)
            if not sentences:
                continue
            ids.append(int(story_id))
            for sentence in sentences:
                w = [words.get(t, UNK) for t in word_tokenize(sentence)[:cfg["max_words"]]] or [UNK]
                target = tokenizer.encode(sentence).ids[:cfg["max_target_length"] - 1] + [EOS]
                np.asarray(w, dtype=np.int32).tofile(wf)
                np.asarray(target, dtype=np.int32).tofile(bf)
                n_word += len(w)
                n_bpe += len(target)
                n_sent += 1
                word_offsets.append(n_word)
                bpe_offsets.append(n_bpe)
            story_offsets.append(n_sent)
    for name, values in [("word_offsets", word_offsets), ("bpe_offsets", bpe_offsets),
                         ("story_offsets", story_offsets), ("story_ids", ids)]:
        np.save(root / f"{name}.npy", np.asarray(values, dtype=np.int64))
    return {"stories": len(ids), "sentences": n_sent, "target_tokens": n_bpe}


def verify_cache(root, manifest):
    for relative, expected in manifest["hashes"].items():
        if sha256(Path(root) / relative) != expected:
            raise ValueError(f"Corrupt cache file: {relative}")


def prepare(cfg, demo=False, max_train_stories=None, max_validation_stories=None, verify=False, hf_token=None):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    root = Path(cfg["cache_dir"])
    root.mkdir(parents=True, exist_ok=True)
    spec = {"version": CACHE_VERSION, "data": {k: v for k, v in cfg.items() if k != "cache_dir"}, "demo": demo,
            "max_train_stories": max_train_stories, "max_validation_stories": max_validation_stories}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["spec"] != spec:
            raise ValueError("Cache configuration differs; select another cache_dir (no silent reuse).")
        if verify:
            verify_cache(root, manifest)
        print(f"Reusing cache {root}; fingerprint={manifest['fingerprint']}")
        return manifest
    raw_train = cache_raw(root / "raw", "train", max_train_stories, demo, hf_token=hf_token)
    raw_val = cache_raw(root / "raw", "validation", max_validation_stories, demo, hf_token=hf_token)
    counter = Counter()
    for text in read_raw(raw_train):
        counter.update(word_tokenize(text))
    # Deterministic frequency ties. Never fit either vocabulary on validation.
    ranked = sorted(counter, key=lambda key: (-counter[key], key))[:cfg["word_vocab_size"] - len(SPECIAL)]
    words = {word: i for i, word in enumerate(SPECIAL + ranked)}
    atomic_json(words, root / "word_vocab.json")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=cfg["vocab_size"], special_tokens=SPECIAL,
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    tokenizer.train_from_iterator(read_raw(raw_train), trainer=trainer)
    tokenizer.save(str(root / "tokenizer.json"))
    if [tokenizer.token_to_id(t) for t in SPECIAL] != list(range(4)):
        raise RuntimeError("Unexpected special token IDs")
    n_val = sum(1 for _ in read_raw(raw_val))
    if n_val < 2:
        raise ValueError("Need at least two validation stories for disjoint dev/test")
    order = np.random.default_rng(cfg["split_seed"]).permutation(n_val)
    n_dev = max(1, min(n_val - 1, int(n_val * cfg["dev_fraction"])))
    dev_ids = set(order[:n_dev].tolist())
    stats = {}
    n_train = sum(1 for _ in read_raw(raw_train))
    stats["train"] = build_split(read_raw(raw_train), range(n_train), root / "train", tokenizer, words, cfg)
    for split, keep_dev in [("dev", True), ("test", False)]:
        ids = [i for i in range(n_val) if (i in dev_ids) == keep_dev]
        texts = (text for i, text in enumerate(read_raw(raw_val)) if (i in dev_ids) == keep_dev)
        stats[split] = build_split(texts, ids, root / split, tokenizer, words, cfg)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix != ".tmp" and "raw" not in p.parts)
    hashes = {str(p.relative_to(root)).replace("\\", "/"): sha256(p) for p in files}
    import hashlib
    fingerprint = hashlib.sha256(json.dumps({"spec": spec, "hashes": hashes}, sort_keys=True).encode()).hexdigest()
    manifest = dict(spec=spec, fingerprint=fingerprint, hashes=hashes, stats=stats,
                    bpe_vocab_size=tokenizer.get_vocab_size(), word_vocab_size=len(words),
                    raw_hashes={"train": sha256(raw_train), "validation": sha256(raw_val)})
    # Manifest is the commit marker. Interrupted preparation is never considered ready.
    atomic_json(manifest, manifest_path)
    print(json.dumps(manifest["stats"], indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--max-train-stories", type=int)
    parser.add_argument("--max-validation-stories", type=int)
    parser.add_argument("--verify-cache", action="store_true", help="Check all SHA256 hashes when reusing a cache")
    parser.add_argument("--hf-token", "--token", dest="hf_token", help="Hugging Face token (overrides HF_TOKEN and .hf_token)")
    args = parser.parse_args()
    prepare(load_config(args.config)["data"], args.demo, args.max_train_stories, args.max_validation_stories,
            args.verify_cache, hf_token=args.hf_token)


if __name__ == "__main__":
    main()
