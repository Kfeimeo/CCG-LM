"""Download once, fit train-only vocabularies, build memory-mapped sentence caches.

No sentence or story concatenation. Validation is split at the story level before
sample construction. The --demo source is synthetic and ONLY for software tests.

Vocabulary counting and sentence encoding run in worker processes (``--workers``).
Chunks are processed in story order with a bounded in-flight window, so the cache
bytes and manifest fingerprint are identical to a single-process build.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import itertools
import json
import os
from pathlib import Path
import re
import time
import numpy as np

# utils imports torch; keep it out of the module scope so spawned workers stay light.

SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>"]
PAD, UNK, BOS, EOS = range(4)
CACHE_VERSION = 2
CHUNK_STORIES = 2000
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


def read_raw_lines(path):
    """Undecoded JSONL lines; workers parse them so the main process stays light."""
    with Path(path).open(encoding="utf-8") as handle:
        yield from handle


def count_lines(path):
    """Story count without JSON parsing (cache_raw writes exactly one line per story)."""
    total = 0
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**22), b""):
            total += block.count(b"\n")
    return total


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


# ---------------------------------------------------------------------------
# Parallel helpers. Worker functions are module-level so Windows "spawn" can
# import them; per-process state is loaded once by the initializer.
# ---------------------------------------------------------------------------

def _chunks(iterable, size):
    iterator = iter(iterable)
    while True:
        chunk = list(itertools.islice(iterator, size))
        if not chunk:
            return
        yield chunk


def _ordered_map(func, chunks, workers, initializer=None, initargs=()):
    """Yield func(chunk) in input order with at most 2*workers chunks in flight."""
    if workers <= 1:
        if initializer is not None:
            initializer(*initargs)
        for chunk in chunks:
            yield func(chunk)
        return
    with ProcessPoolExecutor(workers, initializer=initializer, initargs=initargs) as pool:
        pending = deque()
        for chunk in chunks:
            pending.append(pool.submit(func, chunk))
            if len(pending) >= 2 * workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def _resolve_workers(workers, n_items):
    requested = (os.cpu_count() or 1) if workers is None else int(workers)
    return max(1, min(requested, -(-n_items // CHUNK_STORIES)))


def _count_chunk(lines):
    counter = Counter()
    for line in lines:
        counter.update(word_tokenize(json.loads(line)["text"]))
    return counter


def count_words(raw_path, n_stories, workers):
    counter = Counter()
    for partial in _ordered_map(_count_chunk, _chunks(read_raw_lines(raw_path), CHUNK_STORIES),
                                _resolve_workers(workers, n_stories)):
        counter.update(partial)
    return counter


_ENCODER = None


def _init_encoder(tokenizer_path, vocab_path, max_words, max_target_length):
    global _ENCODER
    # One Rust thread pool per worker process would oversubscribe the machine.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from tokenizers import Tokenizer
    words = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    _ENCODER = (Tokenizer.from_file(str(tokenizer_path)), words, max_words, max_target_length)


def _encode_chunk(items):
    """items: [(story_id, raw_jsonl_line)] -> per-chunk arrays in story/sentence order."""
    tokenizer, words, max_words, max_target_length = _ENCODER
    kept_ids, per_story, sentences = [], [], []
    for story_id, line in items:
        split = split_sentences(json.loads(line)["text"])
        if not split:
            continue
        kept_ids.append(int(story_id))
        per_story.append(len(split))
        sentences.extend(split)
    word_ids, word_lengths = [], []
    for sentence in sentences:
        w = [words.get(t, UNK) for t in word_tokenize(sentence)[:max_words]] or [UNK]
        word_ids.extend(w)
        word_lengths.append(len(w))
    target_ids, target_lengths = [], []
    for encoding in (tokenizer.encode_batch(sentences) if sentences else []):
        target = encoding.ids[:max_target_length - 1] + [EOS]
        target_ids.extend(target)
        target_lengths.append(len(target))
    return (np.asarray(kept_ids, dtype=np.int64), np.asarray(per_story, dtype=np.int64),
            np.asarray(word_ids, dtype=np.int32), np.asarray(word_lengths, dtype=np.int64),
            np.asarray(target_ids, dtype=np.int32), np.asarray(target_lengths, dtype=np.int64))


def build_split(items, n_items, root, tokenizer_path, vocab_path, cfg, workers=None, label=""):
    """items: iterable of (story_id, raw_jsonl_line). Streams token arrays to disk."""
    root.mkdir(parents=True, exist_ok=True)
    workers = _resolve_workers(workers, n_items)
    initargs = (str(tokenizer_path), str(vocab_path), cfg["max_words"], cfg["max_target_length"])
    ids, story_counts, word_lengths, target_lengths = [], [], [], []
    n_bpe = 0
    started, done = time.time(), 0
    with (root / "words.bin").open("wb") as wf, (root / "targets.bin").open("wb") as bf:
        for chunk_ids, per_story, w_ids, w_len, t_ids, t_len in _ordered_map(
                _encode_chunk, _chunks(items, CHUNK_STORIES), workers, _init_encoder, initargs):
            w_ids.tofile(wf)
            t_ids.tofile(bf)
            ids.append(chunk_ids)
            story_counts.append(per_story)
            word_lengths.append(w_len)
            target_lengths.append(t_len)
            n_bpe += int(t_ids.size)
            done += 1
            if done % 100 == 0:
                print(f"  {label}: {min(done * CHUNK_STORIES, n_items)}/{n_items} stories, "
                      f"{time.time() - started:.0f}s", flush=True)

    def offsets(parts):
        values = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
        return np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(values, dtype=np.int64)])

    story_ids = np.concatenate(ids) if ids else np.zeros(0, dtype=np.int64)
    tables = {"word_offsets": offsets(word_lengths), "bpe_offsets": offsets(target_lengths),
              "story_offsets": offsets(story_counts), "story_ids": story_ids}
    for name, values in tables.items():
        np.save(root / f"{name}.npy", np.asarray(values, dtype=np.int64))
    n_sent = int(tables["bpe_offsets"].size - 1)
    return {"stories": int(story_ids.size), "sentences": n_sent, "target_tokens": n_bpe}


def hash_files(paths, workers=None):
    """SHA256 of each path; hashlib releases the GIL so threads overlap disk and CPU."""
    from utils import sha256
    paths = list(paths)
    with ThreadPoolExecutor(max(1, min(len(paths) or 1, workers or os.cpu_count() or 1, 8))) as pool:
        return dict(zip(paths, pool.map(sha256, paths)))


def verify_cache(root, manifest):
    hashes = hash_files([Path(root) / relative for relative in manifest["hashes"]])
    for relative, expected in manifest["hashes"].items():
        if hashes[Path(root) / relative] != expected:
            raise ValueError(f"Corrupt cache file: {relative}")


def prepare(cfg, demo=False, max_train_stories=None, max_validation_stories=None, verify=False, hf_token=None,
            workers=None):
    """workers=None uses all CPUs; small inputs automatically fall back to a single process."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from utils import atomic_json
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
    clock = time.time()

    def stage(name):
        nonlocal clock
        print(f"[{name}] {time.time() - clock:.0f}s", flush=True)
        clock = time.time()

    raw_train = cache_raw(root / "raw", "train", max_train_stories, demo, hf_token=hf_token)
    raw_val = cache_raw(root / "raw", "validation", max_validation_stories, demo, hf_token=hf_token)
    stage("raw data")
    n_train, n_val = count_lines(raw_train), count_lines(raw_val)
    counter = count_words(raw_train, n_train, workers)
    # Deterministic frequency ties. Never fit either vocabulary on validation.
    ranked = sorted(counter, key=lambda key: (-counter[key], key))[:cfg["word_vocab_size"] - len(SPECIAL)]
    words = {word: i for i, word in enumerate(SPECIAL + ranked)}
    atomic_json(words, root / "word_vocab.json")
    stage("word vocabulary")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=cfg["vocab_size"], special_tokens=SPECIAL,
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    tokenizer.train_from_iterator(read_raw(raw_train), trainer=trainer)
    tokenizer.save(str(root / "tokenizer.json"))
    if [tokenizer.token_to_id(t) for t in SPECIAL] != list(range(4)):
        raise RuntimeError("Unexpected special token IDs")
    stage("bpe tokenizer")
    if n_val < 2:
        raise ValueError("Need at least two validation stories for disjoint dev/test")
    order = np.random.default_rng(cfg["split_seed"]).permutation(n_val)
    n_dev = max(1, min(n_val - 1, int(n_val * cfg["dev_fraction"])))
    dev_ids = set(order[:n_dev].tolist())
    stats = {}
    tokenizer_path, vocab_path = root / "tokenizer.json", root / "word_vocab.json"
    stats["train"] = build_split(enumerate(read_raw_lines(raw_train)), n_train, root / "train",
                                 tokenizer_path, vocab_path, cfg, workers, label="train")
    stage("train split")
    for split, keep_dev in [("dev", True), ("test", False)]:
        items = [(i, line) for i, line in enumerate(read_raw_lines(raw_val)) if (i in dev_ids) == keep_dev]
        stats[split] = build_split(items, len(items), root / split, tokenizer_path, vocab_path, cfg, workers,
                                   label=split)
    stage("dev/test splits")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix != ".tmp" and "raw" not in p.parts)
    digests = hash_files(files + [raw_train, raw_val], workers)
    hashes = {str(p.relative_to(root)).replace("\\", "/"): digests[p] for p in files}
    import hashlib
    fingerprint = hashlib.sha256(json.dumps({"spec": spec, "hashes": hashes}, sort_keys=True).encode()).hexdigest()
    manifest = dict(spec=spec, fingerprint=fingerprint, hashes=hashes, stats=stats,
                    bpe_vocab_size=tokenizer.get_vocab_size(), word_vocab_size=len(words),
                    raw_hashes={"train": digests[raw_train], "validation": digests[raw_val]})
    # Manifest is the commit marker. Interrupted preparation is never considered ready.
    atomic_json(manifest, manifest_path)
    stage("hashes + manifest")
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
    parser.add_argument("--workers", type=int, default=None,
                        help="Worker processes for counting/encoding (default: all CPUs; output is identical for any value)")
    args = parser.parse_args()
    from utils import load_config
    prepare(load_config(args.config)["data"], args.demo, args.max_train_stories, args.max_validation_stories,
            args.verify_cache, hf_token=args.hf_token, workers=args.workers)


if __name__ == "__main__":
    main()
