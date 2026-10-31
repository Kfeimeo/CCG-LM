"""Exploratory, counterbalanced semantic probes; these are NOT gold benchmarks.

Each pair swaps context while reusing exactly the same two continuation strings.
Thus correct/wrong aggregate word frequencies are identical within each pair.
The cache's BPE tokenizer must give both continuations the same token count.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from data.dataset import Collator
from data.prepare import BOS, EOS, UNK, split_sentences, word_tokenize
from models.model import LatentForestLM
from utils import atomic_json, autocast_context, choose_device


def generate_probes(tokenizer):
    rows = []
    families = []
    # Every template has a context-reversal partner with identical target strings.
    for first, second in combinations(["girl", "boy", "woman", "man", "cat", "dog", "duck", "bear", "frog", "lion", "pig", "hen"], 2):
        families.append(("pp_attachment", [
            f"The {first} carried a telescope. The {second} had empty hands. The {second} saw the {first} with the telescope.",
            f"The {second} carried a telescope. The {first} had empty hands. The {second} saw the {first} with the telescope."],
            [f"The telescope belonged to the {first}.", f"The telescope belonged to the {second}."]))
        families.append(("context_switch", [
            f"The {first} owned the red hat. The {second} owned the blue hat. The {second} met the {first} with the red hat.",
            f"The {second} owned the red hat. The {first} owned the blue hat. The {second} met the {first} with the red hat."],
            [f"The red hat was on the {first}.", f"The red hat was on the {second}."]))
    for girl, boy in [("Amy", "Tom"), ("Mia", "Ben"), ("Lily", "Sam")]:
        families.append(("cross_sentence_reference", [
            f"{girl} found a ball. {boy} found a book. She put her toy in a box.",
            f"{girl} found a book. {boy} found a ball. She put her toy in a box."],
            ["The box held the ball.", "The box held the book."]))
    # Combinations withheld from any probe-specific fitting (there is no probe
    # training). This is recombination, not evidence of a TinyStories holdout.
    for color1, color2 in [("red", "blue"), ("green", "yellow"), ("black", "white")]:
        for object1, object2 in [("ball", "book"), ("hat", "kite")]:
            families.append(("compositional_generalization", [
                f"A {color1} {object1} was in the box. A {color2} {object2} was on the table.",
                f"A {color2} {object1} was in the box. A {color1} {object2} was on the table."],
                [f"The {object1} was {color1}.", f"The {object1} was {color2}."]))
    rejected = []
    accepted = Counter()
    for pair_id, (category, contexts, continuations) in enumerate(families):
        if accepted[category] >= 8:
            continue
        lengths = [len(tokenizer.encode(text).ids) + 1 for text in continuations]
        if lengths[0] != lengths[1]:
            rejected.append(dict(pair_id=pair_id, category=category, lengths=lengths,
                                 reason="unequal BPE continuation length"))
            continue
        for direction, context in enumerate(contexts):
            rows.append(dict(id=f"{pair_id}:{direction}", pair_id=pair_id, category=category,
                             context=context, correct=continuations[direction], wrong=continuations[1 - direction],
                             target_tokens=lengths[0]))
        accepted[category] += 1
    # Check exact lexical balance of answer roles across reversal partners.
    assert Counter(t for row in rows for t in word_tokenize(row["correct"])) == Counter(t for row in rows for t in word_tokenize(row["wrong"]))
    return rows, rejected


def make_probe_example(row, continuation, tokenizer, words, cfg):
    sentences = split_sentences(row["context"])[-cfg["max_history"]:]
    history = [[words.get(word, UNK) for word in word_tokenize(text)[:cfg["max_words"]]] or [UNK] for text in sentences]
    target = tokenizer.encode(continuation).ids + [EOS]
    if len(target) > cfg["max_target_length"]:
        raise ValueError("Do not truncate semantic continuations; increase target capacity")
    return dict(history=history, input_ids=[BOS] + target[:-1], labels=target,
                story_id=row["id"], sample_id=row["id"])


@torch.no_grad()
def score_probes(checkpoint_path, cache_dir, rows, device):
    from tokenizers import Tokenizer
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest = json.loads((Path(cache_dir) / "manifest.json").read_text(encoding="utf-8"))
    if manifest["fingerprint"] != checkpoint["data_fingerprint"]:
        raise ValueError("Probe tokenizer/cache differs from checkpoint")
    cfg = checkpoint["config"]
    model = LatentForestLM(cfg["model"], cfg["seed"]).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    tokenizer = Tokenizer.from_file(str(Path(cache_dir) / "tokenizer.json"))
    words = json.loads((Path(cache_dir) / "word_vocab.json").read_text(encoding="utf-8"))
    result = []
    for row in rows:
        ex = [make_probe_example(row, row[key], tokenizer, words, cfg["model"]) for key in ("correct", "wrong")]
        batch = Collator()(ex)
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        with autocast_context(device, cfg["train"]["precision"]):
            output = model(**batch)
        logp = (-output["token_losses"].sum(-1)).float().cpu().tolist()
        result.append(dict(**row, correct_logp=logp[0], wrong_logp=logp[1], logp_margin=logp[0] - logp[1],
                           chosen_correct=logp[0] > logp[1], checkpoint=str(checkpoint_path)))
    return result


def main():
    from tokenizers import Tokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="cache/demo")
    parser.add_argument("--output-dir", default="runs/semantic_tests")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(Path(args.cache_dir) / "tokenizer.json"))
    rows, rejected = generate_probes(tokenizer)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No equal-length probes for this tokenizer")
    with (output / "probes.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    atomic_json(dict(pairs=len(rows) // 2, examples=len(rows), rejected=rejected,
                     caveat="Exploratory templates; not gold parses or a curated human semantic benchmark. "
                            "Reversal controls answer frequencies and continuation length, not all context surface cues."), output / "generation.json")
    if args.checkpoint:
        device = choose_device(args.device)
        if device.type == "cpu":
            torch.set_num_threads(min(4, torch.get_num_threads()))
        scores = score_probes(args.checkpoint, args.cache_dir, rows, device)
        frame = pd.DataFrame(scores)
        frame.to_csv(output / "scores.csv", index=False)
        summary = frame.groupby("category").agg(examples=("id", "count"), mean_margin=("logp_margin", "mean"), accuracy=("chosen_correct", "mean"))
        summary.to_csv(output / "summary.csv")
        print(summary.to_string())


if __name__ == "__main__":
    main()
