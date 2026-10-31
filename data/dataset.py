"""Random-access mmap dataset; exact target-token batches and resumable sampler."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from data.prepare import BOS, PAD


class StoryDataset(Dataset):
    def __init__(self, cache_dir, split="train", max_history=None):
        root = Path(cache_dir)
        self.manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        cfg = self.manifest["spec"]["data"]
        self.max_history = cfg["max_history"] if max_history is None else max_history
        self.split = split
        root = root / split
        self.word_offsets = np.load(root / "word_offsets.npy", mmap_mode="r")
        self.bpe_offsets = np.load(root / "bpe_offsets.npy", mmap_mode="r")
        self.story_offsets = np.load(root / "story_offsets.npy", mmap_mode="r")
        self.story_ids = np.load(root / "story_ids.npy", mmap_mode="r")
        self.words = np.memmap(root / "words.bin", dtype=np.int32, mode="r")
        self.targets = np.memmap(root / "targets.bin", dtype=np.int32, mode="r")
        if len(self) == 0:
            raise ValueError(f"Empty split: {split}")

    def __len__(self):
        return len(self.bpe_offsets) - 1

    def __getitem__(self, index):
        story = int(np.searchsorted(self.story_offsets, index, side="right") - 1)
        start = int(self.story_offsets[story])
        history_indices = range(max(start, index - self.max_history), index)
        history = [self.words[self.word_offsets[i]:self.word_offsets[i + 1]].tolist() for i in history_indices]
        target = self.targets[self.bpe_offsets[index]:self.bpe_offsets[index + 1]].tolist()
        return dict(history=history, input_ids=[BOS] + target[:-1], labels=target,
                    story_id=f"{'train' if self.split == 'train' else 'validation'}:{self.story_ids[story]}",
                    sample_id=int(index), history_sentence_ids=list(history_indices), target_sentence_id=int(index))


class Collator:
    def __call__(self, examples):
        b, t = len(examples), max(len(x["labels"]) for x in examples)
        h = max(1, max(len(x["history"]) for x in examples))
        w = max(1, max((len(s) for x in examples for s in x["history"]), default=0))
        ids = torch.full((b, t), PAD, dtype=torch.long)
        labels = torch.full((b, t), -100, dtype=torch.long)
        history = torch.full((b, h, w), PAD, dtype=torch.long)
        history_mask = torch.zeros((b, h), dtype=torch.bool)
        for i, ex in enumerate(examples):
            ids[i, :len(ex["input_ids"])] = torch.tensor(ex["input_ids"])
            labels[i, :len(ex["labels"])] = torch.tensor(ex["labels"])
            for j, sentence in enumerate(ex["history"]):
                history[i, j, :len(sentence)] = torch.tensor(sentence)
                history_mask[i, j] = True
        return dict(input_ids=ids, labels=labels, target_mask=labels.ne(-100),
                    history_ids=history, word_mask=history.ne(PAD), history_mask=history_mask,
                    story_ids=[ex["story_id"] for ex in examples],
                    sample_ids=[ex["sample_id"] for ex in examples])


class StatefulSampler:
    """No prefetched worker state: cursor advances only when an example is consumed.

    An example may be split across optimizer updates to meet exact CE-token
    budgets. The same full prefix is recomputed; only unconsumed labels contribute.
    """
    def __init__(self, dataset, seed):
        self.dataset, self.seed = dataset, seed
        self.epoch = self.cursor = self.token_offset = 0
        self._make_order()

    def _make_order(self):
        self.order = np.random.default_rng(self.seed + self.epoch).permutation(len(self.dataset))

    def state_dict(self):
        return dict(seed=self.seed, epoch=self.epoch, cursor=self.cursor, token_offset=self.token_offset,
                    dataset_size=len(self.dataset))

    def load_state_dict(self, state):
        if state["dataset_size"] != len(self.dataset) or state["seed"] != self.seed:
            raise ValueError("Sampler dataset or seed changed")
        self.epoch, self.cursor, self.token_offset = state["epoch"], state["cursor"], state["token_offset"]
        self._make_order()

    def batch(self, max_examples, token_budget):
        examples, used = [], 0
        while len(examples) < max_examples and used < token_budget:
            if self.cursor == len(self.order):
                self.epoch += 1
                self.cursor = 0
                self._make_order()
            ex = self.dataset[int(self.order[self.cursor])]
            remaining = len(ex["labels"]) - self.token_offset
            take = min(remaining, token_budget - used)
            labels = [-100] * self.token_offset + ex["labels"][self.token_offset:self.token_offset + take]
            ex["input_ids"] = ex["input_ids"][:len(labels)]
            ex["labels"] = labels
            examples.append(ex)
            used += take
            self.token_offset += take
            if self.token_offset == self.dataset.bpe_offsets[self.order[self.cursor] + 1] - self.dataset.bpe_offsets[self.order[self.cursor]]:
                self.cursor += 1
                self.token_offset = 0
        return Collator()(examples), used
