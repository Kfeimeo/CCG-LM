"""One causal LM interface for every experimental memory construction."""
from __future__ import annotations

import hashlib
import random
import torch
from torch import nn
from torch.nn import functional as F

from models.composer import Composer
from models.decoder import Decoder
from models.forest import SemanticForest, catalan, compose_tree, tree_for_length
from models.reader import SemanticReader
from models.syntax import SyntaxEncoder

MODES = ("no_memory", "pooling4", "fixed_tree4", "random_tree4", "forest_scalar4", "forest_vector4")


class PoolingMemory(nn.Module):
    def __init__(self, size, slots=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(slots, size) / size**0.5)
        self.key = nn.Linear(size, size)
        self.value = nn.Sequential(nn.Linear(size, size), nn.GELU(), nn.Linear(size, size))

    def forward(self, leaves, mask):
        safe = mask.clone()
        safe[:, 0] = True
        logits = torch.einsum("kd,nwd->nkw", self.query, self.key(leaves)) / leaves.shape[-1]**0.5
        attention = logits.masked_fill(~safe[:, None, :], -torch.inf).softmax(-1)
        return torch.einsum("nkw,nwd->nkd", attention, self.value(leaves))


class LatentForestLM(nn.Module):
    def __init__(self, cfg, seed=42):
        super().__init__()
        self.cfg, self.mode, self.seed = dict(cfg), cfg["mode"], seed
        if self.mode == "forest_top1_eval":
            raise ValueError("Load a trained forest and call eval(..., ablation='forest_top1_eval'); never train hard top-1.")
        if self.mode not in MODES:
            raise ValueError(f"Unsupported mode {self.mode}")
        if cfg["candidates"] != 4:
            raise ValueError("The comparison protocol requires four slots; exact CKY tests use SemanticForest directly")
        self.decoder = Decoder(cfg)
        self.readers = nn.ModuleDict()
        self.syntax = self.semantic_embedding = self.pooling = self.forest = None
        self.tree_composer = None
        if self.mode != "no_memory":
            self.semantic_embedding = nn.Embedding(cfg["word_vocab_size"], cfg["semantic_size"], padding_idx=0)
            vector = self.mode == "forest_vector4"
            if self.mode.startswith("forest_"):
                self.syntax = SyntaxEncoder(cfg, vector)
                self.forest = SemanticForest(Composer(cfg["semantic_size"], cfg["composer_hidden"]),
                                              cfg["candidates"], cfg["max_compositions"], cfg["prune_delta"], cfg["exploration"],
                                              implementation=cfg.get("forest_implementation", "reference"))
            elif self.mode == "pooling4":
                self.pooling = PoolingMemory(cfg["semantic_size"])
            else:
                self.tree_composer = Composer(cfg["semantic_size"], cfg["composer_hidden"])
            self.readers = nn.ModuleDict({str(layer): SemanticReader(cfg, vector) for layer in cfg["cross_layers"]})
        self.last_diagnostics = {}

    @property
    def composer(self):
        return self.forest.composer if self.forest is not None else self.tree_composer

    def encode_history(self, history_ids, word_mask, history_mask, return_trees=False):
        b, h, w = history_ids.shape
        if h == 0 or w == 0:
            c = self.cfg["coefficient_size"] if self.mode == "forest_vector4" else 0
            zeros = self.semantic_embedding.weight.new_zeros
            return dict(semantic=zeros(b, 1, 4, self.cfg["semantic_size"]), log_prior=zeros(b, 1, 4),
                        vector_coeff=zeros(b, 1, 4, c),
                        candidate_mask=torch.zeros(b, 1, 4, dtype=torch.bool, device=history_ids.device),
                        history_mask=torch.zeros(b, 1, dtype=torch.bool, device=history_ids.device), diagnostics={})
        if h > self.cfg["max_history"]:
            raise ValueError("Too many history sentences")
        # Ignore all masked token IDs, even if a caller supplied nonzero padding.
        word_mask = word_mask & history_mask.unsqueeze(-1)
        ids = history_ids.masked_fill(~word_mask, 0).reshape(b * h, w)
        mask = word_mask.reshape(b * h, w)
        leaves = self.semantic_embedding(ids) * mask.unsqueeze(-1)
        if self.forest is not None:
            scores, coeff = self.syntax(ids, mask)
            mem = self.forest(leaves, scores, coeff, mask, return_trees)
        else:
            valid = mask.any(-1)
            if self.pooling is not None:
                vectors = self.pooling(leaves, mask)
                trees = None
            else:
                # Baseline control paths group by sentence length and shape.
                vectors = [None] * (b * h)
                trees = [[] for _ in range(b * h)]
                cpu_ids, lengths = ids.detach().cpu().tolist(), mask.sum(-1).detach().cpu().tolist()
                groups = {}
                for row, length in enumerate(lengths):
                    if not length:
                        vectors[row] = leaves.new_zeros(4, leaves.shape[-1])
                        continue
                    if self.mode == "random_tree4":
                        digest = hashlib.sha256((str(self.seed) + ":" + str(cpu_ids[row][:length])).encode()).digest()
                        rng = random.Random(int.from_bytes(digest[:8], "little"))
                        shapes = []
                        while len(shapes) < min(4, catalan(length - 1)):
                            tree = tree_for_length(length, "random", rng)
                            if tree not in shapes:
                                shapes.append(tree)
                        shapes = [shapes[i % len(shapes)] for i in range(4)]
                    else:
                        shapes = [tree_for_length(length, kind) for kind in ("left", "right", "balanced", "zigzag")]
                    trees[row] = shapes
                    for slot, tree in enumerate(shapes):
                        groups.setdefault(tree, []).append((row, slot))
                slots = {}
                for tree, entries in groups.items():
                    rows = torch.tensor([entry[0] for entry in entries], device=ids.device)
                    composed = compose_tree(leaves[rows], tree, self.tree_composer)
                    for i, (row, slot) in enumerate(entries):
                        slots[row, slot] = composed[i]
                vectors = torch.stack([vectors[row] if vectors[row] is not None else torch.stack([slots[row, slot] for slot in range(4)])
                                       for row in range(b * h)])
            candidate_mask = valid[:, None].expand(-1, 4)
            mem = dict(semantic=vectors, log_prior=leaves.new_full((b * h, 4), -1.3862943611198906),
                       vector_coeff=leaves.new_zeros(b * h, 4, 0), candidate_mask=candidate_mask,
                       diagnostics={"mean_span_candidates": None})
            if return_trees and trees is not None:
                mem["trees"] = trees
        for key in ("semantic", "log_prior", "vector_coeff", "candidate_mask"):
            mem[key] = mem[key].reshape(b, h, *mem[key].shape[1:])
        # Every method presents four physical memory slots; short forest spans
        # can have fewer than four legal distinct derivations.
        pad = 4 - mem["semantic"].shape[2]
        if pad:
            for key in ("semantic", "vector_coeff"):
                mem[key] = F.pad(mem[key], (0, 0, 0, pad))
            for key in ("log_prior", "candidate_mask"):
                mem[key] = F.pad(mem[key], (0, pad))
        mem["history_mask"] = history_mask
        return mem

    def forward(self, input_ids, history_ids=None, word_mask=None, history_mask=None,
                labels=None, target_mask=None, memory_off=False, ablation=None,
                return_attention=False, return_trees=False, **unused):
        if self.training and ablation is not None:
            raise ValueError("Ablations are evaluation-only")
        memory = None
        if self.mode != "no_memory" and not memory_off and ablation != "memory_off":
            if history_ids is None:
                raise ValueError("History tensors required for memory mode")
            memory = self.encode_history(history_ids, word_mask, history_mask, return_trees)
            if ablation == "shuffled_memory":
                # No target text is ever used in a shuffled memory.
                if input_ids.shape[0] < 2:
                    raise ValueError("Shuffled-memory evaluation requires batch size >=2")
                stories = unused.get("story_ids", list(range(input_ids.shape[0])))
                size = input_ids.shape[0]
                shift = min(range(1, size), key=lambda s: sum(stories[i] == stories[(i - s) % size] for i in range(size)))
                memory["shuffle_valid"] = torch.tensor([stories[i] != stories[(i - shift) % size] for i in range(size)], device=input_ids.device)
                for key in ("semantic", "log_prior", "vector_coeff", "candidate_mask", "history_mask"):
                    memory[key] = memory[key].roll(shift, 0)
            elif ablation == "forest_top1_eval":
                if self.forest is None:
                    raise ValueError("Top-1 ablation requires a trained forest")
                priority = memory["log_prior"].masked_fill(~memory["candidate_mask"], -torch.inf)
                best = priority.argmax(-1, keepdim=True)
                selected = torch.zeros_like(memory["candidate_mask"]).scatter_(-1, best, True)
                memory["candidate_mask"] = selected & memory["candidate_mask"]
                memory["log_prior"] = torch.zeros_like(memory["log_prior"])
            elif ablation not in (None, "normal"):
                raise ValueError(ablation)
        logits, attention = self.decoder(input_ids, memory, self.readers, return_attention)
        result = dict(logits=logits, memory=memory, attention=attention)
        if labels is not None:
            losses = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="none").reshape_as(labels)
            count = labels.ne(-100).sum()
            result.update(loss_sum=losses.sum(), token_count=count, loss=losses.sum() / count.clamp_min(1), token_losses=losses)
        return result


def parameter_report(model):
    # Named parameters removes tied-weight aliases; each real parameter is counted once.
    modules = {}
    for name, parameter in model.named_parameters():
        group = "composer" if name.startswith(("forest.composer.", "tree_composer.")) else name.split(".")[0]
        modules[group] = modules.get(group, 0) + parameter.numel()
    return {"mode": model.mode, "modules": modules, "total": sum(modules.values())}
