"""Batched CKY beam search with differentiable retained scores.

Hard priority selection and pruning are not differentiable. The expansion budget
is a search approximation, NOT exact inference over all binary trees. Exact
short-sentence mode is available via k=None, max_compositions=None, delta=None.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Cell:
    semantic: torch.Tensor  # [N, retained_slots, D]
    score: torch.Tensor
    coeff: torch.Tensor
    mask: torch.Tensor


def masked_log_softmax(scores, mask):
    # An empty history has zero log-prior and an entirely false candidate mask.
    safe = scores.float().masked_fill(~mask, -torch.inf)
    safe = torch.where(mask.any(-1, keepdim=True), safe, torch.zeros_like(safe))
    return F.log_softmax(safe, dim=-1).masked_fill(~mask, 0.0)


def _gather(values, indices):
    # values [N,S,E,...], indices [N,S,P].
    extra = values.shape[3:]
    idx = indices.reshape(*indices.shape, *((1,) * len(extra))).expand(*indices.shape, *extra)
    return values.gather(2, idx)


class ReferenceSemanticForest(nn.Module):
    def __init__(self, composer, k=4, max_compositions=12, delta=8.0, exploration=0.1):
        super().__init__()
        if (k is not None and k < 1) or (max_compositions is not None and max_compositions < 1):
            raise ValueError("Candidate and expansion budgets must be positive or None")
        if (delta is not None and delta < 0) or not 0 <= exploration <= 1:
            raise ValueError("Require delta >=0 and exploration in [0,1]")
        self.composer = composer
        self.k, self.max_compositions, self.delta, self.exploration = k, max_compositions, delta, exploration
        self.implementation = "reference"
        self.capture_spans = False
        self.collect_audit = False

    def _compose(self,left,right):
        policy=getattr(self,"precision_policy",None)
        return self.composer(left,right) if policy is None else policy.compose(self.composer,left,right)

    @staticmethod
    def _span_debug(cell, selected=None, retained=None, metadata=()):
        return dict(semantic=cell.semantic, scores=cell.score, vector_coeff=cell.coeff,
                    candidate_mask=cell.mask, selected_flat=selected, retained_flat=retained,
                    metadata=tuple(metadata))

    def _expand_representations(self, left, right, kl, kr, audit):
        lh, rh = torch.stack([x.semantic for x in left], 1), torch.stack([x.semantic for x in right], 1)
        lu, ru = torch.stack([x.coeff for x in left], 1), torch.stack([x.coeff for x in right], 1)
        lv = lh.unsqueeze(3).expand(-1, -1, -1, kr, -1).flatten(2, 3)
        rv = rh.unsqueeze(2).expand(-1, -1, kl, -1, -1).flatten(2, 3)
        coeff = (lu.unsqueeze(3) + ru.unsqueeze(2)).flatten(2, 3)
        if audit is not None:
            # Storage metadata only, never a device-to-host tensor-value read.
            audit["before_selection_semantic"] += lh.numel() + rh.numel()
            for expanded, original in ((lv, lh), (rv, rh)):
                if expanded.untyped_storage().data_ptr() != original.untyped_storage().data_ptr():
                    audit["before_selection_semantic"] += expanded.numel()
            audit["before_selection_coeff"] += lu.numel() + ru.numel() + coeff.numel()
        return lv, rv, coeff

    def _selected_semantics(self, chart, starts, width, metadata, chosen, representations, audit):
        left = torch.cat([x[0] for x in representations], 2)
        right = torch.cat([x[1] for x in representations], 2)
        lv, rv = _gather(left, chosen), _gather(right, chosen)
        if audit is not None:
            audit["after_selection_semantic"] += left.numel() + right.numel() + lv.numel() + rv.numel()
        return lv, rv, representations

    def _selected_coefficients(self, chart, context, chosen, audit):
        all_coeff = torch.cat([x[2] for x in context], 2)
        coeff = _gather(all_coeff, chosen)
        if audit is not None:
            audit["after_selection_coeff"] += all_coeff.numel() + coeff.numel()
        return coeff

    def forward(self, leaves, span_scores, span_coeff, word_mask, return_trees=False, *, return_spans=False, return_audit=False):
        n, w, d = leaves.shape
        c = span_coeff.shape[-1]
        lengths = word_mask.sum(-1)
        chart, trace, debug_spans = {}, {}, {}
        return_spans = return_spans or self.capture_spans
        audit = dict(before_selection_semantic=0, before_selection_coeff=0,
                     after_selection_semantic=0, after_selection_coeff=0,
                     possible_pair_rows=0, selected_pair_rows=0, composer_calls=0,
                     composer_input_candidate_rows=0) if return_audit or self.collect_audit else None
        total_candidates = leaves.new_zeros((), dtype=torch.float32)
        total_spans = leaves.new_zeros((), dtype=torch.float32)
        for i in range(w):
            leaf_score = span_scores[:, i, i:i + 1]
            mask = (lengths > i).unsqueeze(-1) & torch.isfinite(leaf_score)
            chart[i, i + 1] = Cell(leaves[:, i:i + 1], leaf_score.masked_fill(~mask, 0.0),
                                  span_coeff[:, i, i:i + 1], mask)
            if return_spans:
                debug_spans[i, i + 1] = self._span_debug(chart[i, i + 1])
            total_candidates = total_candidates + mask.sum()
            total_spans = total_spans + mask.sum()
        for width in range(2, w + 1):
            starts = list(range(w - width + 1))
            # Same-width spans share child slot counts. Build scalar priorities
            # before invoking the neural composer, then batch selected pairs.
            scores, masks, representations = [], [], []
            metadata = []
            for offset in range(1, width):
                left = [chart[i, i + offset] for i in starts]
                right = [chart[i + offset, i + width] for i in starts]
                ls, rs = torch.stack([x.score for x in left], 1), torch.stack([x.score for x in right], 1)
                lm, rm = torch.stack([x.mask for x in left], 1), torch.stack([x.mask for x in right], 1)
                kl, kr = ls.shape[-1], rs.shape[-1]
                scores.append((ls.unsqueeze(-1) + rs.unsqueeze(-2)).flatten(2, 3))
                masks.append((lm.unsqueeze(-1) & rm.unsqueeze(-2)).flatten(2, 3))
                representations.append(self._expand_representations(left, right, kl, kr, audit))
                metadata.extend((offset, l, r) for l in range(kl) for r in range(kr))
            score = torch.cat(scores, 2).float()
            valid = torch.cat(masks, 2)
            local = torch.stack([span_scores[:, i, i + width - 1] for i in starts], 1)
            score = score + local.unsqueeze(-1)
            # -inf denotes an unreachable potential. Both implementations share
            # this guard; finite-input search order remains identical to V0.
            valid = valid & torch.isfinite(score)
            score = score.masked_fill(~valid, -torch.inf)
            e = score.shape[-1]
            budget = min(e, self.max_compositions) if self.max_compositions is not None else e
            explore = min(budget - 1, max(1, round(budget * self.exploration))) if self.training and self.exploration > 0 and budget < e else 0
            priority = score.detach()
            chosen = priority.topk(budget - explore, dim=-1).indices
            if explore:
                noise = torch.rand_like(priority).masked_fill(~valid, -torch.inf)
                noise.scatter_(2, chosen, -torch.inf)
                chosen = torch.cat((chosen, noise.topk(explore, dim=-1).indices), -1)
            pair_mask = valid.gather(2, chosen)
            expanded_score = score.gather(2, chosen)
            observer=getattr(self,"precision_observer",None)
            if observer is not None:
                observer(width,expanded_score,pair_mask,min(budget,self.k) if self.k is not None else budget)
            lvec, rvec, context = self._selected_semantics(chart, starts, width, metadata, chosen, representations, audit)
            if audit is not None:
                audit["possible_pair_rows"] += n * len(starts) * e
                audit["selected_pair_rows"] += n * len(starts) * budget
                audit["composer_calls"] += 1
                audit["composer_input_candidate_rows"] += n * len(starts) * budget
            if getattr(self,"precision_diagnostics",False): self.composer._forest_width=width
            semantic = self._compose(lvec, rvec)
            coeff = self._selected_coefficients(chart, context, chosen, audit)
            local_coeff = torch.stack([span_coeff[:, i, i + width - 1] for i in starts], 1)
            coeff = coeff + local_coeff.unsqueeze(2)
            keep = min(budget, self.k) if self.k is not None else budget
            retained_score, rank = expanded_score.topk(keep, dim=-1)
            if self.training and self.exploration > 0 and keep > 1 and budget > keep:
                # Exploration must survive INTO the beam. Merely composing a
                # low-score pair and then taking score top-k would never explore.
                eligible = pair_mask.clone()
                eligible.scatter_(2, rank[..., :-1], False)
                if self.delta is not None:
                    eligible &= expanded_score >= retained_score[..., :1] - self.delta
                random_priority = torch.rand_like(expanded_score).masked_fill(~eligible, -torch.inf)
                random_rank = random_priority.argmax(-1, keepdim=True)
                use_random = (torch.rand_like(retained_score[..., :1]) < self.exploration) & eligible.any(-1, keepdim=True)
                last = torch.where(use_random, random_rank, rank[..., -1:])
                rank = torch.cat((rank[..., :-1], last), -1)
                retained_score = expanded_score.gather(2, rank)
            retained_mask = pair_mask.gather(2, rank)
            if self.delta is not None:
                best = retained_score[..., :1]
                retained_mask = retained_mask & (retained_score >= best - self.delta)
            retained_h, retained_u = _gather(semantic, rank), _gather(coeff, rank)
            # Zero invalid slots, avoiding NaNs in empty-history computations.
            retained_score = retained_score.masked_fill(~retained_mask, 0.0)
            retained_h = retained_h * retained_mask.unsqueeze(-1)
            retained_u = retained_u * retained_mask.unsqueeze(-1)
            for s, i in enumerate(starts):
                chart[i, i + width] = Cell(retained_h[:, s], retained_score[:, s], retained_u[:, s], retained_mask[:, s])
                if return_trees:
                    trace[i, i + width] = (chosen.gather(2, rank)[:, s].detach().cpu(), metadata)
                if return_spans:
                    debug_spans[i, i + width] = self._span_debug(
                        chart[i, i + width], chosen[:, s], chosen.gather(2, rank)[:, s], metadata)
            total_candidates = total_candidates + retained_mask.sum()
            total_spans = total_spans + (lengths[:, None] >= torch.tensor([i + width for i in starts], device=lengths.device)).sum()
        root_slots = max(chart[0, length].score.shape[-1] for length in range(1, w + 1))
        hs, ss, us, ms = [], [], [], []
        for length in range(1, w + 1):
            cell = chart[0, length]
            pad = root_slots - cell.score.shape[-1]
            hs.append(F.pad(cell.semantic, (0, 0, 0, pad)))
            ss.append(F.pad(cell.score, (0, pad)))
            us.append(F.pad(cell.coeff, (0, 0, 0, pad)))
            ms.append(F.pad(cell.mask, (0, pad)))
        row = torch.arange(n, device=leaves.device)
        index = lengths.clamp_min(1) - 1
        root_h = torch.stack(hs, 1)[row, index]
        root_s = torch.stack(ss, 1)[row, index]
        root_u = torch.stack(us, 1)[row, index]
        root_m = torch.stack(ms, 1)[row, index] & lengths.ne(0).unsqueeze(-1)
        result = dict(semantic=root_h, scores=root_s, vector_coeff=root_u,
                      candidate_mask=root_m, log_prior=masked_log_softmax(root_s, root_m),
                      diagnostics={"mean_span_candidates": total_candidates / total_spans.clamp_min(1),
                                   "span_candidates_sum": total_candidates, "span_count": total_spans})
        if return_spans:
            result["spans"] = debug_spans
        if audit is not None:
            audit["total_materialized_semantic"] = audit["before_selection_semantic"] + audit["after_selection_semantic"]
            audit["total_materialized_coeff"] = audit["before_selection_coeff"] + audit["after_selection_coeff"]
            result["materialization_audit"] = audit
        if return_trees:
            def build(row_id, i, j, candidate):
                if j - i == 1:
                    return i
                indices, meta = trace[i, j]
                offset, l, r = meta[int(indices[row_id, candidate])]
                return (build(row_id, i, i + offset, l), build(row_id, i + offset, j, r))
            result["trees"] = [[build(r, 0, int(length), k) for k in range(root_slots) if root_m[r, k]]
                               if length else [] for r, length in enumerate(lengths.detach().cpu().tolist())]
        return result


def catalan(n):
    return math.comb(2 * n, n) // (n + 1)


def tree_for_length(length, kind, rng=None):
    def build(i, j, depth=0):
        size = j - i
        if size == 1:
            return i
        if kind == "left":
            k = j - 1
        elif kind == "right":
            k = i + 1
        elif kind == "balanced":
            k = i + size // 2
        elif kind == "zigzag":
            k = i + 1 if depth % 2 == 0 else j - 1
        elif kind == "random":
            # Catalan weights yield a uniform distribution over binary shapes.
            choices = list(range(1, size))
            weights = [catalan(s - 1) * catalan(size - s - 1) for s in choices]
            k = i + rng.choices(choices, weights=weights, k=1)[0]
        else:
            raise ValueError(kind)
        return (build(i, k, depth + 1), build(k, j, depth + 1))
    return build(0, length)


def compose_tree(leaves, tree, composer):
    if isinstance(tree, int):
        return leaves[..., tree, :]
    return composer(compose_tree(leaves, tree[0], composer), compose_tree(leaves, tree[1], composer))
