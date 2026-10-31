"""V2 delayed gather; Cell dictionaries and shared search stay unchanged."""
from __future__ import annotations

import torch
from models.forest_reference import ReferenceSemanticForest


def _bank_gather(bank, indices):
    n, spans, selected = indices.shape
    dim = bank.shape[-1]
    index = indices.reshape(n, spans * selected, 1).expand(n, spans * selected, dim)
    return bank.gather(1, index).reshape(n, spans, selected, dim)


class SemanticForest(ReferenceSemanticForest):
    def __init__(self, composer, k=4, max_compositions=12, delta=8.0, exploration=0.1,
                 implementation="score_first"):
        super().__init__(composer, k, max_compositions, delta, exploration)
        if implementation not in ("reference", "score_first"):
            raise ValueError(f"Unknown forest implementation: {implementation}")
        self.implementation = implementation
        self._lookup_cache = {}

    def _apply(self, fn, recurse=True):
        self._lookup_cache.clear()
        return super()._apply(fn, recurse)

    def _expand_representations(self, left, right, kl, kr, audit):
        if self.implementation == "reference":
            return super()._expand_representations(left, right, kl, kr, audit)
        # Until Top-R, V2 accesses only score/mask fields in the base loop.
        return None

    def _lookup(self, chart, starts, width, metadata, device):
        # Actual child slot counts vary by width, K and R; a fixed split*K*K
        # formula would incorrectly decode leaves and underfull beams.
        counts = tuple(chart[starts[0], starts[0] + offset].score.shape[-1]
                       for offset in range(1, width))
        cache_key = (len(starts), width, counts, device)
        cached = self._lookup_cache.get(cache_key)
        if cached is None:
            keys, bases, slots = [], {}, 0
            for offset in range(1, width):
                for i in starts:
                    for key in ((i, i + offset), (i + offset, i + width)):
                        if key not in bases:
                            bases[key] = slots
                            keys.append(key)
                            slots += chart[key].score.shape[-1]
            lookup = [[(bases[i, i + offset] + left,
                        bases[i + offset, i + width] + right)
                       for offset, left, right in metadata] for i in starts]
            # Shape-only constants, never GPU-to-CPU reads or candidate sync.
            table = torch.tensor(lookup, dtype=torch.long, device=device)
            cached = (tuple(keys), table)
            self._lookup_cache[cache_key] = cached
        return cached

    def _selected_semantics(self, chart, starts, width, metadata, chosen, representations, audit):
        if self.implementation == "reference":
            return super()._selected_semantics(chart, starts, width, metadata, chosen, representations, audit)
        keys, table = self._lookup(chart, starts, width, metadata, chosen.device)
        decoded = table.unsqueeze(0).expand(chosen.shape[0], -1, -1, -1).gather(
            2, chosen.unsqueeze(-1).expand(*chosen.shape, 2))
        left_index, right_index = decoded[..., 0], decoded[..., 1]
        # Temporary read-only bank from existing Cells, built AFTER selection.
        # Each source Cell appears once. This is not a tensorized chart and
        # never builds [split,K,K,D] representations or writes chart in-place.
        bank = torch.cat([chart[key].semantic for key in keys], 1)
        lv, rv = _bank_gather(bank, left_index), _bank_gather(bank, right_index)
        if audit is not None:
            audit["after_selection_semantic"] += bank.numel() + lv.numel() + rv.numel()
        return lv, rv, (keys, left_index, right_index)

    def _selected_coefficients(self, chart, context, chosen, audit):
        if self.implementation == "reference":
            return super()._selected_coefficients(chart, context, chosen, audit)
        keys, left_index, right_index = context
        bank = torch.cat([chart[key].coeff for key in keys], 1)
        left, right = _bank_gather(bank, left_index), _bank_gather(bank, right_index)
        coeff = left + right  # Same (u_L + u_R) + u_parent arithmetic order.
        if audit is not None:
            audit["after_selection_coeff"] += bank.numel() + left.numel() + right.numel() + coeff.numel()
        return coeff
