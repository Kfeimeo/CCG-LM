"""History-only syntax channel and local (not globally normalized) potentials."""
from __future__ import annotations

import torch
from torch import nn


class SpanScorer(nn.Module):
    def __init__(self, hidden, max_words, coefficient_size=0):
        super().__init__()
        self.max_words, self.coefficient_size = max_words, coefficient_size
        self.length_embedding = nn.Embedding(max_words + 1, 32)
        self.features = nn.Sequential(nn.Linear(2 * hidden + 32, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.scalar = nn.Linear(hidden, 1)
        self.vector = nn.Linear(hidden, coefficient_size) if coefficient_size else None

    def forward(self, hidden):
        n, w, d = hidden.shape
        left = hidden[:, :, None, :].expand(n, w, w, d)
        right = hidden[:, None, :, :].expand(n, w, w, d)
        lengths = (torch.arange(w, device=hidden.device)[None, :] - torch.arange(w, device=hidden.device)[:, None] + 1).clamp(0)
        feature = self.features(torch.cat((left, right, self.length_embedding(lengths).expand(n, -1, -1, -1)), -1))
        score = self.scalar(feature).squeeze(-1)
        coeff = self.vector(feature) if self.vector is not None else hidden.new_zeros(n, w, w, 0)
        # Tensor index [i,j-1] represents the half-open interval [i,j).
        return score, coeff


class SyntaxEncoder(nn.Module):
    def __init__(self, cfg, vector=False):
        super().__init__()
        d = cfg["syntax_hidden"]
        self.embedding = nn.Embedding(cfg["word_vocab_size"], cfg["syntax_embedding"], padding_idx=0)
        self.projection = nn.Linear(cfg["syntax_embedding"], d)
        self.position = nn.Embedding(cfg["max_words"], d)
        layer = nn.TransformerEncoderLayer(d, cfg["syntax_heads"], 4 * d, cfg["dropout"],
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg["syntax_layers"], norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.span_scorer = SpanScorer(d, cfg["max_words"], cfg["coefficient_size"] if vector else 0)

    def forward(self, ids, mask):
        n, w = ids.shape
        if w > self.position.num_embeddings:
            raise ValueError("History sentence exceeds max_words")
        safe_mask = mask.clone()
        safe_mask[:, 0] = True  # Avoid all-masked self attention for empty histories.
        x = self.projection(self.embedding(ids)) + self.position(torch.arange(w, device=ids.device))
        x = self.encoder(x, src_key_padding_mask=~safe_mask)
        x = x * mask.unsqueeze(-1)
        return self.span_scorer(x)
