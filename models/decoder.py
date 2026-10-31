"""Pre-LN GPT with tied embeddings and native causal SDPA (no custom kernels)."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalAttention(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        assert width % heads == 0
        self.heads, self.dropout = heads, dropout
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        # Right-padded keys never precede a valid query, so causal masking alone
        # is sufficient; no future target token can be read.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                          dropout_p=self.dropout if self.training else 0.0)
        return self.residual_dropout(self.output(y.transpose(1, 2).reshape(b, t, d)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["hidden_size"]
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attention = CausalAttention(d, cfg["heads"], cfg["dropout"])
        self.mlp = nn.Sequential(nn.Linear(d, cfg["ffn_size"]), nn.GELU(),
                                 nn.Linear(cfg["ffn_size"], d), nn.Dropout(cfg["dropout"]))

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg["hidden_size"]
        self.embedding = nn.Embedding(cfg["vocab_size"], d)
        self.position = nn.Embedding(cfg["max_target_length"], d)
        self.dropout = nn.Dropout(cfg["dropout"])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["layers"])])
        self.norm = nn.LayerNorm(d)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, memory=None, readers=None, return_attention=False):
        t = input_ids.shape[1]
        if t > self.cfg["max_target_length"]:
            raise ValueError("Target exceeds positional embedding capacity")
        x = self.dropout(self.embedding(input_ids) + self.position(torch.arange(t, device=input_ids.device)))
        attention = []
        for layer, block in enumerate(self.blocks, 1):
            x = block(x)
            if memory is not None and readers is not None and str(layer) in readers:
                x, weights = readers[str(layer)](x, memory, return_attention)
                if weights is not None:
                    attention.append(weights)
        # The LM head shares exactly the input embedding parameter.
        return F.linear(self.norm(x), self.embedding.weight), attention
