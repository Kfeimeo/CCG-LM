"""LLaMA2 / LLaMA3 / Qwen3 style decoders trained from scratch.

RMSNorm, rotary positions, SwiGLU, optional grouped-query attention and QK-norm.
Same forward interface and Reader insertion points as the GPT decoder in
models/decoder.py; `decoder_arch: gpt2` (the default) keeps that decoder and its
checkpoint keys untouched. No pretrained weights are loaded.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.decoder import Decoder

ARCH_PRESETS = {
    # kv_divisor None -> multi-head attention (kv_heads == heads); otherwise kv_heads = heads // divisor.
    "llama2": dict(kv_divisor=None, rope_theta=10000.0, qk_norm=False, tie_embeddings=False, norm_eps=1e-5),
    "llama3": dict(kv_divisor=4, rope_theta=500000.0, qk_norm=False, tie_embeddings=False, norm_eps=1e-5),
    "qwen3": dict(kv_divisor=2, rope_theta=1000000.0, qk_norm=True, tie_embeddings=True, norm_eps=1e-6),
}
DECODER_ARCHS = ("gpt2",) + tuple(ARCH_PRESETS)
OVERRIDE_KEYS = ("kv_heads", "rope_theta", "qk_norm", "tie_embeddings", "norm_eps")


def resolve_arch(cfg):
    """Merge the architecture preset with explicit config overrides."""
    arch = cfg.get("decoder_arch", "gpt2")
    if arch not in DECODER_ARCHS:
        raise ValueError(f"Unsupported decoder_arch {arch!r}; choose from {DECODER_ARCHS}")
    if arch == "gpt2":
        return dict(arch=arch)
    preset = ARCH_PRESETS[arch]
    heads = cfg["heads"]
    kv_default = heads if preset["kv_divisor"] is None else max(1, heads // preset["kv_divisor"])
    out = dict(arch=arch,
               kv_heads=int(cfg.get("kv_heads") or kv_default),
               rope_theta=float(cfg.get("rope_theta", preset["rope_theta"])),
               qk_norm=bool(cfg.get("qk_norm", preset["qk_norm"])),
               tie_embeddings=bool(cfg.get("tie_embeddings", preset["tie_embeddings"])),
               norm_eps=float(cfg.get("norm_eps", preset["norm_eps"])))
    if cfg["hidden_size"] % heads:
        raise ValueError("hidden_size must be divisible by heads")
    if heads % out["kv_heads"]:
        raise ValueError(f"heads ({heads}) must be a multiple of kv_heads ({out['kv_heads']})")
    if (cfg["hidden_size"] // heads) % 2:
        raise ValueError("RoPE needs an even head dimension")
    return out


class RotaryEmbedding(nn.Module):
    """Half-rotation RoPE (the HF LLaMA/Qwen convention); tables are non-persistent buffers."""

    def __init__(self, head_dim, max_positions, theta):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        angles = torch.outer(torch.arange(max_positions, dtype=torch.float32), inv_freq)
        emb = torch.cat((angles, angles), -1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), -1)

    def forward(self, q, k):
        # q, k: (b, heads, t, hd). Rotate in FP32 and return in the input dtype.
        t = q.shape[-2]
        cos, sin = self.cos[:t][None, None], self.sin[:t][None, None]
        qf, kf = q.float(), k.float()
        q_out = qf * cos + self._rotate_half(qf) * sin
        k_out = kf * cos + self._rotate_half(kf) * sin
        return q_out.to(q.dtype), k_out.to(k.dtype)


class GroupedCausalAttention(nn.Module):
    def __init__(self, width, heads, kv_heads, dropout, rotary, qk_norm, norm_eps):
        super().__init__()
        self.heads, self.kv_heads, self.dropout = heads, kv_heads, dropout
        self.head_dim = width // heads
        self.q_proj = nn.Linear(width, heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(width, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(width, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(heads * self.head_dim, width, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else None
        self.rotary = rotary
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, t, _ = x.shape
        q = self.q_proj(x).reshape(b, t, self.heads, self.head_dim)
        k = self.k_proj(x).reshape(b, t, self.kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(b, t, self.kv_heads, self.head_dim)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k = self.rotary(q, k)
        # Right-padded keys never precede a valid query, so causal masking alone is sufficient.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                          dropout_p=self.dropout if self.training else 0.0,
                                          enable_gqa=self.kv_heads != self.heads)
        return self.residual_dropout(self.o_proj(y.transpose(1, 2).reshape(b, t, self.heads * self.head_dim)))


class SwiGLU(nn.Module):
    def __init__(self, width, hidden, dropout):
        super().__init__()
        self.gate_proj = nn.Linear(width, hidden, bias=False)
        self.up_proj = nn.Linear(width, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class LlamaBlock(nn.Module):
    def __init__(self, cfg, arch, rotary):
        super().__init__()
        d = cfg["hidden_size"]
        self.ln1, self.ln2 = nn.RMSNorm(d, eps=arch["norm_eps"]), nn.RMSNorm(d, eps=arch["norm_eps"])
        self.attention = GroupedCausalAttention(d, cfg["heads"], arch["kv_heads"], cfg["dropout"], rotary,
                                                arch["qk_norm"], arch["norm_eps"])
        self.mlp = SwiGLU(d, cfg["ffn_size"], cfg["dropout"])

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class LlamaDecoder(nn.Module):
    """Pre-RMSNorm decoder with RoPE and SwiGLU; `decoder_arch` selects the LLaMA2/LLaMA3/Qwen3 preset."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg, self.arch = cfg, resolve_arch(cfg)
        d = cfg["hidden_size"]
        self.embedding = nn.Embedding(cfg["vocab_size"], d)
        self.dropout = nn.Dropout(cfg["dropout"])
        self.rotary = RotaryEmbedding(d // cfg["heads"], cfg["max_target_length"], self.arch["rope_theta"])
        self.blocks = nn.ModuleList([LlamaBlock(cfg, self.arch, self.rotary) for _ in range(cfg["layers"])])
        self.norm = nn.RMSNorm(d, eps=self.arch["norm_eps"])
        self.lm_head = None if self.arch["tie_embeddings"] else nn.Linear(d, cfg["vocab_size"], bias=False)
        self.apply(Decoder._init)

    def forward(self, input_ids, memory=None, readers=None, return_attention=False):
        t = input_ids.shape[1]
        if t > self.cfg["max_target_length"]:
            raise ValueError("Target exceeds rotary table capacity")
        x = self.dropout(self.embedding(input_ids))
        attention = []
        for layer, block in enumerate(self.blocks, 1):
            x = block(x)
            if memory is not None and readers is not None and str(layer) in readers:
                x, weights = readers[str(layer)](x, memory, return_attention)
                if weights is not None:
                    attention.append(weights)
        x = self.norm(x)
        if self.lm_head is None:
            return F.linear(x, self.embedding.weight), attention
        return self.lm_head(x), attention


def build_decoder(cfg):
    """`decoder_arch: gpt2` (default) keeps the original GPT decoder and its checkpoint keys."""
    return Decoder(cfg) if resolve_arch(cfg)["arch"] == "gpt2" else LlamaDecoder(cfg)
