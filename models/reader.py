"""Multi-head semantic reader with within-sentence syntax log-prior bias."""
from __future__ import annotations

from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


class SemanticReader(nn.Module):
    def __init__(self, cfg, vector=False):
        super().__init__()
        d, s, width = cfg["hidden_size"], cfg["semantic_size"], cfg["reader_width"]
        self.heads, self.width = cfg["reader_heads"], width
        assert width % self.heads == 0
        self.beta, self.dropout = cfg["prior_beta"], cfg["dropout"]
        self.norm = nn.LayerNorm(d)
        self.query = nn.Linear(d, width, bias=False)
        self.key = nn.Linear(s + (cfg["coefficient_size"] if vector else 0), width, bias=False)
        self.value = nn.Linear(s, width, bias=False)
        self.output = nn.Linear(width, d, bias=False)
        self.sentence_bias = nn.Parameter(torch.zeros(self.heads, cfg["max_history"]))
        self.gate = nn.Parameter(torch.tensor(float(cfg["adapter_gate"])))
        self.vector = vector

    def forward(self, hidden, memory, return_attention=False):
        mode=getattr(self,"precision_mode","bf16")
        if mode=="bf16":
            return self._forward(hidden,memory,return_attention)
        if mode!="fp32_ieee": raise ValueError(f"Unknown Reader precision: {mode}")
        if hidden.is_cuda and torch.backends.cuda.matmul.fp32_precision!="ieee":
            raise RuntimeError("FP32 Reader requires process-wide IEEE matmul; configure backend at startup")
        original_dtype=hidden.dtype
        with torch.autocast(hidden.device.type,enabled=False):
            # Casts remain differentiable; no mutation of the forest memory.
            work=dict(memory)
            for key in ("semantic","vector_coeff","log_prior"):work[key]=memory[key].float()
            output,weights=self._forward(hidden.float(),work,return_attention)
        return output.to(original_dtype),weights

    def _forward(self, hidden, memory, return_attention=False):
        observer=getattr(self,"precision_observer",None)
        b, t, d = hidden.shape
        _, h, k, _ = memory["semantic"].shape
        semantic = memory["semantic"].flatten(1, 2)
        keys = torch.cat((semantic, memory["vector_coeff"].flatten(1, 2)), -1) if self.vector else semantic
        hd = self.width // self.heads
        q = self.query(self.norm(hidden)).reshape(b, t, self.heads, hd).transpose(1, 2)
        key = self.key(keys).reshape(b, h * k, self.heads, hd).transpose(1, 2)
        value = self.value(semantic).reshape(b, h * k, self.heads, hd).transpose(1, 2)
        if observer is not None:
            observer.record("query_input",hidden)
            observer.record("query_projection",q)
            observer.record("key_input",keys)
            observer.record("key_projection",key)
            observer.record("value_input",semantic)
            observer.record("value_projection",value)
        valid = (memory["candidate_mask"] & memory["history_mask"].unsqueeze(-1)).flatten(1, 2)
        # Relative distance from the target, not candidate-specific time positions.
        count = memory["history_mask"].sum(-1)
        distances = (count[:, None] - 1 - torch.arange(h, device=hidden.device)).clamp_min(0)
        positions = self.sentence_bias[:, distances].permute(1, 0, 2)
        bias = self.beta * memory["log_prior"].flatten(1, 2)[:, None, :] + positions.repeat_interleave(k, -1)
        bias = bias[:, :, None, :].float().masked_fill(~valid[:, None, None, :], -torch.inf)
        has_memory = valid.any(-1)
        # Add a safe, masked-out-afterwards sentinel for empty rows only.
        safe = valid.clone()
        safe[:, 0] |= ~has_memory
        bias = torch.where((~has_memory)[:, None, None, None] & safe[:, None, None, :], torch.zeros_like(bias), bias)
        if return_attention:
            logits = torch.matmul(q.float(), key.float().transpose(-2, -1)) / hd**0.5 + bias
            weights = logits.softmax(-1) * valid[:, None, None, :]
            dropped = F.dropout(weights, self.dropout, self.training)
            output = torch.matmul(dropped.to(value.dtype), value)
        else:
            # Learnable masks must receive syntax-prior and position gradients.
            # Fused mask derivatives differ across PyTorch/CUDA combinations.
            backend = sdpa_kernel(SDPBackend.MATH) if bias.requires_grad else nullcontext()
            sdpa_bias=bias.to(q.dtype)
            trace=observer.attention_context() if observer is not None else nullcontext()
            with backend,trace:
                if observer is not None:
                    observer.sdpa(q,key,value,sdpa_bias,"math" if bias.requires_grad else "automatic")
                output = F.scaled_dot_product_attention(q, key, value, attn_mask=sdpa_bias,
                                                        dropout_p=self.dropout if self.training else 0.0)
            weights = None
            if observer is not None: observer.record("attention_weighted_values",output)
        output = output.transpose(1, 2).reshape(b, t, self.width)
        if observer is not None: observer.record("output_projection_input",output)
        output = self.output(output)
        if observer is not None: observer.record("output_projection",output)
        output = output * has_memory[:, None, None]
        result=hidden + torch.sigmoid(self.gate) * output
        if observer is not None: observer.record("residual_output",result)
        return result, weights
