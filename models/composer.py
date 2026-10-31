"""Shared ordered binary composition; syntax states never enter this module."""
from torch import nn
import torch


class Composer(nn.Module):
    def __init__(self, semantic_size=256, hidden_size=512):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * semantic_size, hidden_size), nn.GELU(),
                                 nn.Linear(hidden_size, semantic_size))
        self.gate = nn.Linear(2 * semantic_size, semantic_size)
        self.left_residual = nn.Linear(semantic_size, semantic_size, bias=False)
        self.right_residual = nn.Linear(semantic_size, semantic_size, bias=False)
        self.norm = nn.LayerNorm(semantic_size)

    def forward(self, left, right):
        ordered = torch.cat((left, right), -1)
        gate = torch.sigmoid(self.gate(ordered))
        residual = self.left_residual(left) + self.right_residual(right)
        result = self.norm(residual + gate * self.mlp(ordered))
        observer=getattr(self,"precision_trace",None)
        if observer is not None: observer(self,left,right,gate,residual,result)
        return result
