"""Local, ordered history encoding before the shared four-slot pooling reader."""
from __future__ import annotations

from torch import nn
from torch.nn import functional as F


class SequentialCNN(nn.Module):
    """Stride-one, non-dilated convolutions; normalization is token-local.

    Both directions are available because these are completed history sentences.
    Mask after every block: padded activations must not leak back into valid
    positions in a subsequent convolution. Pooling is deliberately outside this
    encoder so pooling4 and seq_cnn4 use the same memory compression mechanism.
    """

    def __init__(self, semantic_size, kernel_size=3, layers=2, hidden_size=None):
        super().__init__()
        if not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("cnn_kernel_size must be a positive odd integer")
        if not isinstance(layers, int) or layers < 1:
            raise ValueError("cnn_layers must be a positive integer")
        hidden_size = semantic_size if hidden_size is None else hidden_size
        if not isinstance(hidden_size, int) or hidden_size < 1:
            raise ValueError("cnn_hidden_size must be a positive integer")
        self.receptive_field = 1 + layers * (kernel_size - 1)
        self.input = nn.Identity() if hidden_size == semantic_size else nn.Linear(semantic_size, hidden_size)
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2)
            for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(layers)])
        self.output = nn.Identity() if hidden_size == semantic_size else nn.Linear(hidden_size, semantic_size)

    def forward(self, leaves, mask):
        active = mask.unsqueeze(-1)
        x = self.input(leaves.masked_fill(~active, 0)).masked_fill(~active, 0)
        for conv, norm in zip(self.convs, self.norms):
            local = conv(x.transpose(1, 2)).transpose(1, 2)
            x = norm(x + F.gelu(local)).masked_fill(~active, 0)
        return self.output(x).masked_fill(~active, 0)
