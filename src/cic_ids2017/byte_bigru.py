"""BiGRU sequential baseline for raw-byte flow classification (CIC-IDS2017).

Same input pipeline as the byte_transformer, different inductive bias:
recurrence over packets instead of self-attention.

Architecture:
  bytes (B, P, n_bytes) /255 -> Linear -> (B, P, d_model)
  -> bidirectional GRU (1 layer, hidden=d_model)
  -> masked mean pool over packets (using last hidden states is also possible
     but mean pool is comparable to the Transformer)
  -> Linear -> n_classes
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ByteBiGRU(nn.Module):
    def __init__(
        self,
        n_classes: int,
        n_bytes: int = 256,
        max_pkts: int = 5,
        d_model: int = 64,
        hidden_size: int | None = None,
        n_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_bytes = n_bytes
        self.max_pkts = max_pkts
        self.d_model = d_model
        h = hidden_size if hidden_size is not None else d_model

        self.byte_proj = nn.Linear(n_bytes, d_model)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=h,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(2 * h)
        self.head = nn.Linear(2 * h, n_classes)

    def forward(
        self,
        x_bytes: torch.Tensor,    # (B, P, n_bytes) uint8
        pad_mask: torch.Tensor,   # (B, P) bool, True = padding
        t_rel: torch.Tensor,      # unused (kept for interface parity)
    ) -> torch.Tensor:
        x = x_bytes.float() / 255.0
        h = self.byte_proj(x)                                         # (B, P, d_model)

        # Compute true sequence lengths from pad_mask
        lengths = (~pad_mask).sum(dim=1).clamp(min=1).cpu()           # (B,)

        # pack -> GRU -> unpack
        packed = pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
        out_packed, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=h.size(1))
        # out: (B, P, 2h)

        # Masked mean pool
        valid = (~pad_mask).float().unsqueeze(-1)                     # (B, P, 1)
        valid_count = valid.sum(dim=1).clamp(min=1.0)                 # (B, 1)
        pooled = (out * valid).sum(dim=1) / valid_count
        pooled = self.norm(pooled)
        return self.head(pooled)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
