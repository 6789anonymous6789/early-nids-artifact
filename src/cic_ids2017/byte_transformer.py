"""Mini-Transformer for raw-byte flow classification (CIC-IDS2017).

Architecture:
  bytes (B, P, n_bytes) /255 -> Linear -> (B, P, d_model)
  + Time-Aware Sinusoidal PE on relative timestamps
  -> N x TransformerEncoderLayer (d_model, nhead, dim_ff, dropout)
  -> masked mean pool over packets
  -> Linear -> n_classes
"""
from __future__ import annotations

import math

import torch
from torch import nn


def time_aware_sinusoidal_pe(
    t_rel: torch.Tensor,
    d_model: int,
    base: float = 10_000.0,
    t_clip: float = 60.0,
    t_scale: float = 10.0,
) -> torch.Tensor:
    """Continuous-time sinusoidal positional encoding.

    Parameters
    ----------
    t_rel : (B, P) float32 — seconds since flow start, padding positions == 0.
    d_model : int (must be even).
    base, t_clip, t_scale : see docs.

    Returns
    -------
    pe : (B, P, d_model) float32
    """
    assert d_model % 2 == 0, "d_model must be even"
    t = torch.clamp(t_rel, min=0.0, max=t_clip) / t_scale            # (B, P)
    half = d_model // 2
    i = torch.arange(half, device=t.device, dtype=t.dtype)
    omega = base ** (-2 * i / d_model)                                # (half,)
    angles = t.unsqueeze(-1) * omega                                  # (B, P, half)
    sin_part = torch.sin(angles)
    cos_part = torch.cos(angles)
    pe = torch.empty(t.shape[0], t.shape[1], d_model, device=t.device, dtype=t.dtype)
    pe[..., 0::2] = sin_part
    pe[..., 1::2] = cos_part
    return pe


class ByteTransformer(nn.Module):
    def __init__(
        self,
        n_classes: int,
        n_bytes: int = 256,
        max_pkts: int = 5,
        d_model: int = 16,
        nhead: int = 4,
        dim_feedforward: int = 32,
        n_layers: int = 2,
        dropout: float = 0.1,
        t_clip: float = 60.0,
        t_scale: float = 10.0,
    ):
        super().__init__()
        self.n_bytes = n_bytes
        self.max_pkts = max_pkts
        self.d_model = d_model
        self.t_clip = t_clip
        self.t_scale = t_scale

        self.byte_proj = nn.Linear(n_bytes, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(
        self,
        x_bytes: torch.Tensor,    # (B, P, n_bytes) uint8
        pad_mask: torch.Tensor,   # (B, P) bool, True = padding (ignore)
        t_rel: torch.Tensor,      # (B, P) float32
    ) -> torch.Tensor:
        # Normalize bytes to [0, 1]
        x = x_bytes.float() / 255.0
        # Embed
        h = self.byte_proj(x)                                         # (B, P, d_model)
        # Add TA-PE
        pe = time_aware_sinusoidal_pe(
            t_rel, self.d_model, t_clip=self.t_clip, t_scale=self.t_scale
        )
        # Zero-out PE at padding positions so the contribution is null
        pe = pe.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        h = h + pe
        # Transformer encoder with key padding mask
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        # Masked mean pool (avoid div-by-zero with at-least-one trick)
        valid = (~pad_mask).float().unsqueeze(-1)                     # (B, P, 1)
        valid_count = valid.sum(dim=1).clamp(min=1.0)                 # (B, 1)
        pooled = (h * valid).sum(dim=1) / valid_count                 # (B, d_model)
        pooled = self.norm(pooled)
        logits = self.head(pooled)
        return logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
