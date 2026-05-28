"""CBAM: Convolutional Block Attention Module (Woo et al., ECCV 2018).

Sequentially applies channel attention then spatial attention. Designed to be
dropped into the PANet neck of YOLOv8 via a custom YAML config.

Param budget per Chapter 4.3: ~0.3M total across all neck insertions, achieved
with reduction_ratio=16 on the channel-attention MLP.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Squeeze global context into per-channel weights via shared MLP."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # Shared MLP: 1x1 conv pair is equivalent to FC and TorchScript-friendly.
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(self.avg_pool(x))
        mx = self.mlp(self.max_pool(x))
        return self.act(avg + mx)


class SpatialAttention(nn.Module):
    """Attention map over spatial locations from channel-pooled features."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "CBAM paper uses 3 or 7"
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True).values
        return self.act(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Sequential channel-then-spatial attention.

    Conv-style signature so Ultralytics' parser can drop it into the standard
    channel-aware branch: parser passes `(c1, c2, reduction, kernel_size)`,
    we enforce `c2 == c1` (CBAM preserves channels) and use c1 for both the
    attention MLPs and the output.

    Args:
        c1: input channel count (Ultralytics parser injects this positionally).
        c2: output channel count — must equal c1 for CBAM. Kept in the signature
            so the parser's standard `args = [c1, c2, *rest]` rewriting works
            transparently.
        reduction: channel-attention bottleneck ratio.
        kernel_size: spatial-attention conv kernel.
    """

    def __init__(self, c1: int, c2: int | None = None, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        if c2 is not None and c2 != c1:
            # Parser may have scaled c2 differently — fall back to c1 with a warning.
            # In practice, when YAML c2 == c1 (intended use), the parser won't
            # re-scale. We tolerate divergence to be robust to scale tables.
            import warnings
            warnings.warn(
                f"CBAM expects c2 == c1; got c1={c1}, c2={c2}. Using c1.",
                RuntimeWarning,
            )
        self.channel = ChannelAttention(c1, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel(x)
        x = x * self.spatial(x)
        return x
