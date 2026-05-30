"""CBAM: Convolutional Block Attention Module (Woo et al., ECCV 2018).

Author: Rutuja Kulkarni

This module implements enhancement #2 of the RADS Layer 3 pipeline (thesis
Chapter 4.3): a lightweight attention block inserted into the PANet neck of
YOLOv8 to help the detector focus on the salient channels and spatial regions
that distinguish manholes, potholes, and water-logged potholes from cluttered
road backgrounds. CBAM refines a feature map by first reweighting its channels
(what to emphasise) and then reweighting its spatial positions (where to look),
each via a small attention sub-network with a sigmoid gate.

How it fits the bigger picture: in line with the project's architecture
philosophy of "custom YAML + monkey-patch over forking Ultralytics", this file
defines a self-contained `nn.Module` that knows nothing about Ultralytics.
src/modules/register.py is what teaches Ultralytics' model parser to recognise
the `CBAM` name in a model YAML, so this enhancement remains a reversible toggle
(present only when the chosen YAML references it). The combined-enhancement
teacher used in the knowledge-distillation stage stacks this on top of the P2
head and size-aware loss.

Param budget per Chapter 4.3: ~0.3M total across all neck insertions, achieved
with reduction_ratio=16 on the channel-attention MLP.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Squeeze global context into per-channel weights via shared MLP.

    Channel attention answers "which feature channels matter for this image?"
    It collapses each channel's full spatial extent into a single descriptor
    (once by average pooling, once by max pooling), pushes both descriptors
    through a shared bottleneck MLP, and sigmoid-gates the result into a
    per-channel multiplier in (0, 1). Using both avg- and max-pooled summaries
    (as in the CBAM paper) captures complementary cues — overall context plus
    the single most salient response — which empirically beats either alone.

    Args:
        channels: number of input feature channels C (also the output count,
            since attention only reweights, never resizes, the channel axis).
        reduction: bottleneck ratio for the shared MLP; larger means fewer
            hidden units and fewer params. 16 is the project default that hits
            the ~0.3M neck-wide parameter budget (Chapter 4.3).
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        # Bottleneck width = C / reduction, floored at 8 so very thin feature
        # maps still get a usable hidden layer instead of collapsing to ~1 unit.
        hidden = max(channels // reduction, 8)
        # Two global pools reduce HxW -> 1x1 per channel, yielding the two
        # complementary channel descriptors the paper combines.
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # Shared MLP: 1x1 conv pair is equivalent to FC and TorchScript-friendly.
        # C -> hidden -> C with ReLU in between; bias=False because the sigmoid
        # gate and downstream BN/activations make per-unit bias redundant.
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the (N, C, 1, 1) channel-attention gate for input `x`.

        Args:
            x: feature map of shape (N, C, H, W).

        Returns:
            Per-channel gate of shape (N, C, 1, 1) in (0, 1), ready to be
            broadcast-multiplied against `x`.
        """
        # Run BOTH pooled descriptors through the SAME mlp (weight sharing is
        # what keeps the param cost low and is core to the CBAM design).
        avg = self.mlp(self.avg_pool(x))
        mx = self.mlp(self.max_pool(x))
        # Merge the two evidence streams by summation, then sigmoid-squash to a
        # soft gate. (1,1) spatial dims broadcast across H,W when applied later.
        return self.act(avg + mx)


class SpatialAttention(nn.Module):
    """Attention map over spatial locations from channel-pooled features.

    Spatial attention answers "where in the image should the network look?" It
    pools the feature map along the channel axis (again both avg and max) to get
    two single-channel maps that summarise activity at each pixel, stacks them,
    and runs one convolution that learns to fuse them into a single-channel
    saliency map gated by a sigmoid. This complements channel attention: one
    reweights channels, the other reweights positions, and CBAM applies them in
    sequence.

    Args:
        kernel_size: receptive field of the fusing conv. Restricted to 3 or 7
            per the CBAM paper; 7 (the default) gives a wider context window for
            localising road-surface anomalies.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "CBAM paper uses 3 or 7"
        # 'same'-style padding keeps the saliency map at the input's H x W so it
        # can be multiplied back element-wise without any resizing.
        pad = kernel_size // 2
        # 2 -> 1 channel conv: fuses the [avg-pool, max-pool] stack into one
        # saliency channel. bias=False since the sigmoid handles the offset.
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the (N, 1, H, W) spatial-attention gate for input `x`.

        Args:
            x: feature map of shape (N, C, H, W) — typically already channel-
                attended when called from `CBAM`.

        Returns:
            Per-position gate of shape (N, 1, H, W) in (0, 1), broadcast across
            all channels when multiplied against `x`.
        """
        # Collapse the channel axis two ways to describe each spatial location:
        # mean response (overall) and max response (strongest single channel).
        avg = x.mean(dim=1, keepdim=True)            # (N, 1, H, W)
        mx = x.max(dim=1, keepdim=True).values       # (N, 1, H, W)
        # Stack the two descriptors on the channel dim -> (N, 2, H, W), let the
        # conv learn how to weigh them, then sigmoid into a soft spatial mask.
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
        # c1 is injected positionally by the Ultralytics parser (the previous
        # layer's output channel count). c2 is accepted only so the parser's
        # generic `args = [c1, c2, *rest]` rewriting works against CBAM exactly
        # as it does for Conv; CBAM itself preserves channels so c2 must equal c1.
        super().__init__()
        if c2 is not None and c2 != c1:
            # Parser may have scaled c2 differently — fall back to c1 with a warning.
            # In practice, when YAML c2 == c1 (intended use), the parser won't
            # re-scale. We tolerate divergence to be robust to scale tables.
            # (We never actually build with c2; both attention sub-modules are
            # sized from c1, guaranteeing the output channel count is preserved.)
            import warnings
            warnings.warn(
                f"CBAM expects c2 == c1; got c1={c1}, c2={c2}. Using c1.",
                RuntimeWarning,
            )
        # Two sequential attention stages, both dimensioned from the true
        # channel count c1 so the module is a drop-in, shape-preserving refiner.
        self.channel = ChannelAttention(c1, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine `x` with channel-then-spatial attention, preserving its shape.

        Applies the two gates in the order prescribed by the CBAM paper:
        first reweight channels, then reweight spatial positions of the already
        channel-attended map. Each multiply is a broadcast of a soft (0, 1) gate,
        so this only suppresses/emphasises existing features and never changes
        the (N, C, H, W) shape — which is what lets it slot into the neck.

        Args:
            x: input feature map of shape (N, C, H, W).

        Returns:
            Attention-refined feature map of the same shape (N, C, H, W).
        """
        # Stage 1: channel gate (N,C,1,1) broadcasts over H,W.
        x = x * self.channel(x)
        # Stage 2: spatial gate (N,1,H,W) broadcasts over C, applied to the
        # channel-refined features (sequential, not parallel — per the paper).
        x = x * self.spatial(x)
        return x
