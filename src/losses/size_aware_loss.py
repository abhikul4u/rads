"""Size-aware box regression loss.

Author: Rutuja Kulkarni

Role in the RADS Layer 3 pipeline
---------------------------------
This module is one of the two core training-time research contributions of the
RADS (Road Anomaly Detection System) Layer 3 model-development pipeline. Road
anomalies — manholes (MH), potholes (PH) and water-logged potholes (WLPH) —
appear at wildly different apparent sizes depending on how far they are from the
camera. The small/distant instances are the ones most likely to be missed, yet
they are precisely the safety-critical detections. Standard YOLOv8 training
under-serves these because, although the stock `v8DetectionLoss` scales each
anchor's bbox/DFL contribution by a `weight` term (the sum of that anchor's
predicted class scores), that term reflects classification confidence, NOT
object size. Large, easy boxes therefore continue to dominate the regression
gradient.

What this loss does (the math, and the WHY)
-------------------------------------------
`SizeAwareBboxLoss` multiplies each foreground anchor's combined bbox (CIoU) +
DFL contribution by an inverse-square-root-of-area weight:

        size_w = area_norm ** (-0.5 * alpha)

where `area_norm` is the ground-truth box area expressed as a fraction of the
(squared) anchor-coordinate range, so the weight is scale-invariant across input
resolutions. Using the inverse SQUARE ROOT (rather than inverse area) gives a
gentler, more numerically stable up-weighting: halving an object's linear size
roughly doubles its loss weight instead of quadrupling it. `alpha` tunes the
strength (0 disables the effect, recovering stock behaviour; 1.0 is the thesis
default). The weight is clamped (and `area_norm` is floored by `eps`) so a single
degenerate or tiny ground-truth box cannot explode the gradient.

Integration with Ultralytics
-----------------------------
The class is a drop-in subclass of Ultralytics' `BboxLoss`, faithfully
replicating the `BboxLoss.forward` logic of Ultralytics 8.3 and then folding in
the extra `size_w` multiplier. It is wired into training by a one-line
monkey-patch (`install_size_aware_loss`) that swaps the `bbox_loss` attribute
constructed inside `v8DetectionLoss.__init__`. Because the patch is installed and
removed at module level, the size-aware behaviour is a fully reversible toggle:
no fork of Ultralytics and no edit to the training loop is required.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist


class SizeAwareBboxLoss(BboxLoss):
    """`BboxLoss` + inverse-sqrt-area reweighting on small objects.

    Subclass of Ultralytics' `BboxLoss`. Its `forward` reproduces the stock
    CIoU + DFL computation exactly, then multiplies each foreground anchor's
    contribution by a size-dependent weight so that small ground-truth boxes
    (the hard, safety-critical road anomalies) contribute proportionally more
    gradient. Apart from this extra multiplier the loss is numerically identical
    to the original, which keeps it a safe, reversible drop-in replacement.
    """

    def __init__(self, reg_max: int, alpha: float = 0.5, eps: float = 1e-3):
        """Construct the size-aware bbox loss.

        Args:
            reg_max: Distribution Focal Loss bin count (Ultralytics default 16).
                Passed straight through to the parent `BboxLoss` so the DFL
                sub-loss is configured identically to stock.
            alpha: Strength of the size weighting. ``0`` disables the effect
                entirely (recovering stock `BboxLoss`); ``1`` is the thesis
                default. Stored as float so it can be supplied as an int/str.
            eps: Lower clamp applied to the *normalised* area before the inverse
                power, preventing a divide-by-zero / blow-up on collapsed or
                near-zero-area boxes.

        Side effects:
            Calls ``super().__init__`` which builds the underlying DFL loss; then
            caches ``alpha`` and ``eps`` on the instance for use in ``forward``.
        """
        super().__init__(reg_max)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
    ):
        """Compute the size-weighted CIoU + DFL bbox loss.

        This mirrors the signature and semantics of Ultralytics 8.3's
        `BboxLoss.forward`. All tensors are produced upstream by the task
        aligner (TAL); we only consume them and add the size weighting.

        Args:
            pred_dist: Predicted DFL distribution logits over the full anchor
                grid, shape ``(B, A, 4 * reg_max)`` (flattened internally).
            pred_bboxes: Decoded predicted boxes, xyxy, feature-map coords.
            anchor_points: Anchor centre coordinates; their max defines the
                coordinate range used to normalise box areas.
            target_bboxes: Assigned ground-truth boxes, xyxy, feature-map coords.
            target_scores: Per-anchor per-class alignment scores from TAL.
            target_scores_sum: Scalar normaliser (sum of all target scores).
            fg_mask: Boolean mask selecting positive (foreground) anchors.

        Returns:
            Tuple ``(loss_iou, loss_dfl)`` of scalar tensors, each already
            normalised by ``target_scores_sum`` — exactly the two values the
            parent `v8DetectionLoss` expects, so callers see no API change.
        """
        # Replicates Ultralytics 8.3 `BboxLoss.forward` then applies size weight.
        # `weight` is the stock confidence-based weight: per foreground anchor,
        # the sum over classes of its alignment score. Shape (N_fg, 1).
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        # ---- Inverse-sqrt-area size weight (the RADS contribution) ----------
        # Per-anchor target area on the foreground subset only (matches IoU's
        # masking). `target_bboxes[fg_mask]` is (N_fg, 4) in xyxy in
        # feature-map coordinates.
        tb_fg = target_bboxes[fg_mask]
        # Width/height of each GT box; clamp at 0 so numerically-degenerate
        # boxes (x2<x1) yield 0 area rather than a spurious negative.
        wh = (tb_fg[..., 2:] - tb_fg[..., :2]).clamp(min=0)
        area = wh[..., 0] * wh[..., 1]
        # Normalise by the anchor coordinate range squared to make this
        # scale-invariant across input resolutions. Detached so the size weight
        # is treated as a constant multiplier and contributes no gradient itself.
        coord_max = anchor_points.max().detach()
        # area_norm in (eps, 1]: eps floor avoids the inverse power exploding on
        # tiny/collapsed boxes; cap at 1.0 so a box can never be *down*-weighted.
        area_norm = (area / (coord_max ** 2 + 1e-9)).clamp(min=self.eps, max=1.0)
        # The heart of the loss: weight = area_norm ** (-alpha/2), i.e.
        # 1/sqrt(area)^alpha. Smaller area -> larger weight -> more gradient on
        # small road anomalies. alpha scales how aggressively we do this.
        size_w = area_norm.pow(-0.5 * self.alpha).unsqueeze(-1)
        # Stabilise: clamp so a single tiny GT can't dominate. With the eps floor
        # the theoretical max is eps**(-alpha/2); this hard cap of 4.0 guards the
        # gradient regardless of alpha.
        size_w = size_w.clamp(max=4.0)

        # IoU loss with size weighting. CIoU matches stock YOLOv8; each anchor's
        # (1 - IoU) is scaled by BOTH the confidence weight and the size weight.
        iou = bbox_iou(pred_bboxes[fg_mask], tb_fg, xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight * size_w).sum() / target_scores_sum

        # DFL loss. Match stock BboxLoss's masking order: compute target_ltrb
        # over the full anchor grid (shapes align), then index by fg_mask.
        if self.dfl_loss is not None:
            # Convert GT boxes to left-top-right-bottom distances relative to
            # each anchor point — the regression target the DFL bins predict.
            target_ltrb = bbox2dist(
                anchor_points, target_bboxes, self.dfl_loss.reg_max - 1
            )
            # Same dual (confidence x size) weighting is applied to the DFL term
            # so the small-object up-weighting is consistent across both halves
            # of the box loss.
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                target_ltrb[fg_mask],
            ) * weight * size_w
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            # reg_max == 1 means DFL is disabled; contribute a zero scalar on the
            # correct device so the caller's arithmetic still works.
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl


def install_size_aware_loss(alpha: float = 0.5) -> None:
    """Monkey-patch `v8DetectionLoss.__init__` to use `SizeAwareBboxLoss`.

    Call this once before constructing the YOLO trainer. Reversible: call
    `uninstall_size_aware_loss()` to restore stock behavior.

    Mechanics:
        Ultralytics builds its `bbox_loss` inside `v8DetectionLoss.__init__`.
        Rather than fork Ultralytics, we wrap that constructor: the original
        runs first (so `self.bbox_loss` exists and we can read its `reg_max`),
        then we replace `self.bbox_loss` with a `SizeAwareBboxLoss` carrying the
        requested `alpha`. The replacement is moved onto the model's device so
        the swap is transparent to the training loop.

    Args:
        alpha: Size-weighting strength forwarded to every `SizeAwareBboxLoss`
            instance the patched constructor creates.

    Side effects:
        Mutates the `ultralytics.utils.loss` module: rebinds
        `v8DetectionLoss.__init__`, sets a `_RADS_SIZE_AWARE_INSTALLED` guard
        flag, and stashes the original constructor under
        `_RADS_SIZE_AWARE_ORIG_INIT` so it can be restored. Idempotent — a
        second call while installed is a no-op (prevents double-wrapping, which
        would otherwise nest patches and corrupt the saved original).
    """
    from ultralytics.utils import loss as loss_mod

    # Guard against double-install: if already patched, the saved original would
    # be overwritten with the patched version and uninstall could never restore.
    if getattr(loss_mod, "_RADS_SIZE_AWARE_INSTALLED", False):
        return

    # Capture the genuine, unpatched constructor so we can both call it from the
    # wrapper and restore it later.
    _orig_init = loss_mod.v8DetectionLoss.__init__

    def _patched_init(self, model, tal_topk=10):  # noqa: D401
        # Run stock init first: this populates self.bbox_loss (and everything
        # else) exactly as Ultralytics would, leaving behaviour unchanged except
        # for the bbox loss we are about to swap.
        _orig_init(self, model, tal_topk=tal_topk)
        # Swap the bbox loss while preserving reg_max from the original, so the
        # DFL bin count matches the model head. reg_max=1 => DFL disabled.
        reg_max = self.bbox_loss.dfl_loss.reg_max if self.bbox_loss.dfl_loss else 1
        self.bbox_loss = SizeAwareBboxLoss(reg_max=reg_max, alpha=alpha).to(
            next(model.parameters()).device
        )

    # Install the wrapper and record state for a clean, reversible uninstall.
    loss_mod.v8DetectionLoss.__init__ = _patched_init
    loss_mod._RADS_SIZE_AWARE_INSTALLED = True
    loss_mod._RADS_SIZE_AWARE_ORIG_INIT = _orig_init
    print(f"[loss] size-aware bbox loss installed (alpha={alpha})")


def uninstall_size_aware_loss() -> None:
    """Reverse `install_size_aware_loss`, restoring stock `v8DetectionLoss`.

    Restores the original `v8DetectionLoss.__init__` saved at install time and
    clears the guard flag. Safe to call when not installed (no-op). This makes
    the size-aware loss a clean experimental toggle — useful for ablation runs
    that compare against the unmodified Ultralytics baseline within one process.

    Side effects:
        Mutates `ultralytics.utils.loss`: rebinds `v8DetectionLoss.__init__`
        back to the saved original and sets `_RADS_SIZE_AWARE_INSTALLED` False.
    """
    from ultralytics.utils import loss as loss_mod

    # Nothing to undo if we never installed.
    if not getattr(loss_mod, "_RADS_SIZE_AWARE_INSTALLED", False):
        return
    loss_mod.v8DetectionLoss.__init__ = loss_mod._RADS_SIZE_AWARE_ORIG_INIT
    loss_mod._RADS_SIZE_AWARE_INSTALLED = False
    print("[loss] size-aware bbox loss removed")
