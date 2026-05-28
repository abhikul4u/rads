"""Size-aware box regression loss.

Standard YOLOv8 `v8DetectionLoss` already weights bbox loss by target area via
the `weight` term (= sum of class scores) — but it does not explicitly upweight
*small* objects, which is exactly what hurts mAP on small classes (MH, distant
PH/WLPH).

This loss multiplies the per-anchor bbox+DFL contribution by `1/sqrt(area_norm)`,
clamped to avoid blow-up on degenerate boxes. Box areas are normalised to image
fraction so the multiplier is scale-invariant.

Drop-in replacement for `BboxLoss` inside `v8DetectionLoss`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist


class SizeAwareBboxLoss(BboxLoss):
    """`BboxLoss` + inverse-sqrt-area reweighting on small objects."""

    def __init__(self, reg_max: int, alpha: float = 0.5, eps: float = 1e-3):
        """
        Args:
            reg_max: distribution focal loss bin count (Ultralytics default 16).
            alpha: strength of size weighting. 0 disables; 1 = paper default.
            eps: lower clamp on normalised area to avoid /0 on collapsed boxes.
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
        # Replicates Ultralytics 8.3 `BboxLoss.forward` then applies size weight.
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        # Per-anchor target area on the foreground subset only (matches IoU's
        # masking). `target_bboxes[fg_mask]` is (N_fg, 4) in xyxy in
        # feature-map coordinates.
        tb_fg = target_bboxes[fg_mask]
        wh = (tb_fg[..., 2:] - tb_fg[..., :2]).clamp(min=0)
        area = wh[..., 0] * wh[..., 1]
        # Normalise by the anchor coordinate range squared to make this
        # scale-invariant across input resolutions.
        coord_max = anchor_points.max().detach()
        area_norm = (area / (coord_max ** 2 + 1e-9)).clamp(min=self.eps, max=1.0)
        size_w = area_norm.pow(-0.5 * self.alpha).unsqueeze(-1)
        # Stabilise: clamp so a single tiny GT can't dominate.
        size_w = size_w.clamp(max=4.0)

        # IoU loss with size weighting.
        iou = bbox_iou(pred_bboxes[fg_mask], tb_fg, xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight * size_w).sum() / target_scores_sum

        # DFL loss. Match stock BboxLoss's masking order: compute target_ltrb
        # over the full anchor grid (shapes align), then index by fg_mask.
        if self.dfl_loss is not None:
            target_ltrb = bbox2dist(
                anchor_points, target_bboxes, self.dfl_loss.reg_max - 1
            )
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                target_ltrb[fg_mask],
            ) * weight * size_w
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl


def install_size_aware_loss(alpha: float = 0.5) -> None:
    """Monkey-patch `v8DetectionLoss.__init__` to use `SizeAwareBboxLoss`.

    Call this once before constructing the YOLO trainer. Reversible: call
    `uninstall_size_aware_loss()` to restore stock behavior.
    """
    from ultralytics.utils import loss as loss_mod

    if getattr(loss_mod, "_RADS_SIZE_AWARE_INSTALLED", False):
        return

    _orig_init = loss_mod.v8DetectionLoss.__init__

    def _patched_init(self, model, tal_topk=10):  # noqa: D401
        _orig_init(self, model, tal_topk=tal_topk)
        # Swap the bbox loss while preserving reg_max from the original.
        reg_max = self.bbox_loss.dfl_loss.reg_max if self.bbox_loss.dfl_loss else 1
        self.bbox_loss = SizeAwareBboxLoss(reg_max=reg_max, alpha=alpha).to(
            next(model.parameters()).device
        )

    loss_mod.v8DetectionLoss.__init__ = _patched_init
    loss_mod._RADS_SIZE_AWARE_INSTALLED = True
    loss_mod._RADS_SIZE_AWARE_ORIG_INIT = _orig_init
    print(f"[loss] size-aware bbox loss installed (alpha={alpha})")


def uninstall_size_aware_loss() -> None:
    from ultralytics.utils import loss as loss_mod

    if not getattr(loss_mod, "_RADS_SIZE_AWARE_INSTALLED", False):
        return
    loss_mod.v8DetectionLoss.__init__ = loss_mod._RADS_SIZE_AWARE_ORIG_INIT
    loss_mod._RADS_SIZE_AWARE_INSTALLED = False
    print("[loss] size-aware bbox loss removed")
