"""Composite knowledge-distillation loss for YOLOv8n student ← YOLOv8l+ teacher.

Author: Rutuja Kulkarni

Role in the RADS Layer 3 pipeline
---------------------------------
This module implements the loss function behind the RADS knowledge-distillation
experiment. The deployed Road Anomaly Detection System needs the tiny, fast
YOLOv8n to run on edge hardware, but a small network trained alone leaves
accuracy on the table — especially for the harder WLPH/distant-PH cases. We
therefore train the YOLOv8n *student* to imitate a much larger, more accurate
YOLOv8l *teacher* in addition to learning from the ground-truth labels. This
file defines the composite objective that blends those two signals.

The math (and the WHY of each term)
-----------------------------------
Composition per Chapter 4.3.6:
    L_total = 0.4 * L_task(student)
            + 0.4 * KL( log_softmax(s_cls/T), softmax(t_cls/T) ) * T^2
            + 0.2 * MSE( proj(s_feat), t_feat )
    T = 4

  * L_task is the standard YOLOv8 detection loss on the student's predictions —
    it anchors the student to the real labels so distillation cannot drift away
    from the actual task.
  * The KL term is classic Hinton soft-target distillation. The teacher's class
    logits are softened by temperature T (T=4) which spreads probability mass
    onto the runner-up classes, exposing the teacher's "dark knowledge" (e.g.
    that a WLPH looks partly like a PH). The student is pushed to match this
    softened distribution. KL is multiplied by T^2 to restore the gradient
    magnitude, which softmax-by-T otherwise shrinks by ~1/T^2.
  * The feature MSE term distills *representations*, not just outputs: the
    student's deepest neck feature (P5 — the most information-dense, large-
    receptive-field layer) is regressed onto the teacher's P5. A 1x1 projection
    aligns channel counts (n vs l differ) before the MSE.

Integration with Ultralytics
-----------------------------
This file deliberately knows nothing about Ultralytics internals. It simply
exposes a `DistillationLoss` callable that the `DistillationTrainer` invokes
once it has run a forward pass through BOTH teacher and student and captured the
relevant tensors (class logits per head + P5 features) via forward hooks. That
separation keeps the student's training loop, optimizer, scheduler, augmentation
and W&B logging completely untouched.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureProjector(nn.Module):
    """1x1 conv to align student feature channels to teacher's.

    The feature-MSE distillation term compares the student and teacher P5
    feature maps elementwise, which requires matching channel dimensions.
    YOLOv8n (student) and YOLOv8l (teacher) have different channel widths, so
    this learnable 1x1 convolution projects the student's channels up to the
    teacher's count. When the counts already match it degenerates to an identity
    so no spurious parameters or compute are added.

    Created lazily on first call once we know teacher/student channel counts.
    """

    def __init__(self, c_student: int, c_teacher: int):
        """Build the channel-alignment projection.

        Args:
            c_student: Channel count of the student's P5 feature.
            c_teacher: Channel count of the teacher's P5 feature (the target).

        Side effects:
            Registers either an `nn.Identity` (equal channels) or a bias-free
            1x1 `nn.Conv2d` as the submodule `self.proj`. The conv is learnable
            and trains alongside the student.
        """
        super().__init__()
        if c_student == c_teacher:
            # No projection needed; identity keeps the path parameter-free.
            self.proj: nn.Module = nn.Identity()
        else:
            # Bias-free 1x1 conv = a per-pixel learned linear remap of channels.
            self.proj = nn.Conv2d(c_student, c_teacher, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``x`` (student feature) into the teacher's channel space.

        Args:
            x: Student feature map, shape ``(B, c_student, H, W)``.

        Returns:
            Tensor of shape ``(B, c_teacher, H, W)`` ready for MSE against the
            teacher feature.
        """
        return self.proj(x)


class DistillationLoss(nn.Module):
    """Composite distillation loss.

    Blends three signals into a single scalar the trainer can backprop through:
    the student's own task loss, a temperature-scaled KL between teacher and
    student class distributions, and an MSE between their P5 features. See the
    module docstring for the rationale behind each term and weight.

    Args:
        task_w: weight on student task loss. Default 0.4.
        kl_w:   weight on classification KL term. Default 0.4.
        feat_w: weight on feature MSE term. Default 0.2.
        T:      softmax temperature for KL. Default 4.
    """

    def __init__(
        self,
        task_w: float = 0.4,
        kl_w: float = 0.4,
        feat_w: float = 0.2,
        T: float = 4.0,
    ):
        """Store the loss-component weights and temperature.

        Args:
            task_w / kl_w / feat_w: Mixing weights (default 0.4 / 0.4 / 0.2,
                summing to 1.0 as in Chapter 4.3.6).
            T: Distillation temperature for softening the class logits.

        Side effects:
            Initialises `self.projector` to ``None``; it is built lazily on the
            first `forward` once feature channel counts are known.
        """
        super().__init__()
        self.task_w = task_w
        self.kl_w = kl_w
        self.feat_w = feat_w
        self.T = T
        self.projector: nn.Module | None = None  # set on first forward

    def _ensure_projector(
        self, s_feat: torch.Tensor, t_feat: torch.Tensor, device: torch.device
    ) -> None:
        """Lazily construct the channel-alignment projector (idempotent).

        We cannot build the `FeatureProjector` in `__init__` because the
        teacher/student channel counts are only known once real feature tensors
        flow through on the first training step. This creates it on demand from
        the observed channel dimensions and parks it on the right device.

        Args:
            s_feat: A student P5 feature, used for its channel count (dim 1).
            t_feat: The teacher P5 feature, used for its channel count (dim 1).
            device: Device on which to place the newly created projector.

        Side effects:
            Sets `self.projector` on first call; subsequent calls return early.
        """
        if self.projector is not None:
            return
        self.projector = FeatureProjector(
            c_student=s_feat.shape[1], c_teacher=t_feat.shape[1]
        ).to(device)

    def forward(
        self,
        task_loss: torch.Tensor,
        student_cls_logits: Sequence[torch.Tensor],
        teacher_cls_logits: Sequence[torch.Tensor],
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Combine task, KL and feature losses into the total distillation loss.

        Args:
            task_loss: The student's standard YOLOv8 detection loss (scalar,
                with grad) computed by the trainer for this batch.
            student_cls_logits: Per-head classification logit maps from the
                student's Detect head, each shaped ``(B, C, H, W)``.
            teacher_cls_logits: The teacher's matching per-head logit maps. Must
                be index-aligned with `student_cls_logits` (same scales).
            student_feat: Student P5 feature map ``(B, c_s, H_s, W_s)``.
            teacher_feat: Teacher P5 feature map ``(B, c_t, H_t, W_t)``.

        Returns:
            ``(total, components)`` where ``total`` is the weighted-sum scalar to
            backprop, and ``components`` is a dict of detached scalars (task / kl
            / feat / total) for logging.

        Side effects:
            May lazily build `self.projector` on the first call.
        """
        # --- KL classification term, averaged across detection heads ---
        # Done per detection scale/head, then averaged, so every scale's class
        # distribution is distilled equally.
        kl_terms: List[torch.Tensor] = []
        for s_cls, t_cls in zip(student_cls_logits, teacher_cls_logits):
            # Flatten (B, C, H, W) -> (B*H*W, C). Soft targets from teacher.
            # Each spatial location becomes an independent C-way distribution.
            s_flat = s_cls.permute(0, 2, 3, 1).reshape(-1, s_cls.shape[1])
            t_flat = t_cls.permute(0, 2, 3, 1).reshape(-1, t_cls.shape[1])
            # Temperature scaling: dividing logits by T (>1) flattens the
            # softmax, surfacing the teacher's relative confidence across all
            # classes ("dark knowledge"). Student uses log_softmax (KL expects
            # log-probabilities for its first argument); teacher uses softmax.
            log_p_s = F.log_softmax(s_flat / self.T, dim=-1)
            p_t = F.softmax(t_flat / self.T, dim=-1)
            # batchmean gives the proper KL averaged over the batch dimension.
            # The * T^2 factor compensates for the 1/T^2 gradient shrinkage that
            # temperature scaling introduces, keeping this term's magnitude
            # comparable to the unscaled case (standard Hinton distillation).
            kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (self.T ** 2)
            kl_terms.append(kl)
        # Mean across heads; fall back to a zero scalar (on the right device/
        # dtype) if no logit pairs were supplied so the sum below stays valid.
        kl_loss = torch.stack(kl_terms).mean() if kl_terms else task_loss.new_zeros(())

        # --- Feature MSE term (P5 by convention) ---
        # Ensure the channel-aligning projector exists, then map the student
        # feature into the teacher's channel space.
        self._ensure_projector(student_feat, teacher_feat, task_loss.device)
        s_proj = self.projector(student_feat)
        # Match spatial dims if student is at lower resolution. Bilinear resize
        # makes the two feature maps the same H x W so MSE is well-defined.
        if s_proj.shape[-2:] != teacher_feat.shape[-2:]:
            s_proj = F.interpolate(
                s_proj, size=teacher_feat.shape[-2:], mode="bilinear", align_corners=False
            )
        # Regress student feature onto the teacher's. `.detach()` freezes the
        # teacher as a fixed target so no gradient flows back into the (frozen)
        # teacher and only the student + projector learn.
        feat_loss = F.mse_loss(s_proj, teacher_feat.detach())

        # Weighted sum (0.4 / 0.4 / 0.2 by default) — the single scalar the
        # trainer backpropagates.
        total = (
            self.task_w * task_loss
            + self.kl_w * kl_loss
            + self.feat_w * feat_loss
        )
        # Detached components are returned purely for monitoring/W&B; detaching
        # avoids holding the graph and double-counting gradients.
        return total, {
            "loss_task": task_loss.detach(),
            "loss_kl": kl_loss.detach(),
            "loss_feat": feat_loss.detach(),
            "loss_total": total.detach(),
        }
