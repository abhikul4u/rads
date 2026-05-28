"""Composite knowledge-distillation loss for YOLOv8n student ← YOLOv8l+ teacher.

Composition per Chapter 4.3.6:
    L_total = 0.4 * L_task(student)
            + 0.4 * KL( log_softmax(s_cls/T), softmax(t_cls/T) ) * T^2
            + 0.2 * MSE( proj(s_feat), t_feat )
    T = 4

L_task is the standard YOLOv8 detection loss on the student's predictions.
The KL term operates on classification logits from matched detection heads.
The feature MSE operates on the deepest neck feature (P5) — the most
information-dense layer.

We expose a `DistillationLoss` callable that the trainer invokes after a
forward pass on BOTH teacher and student.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureProjector(nn.Module):
    """1x1 conv to align student feature channels to teacher's.

    Created lazily on first call once we know teacher/student channel counts.
    """

    def __init__(self, c_student: int, c_teacher: int):
        super().__init__()
        if c_student == c_teacher:
            self.proj: nn.Module = nn.Identity()
        else:
            self.proj = nn.Conv2d(c_student, c_teacher, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DistillationLoss(nn.Module):
    """Composite distillation loss.

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
        super().__init__()
        self.task_w = task_w
        self.kl_w = kl_w
        self.feat_w = feat_w
        self.T = T
        self.projector: nn.Module | None = None  # set on first forward

    def _ensure_projector(
        self, s_feat: torch.Tensor, t_feat: torch.Tensor, device: torch.device
    ) -> None:
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
        # --- KL classification term, averaged across detection heads ---
        kl_terms: List[torch.Tensor] = []
        for s_cls, t_cls in zip(student_cls_logits, teacher_cls_logits):
            # Flatten (B, C, H, W) -> (B*H*W, C). Soft targets from teacher.
            s_flat = s_cls.permute(0, 2, 3, 1).reshape(-1, s_cls.shape[1])
            t_flat = t_cls.permute(0, 2, 3, 1).reshape(-1, t_cls.shape[1])
            log_p_s = F.log_softmax(s_flat / self.T, dim=-1)
            p_t = F.softmax(t_flat / self.T, dim=-1)
            kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (self.T ** 2)
            kl_terms.append(kl)
        kl_loss = torch.stack(kl_terms).mean() if kl_terms else task_loss.new_zeros(())

        # --- Feature MSE term (P5 by convention) ---
        self._ensure_projector(student_feat, teacher_feat, task_loss.device)
        s_proj = self.projector(student_feat)
        # Match spatial dims if student is at lower resolution.
        if s_proj.shape[-2:] != teacher_feat.shape[-2:]:
            s_proj = F.interpolate(
                s_proj, size=teacher_feat.shape[-2:], mode="bilinear", align_corners=False
            )
        feat_loss = F.mse_loss(s_proj, teacher_feat.detach())

        total = (
            self.task_w * task_loss
            + self.kl_w * kl_loss
            + self.feat_w * feat_loss
        )
        return total, {
            "loss_task": task_loss.detach(),
            "loss_kl": kl_loss.detach(),
            "loss_feat": feat_loss.detach(),
            "loss_total": total.detach(),
        }
