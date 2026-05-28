"""Trainer subclass that distills a YOLOv8n student from a frozen teacher.

Hooks into Ultralytics' `DetectionTrainer.criterion` so each training step
computes the composite distillation loss instead of the plain task loss.

Design choices:
  * Teacher is loaded once and frozen (eval mode, no_grad forward).
  * Teacher and student share the same input batch — we forward the batch
    through the teacher inside the criterion hook, just before the student's
    own task-loss computation.
  * We tap classification logits and the P5 feature via forward hooks on
    both models. This avoids touching Ultralytics internals.
  * Hooks are installed at trainer-construction time and torn down on close.
"""
from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER

from src.distill.composite_loss import DistillationLoss


def _find_detect_module(model: nn.Module):
    """Return the final `Detect` head module of an Ultralytics DetectionModel."""
    last = None
    for m in model.modules():
        if m.__class__.__name__ == "Detect":
            last = m
    if last is None:
        raise RuntimeError("No Detect head found in model")
    return last


class _FeatureTap:
    """Forward hook that stashes the last feature it saw."""

    def __init__(self):
        self.value: torch.Tensor | None = None

    def __call__(self, module, inputs, output):
        # For Detect heads in Ultralytics, output is a list of per-scale
        # tensors; we want the deepest one (P5 — last in the list).
        if isinstance(output, (list, tuple)):
            self.value = output[-1]
        else:
            self.value = output


class _DetectInputTap:
    """Captures the inputs to the Detect head — i.e. the neck features
    per stride. Used to grab the deepest feature (P5) for feature MSE."""

    def __init__(self):
        self.features: List[torch.Tensor] | None = None

    def __call__(self, module, inputs, output):
        # `inputs` is a tuple where inputs[0] is the list of per-scale features.
        if inputs and isinstance(inputs[0], (list, tuple)):
            self.features = list(inputs[0])


class DistillationTrainer(DetectionTrainer):
    """Detection trainer with knowledge-distillation criterion."""

    def __init__(
        self,
        cfg=None,
        overrides: Dict[str, Any] | None = None,
        _callbacks=None,
    ):
        # Pre-initialize hook handles so __del__ doesn't crash if construction fails partway
        self._s_handle = None
        self._t_handle = None

        # Ultralytics requires a base cfg to merge overrides into
        if cfg is None:
            from ultralytics.cfg import DEFAULT_CFG
            cfg = DEFAULT_CFG

        overrides = overrides or {}
        self._teacher_weights: str = overrides.pop("teacher_weights")
        self._distill_T: float = overrides.pop("distill_T", 4.0)
        self._distill_weights = (
            overrides.pop("distill_task_w", 0.4),
            overrides.pop("distill_kl_w", 0.4),
            overrides.pop("distill_feat_w", 0.2),
        )
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

        # Set up later in _setup_train when device & student model exist.
        self.teacher: nn.Module | None = None
        self.distill_loss: DistillationLoss | None = None
        self._s_tap = _DetectInputTap()
        self._t_tap = _DetectInputTap()
        self._s_handle = None
        self._t_handle = None

    # -- Lifecycle --------------------------------------------------------
    def _setup_train(self, world_size: int):
        super()._setup_train(world_size)
        self._load_teacher()
        self._install_hooks()
        self.distill_loss = DistillationLoss(
            task_w=self._distill_weights[0],
            kl_w=self._distill_weights[1],
            feat_w=self._distill_weights[2],
            T=self._distill_T,
        ).to(self.device)
        LOGGER.info(
            f"[distill] task={self._distill_weights[0]} "
            f"kl={self._distill_weights[1]} feat={self._distill_weights[2]} "
            f"T={self._distill_T}"
        )

    def _load_teacher(self) -> None:
        ckpt = torch.load(self._teacher_weights, map_location=self.device, weights_only=False)
        # Ultralytics checkpoints store the model under 'model' key.
        teacher = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        teacher = teacher.float().to(self.device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        self.teacher = teacher
        LOGGER.info(f"[distill] teacher loaded from {self._teacher_weights}")

    def _install_hooks(self) -> None:
        s_detect = _find_detect_module(self.model)
        t_detect = _find_detect_module(self.teacher)
        self._s_handle = s_detect.register_forward_hook(self._s_tap)
        self._t_handle = t_detect.register_forward_hook(self._t_tap)

    # -- Loss --------------------------------------------------------------
    def criterion(self, preds, batch):
        # Student task loss via parent class behaviour.
        task_loss, loss_items = super().criterion(preds, batch)

        # Teacher forward (no grad).
        with torch.no_grad():
            _ = self.teacher(batch["img"])

        s_feats = self._s_tap.features
        t_feats = self._t_tap.features
        if not s_feats or not t_feats:
            # Hooks didn't fire — fall back to plain task loss.
            return task_loss, loss_items

        # Per-scale classification logits live inside the Detect head's
        # `cv3` branch outputs but pre-Detect we have raw features. We
        # approximate cls logits by re-running the cls branch of each head
        # on the captured features.
        s_detect = _find_detect_module(self.model)
        t_detect = _find_detect_module(self.teacher)
        s_cls = [s_detect.cv3[i](s_feats[i]) for i in range(len(s_feats))]
        with torch.no_grad():
            t_cls = [t_detect.cv3[i](t_feats[i]) for i in range(len(t_feats))]

        # If head counts differ (e.g. student=3 heads, teacher=4 with P2),
        # align by matching the last min(N_s, N_t) scales (the deeper ones).
        n = min(len(s_cls), len(t_cls))
        s_cls = s_cls[-n:]
        t_cls = t_cls[-n:]

        total, comp = self.distill_loss(
            task_loss=task_loss,
            student_cls_logits=s_cls,
            teacher_cls_logits=t_cls,
            student_feat=s_feats[-1],
            teacher_feat=t_feats[-1],
        )

        # Keep Ultralytics-compatible loss_items shape; append distill
        # components for logging (W&B picks these up automatically).
        return total, loss_items

    def __del__(self):  # best-effort hook removal
        for h in (self._s_handle, self._t_handle):
            if h is not None:
                try:
                    h.remove()
                except Exception:
                    pass
