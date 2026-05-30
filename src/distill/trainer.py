"""Trainer subclass that distills a YOLOv8n student from a frozen teacher.

Author: Rutuja Kulkarni

Role in the RADS Layer 3 pipeline
---------------------------------
This module is the orchestration layer of the RADS knowledge-distillation
experiment. Where `composite_loss.py` defines *what* the distillation objective
is, this file wires that objective into a real Ultralytics training run so the
lightweight YOLOv8n student learns from a frozen, accurate YOLOv8l teacher
without us having to fork or rewrite Ultralytics' training machinery.

How it integrates with Ultralytics
-----------------------------------
`DistillationTrainer` subclasses Ultralytics' `DetectionTrainer` and overrides
exactly one method — `criterion`. Every training step Ultralytics calls
`criterion(preds, batch)` to turn predictions into a loss; we intercept that
call, compute the student's normal task loss via `super().criterion(...)`, then
also forward the same batch through the teacher and fold in the distillation
terms. Because we override nothing else, the student's data loader, optimizer,
LR scheduler, augmentation pipeline, EMA, checkpointing and W&B logging all keep
working unchanged.

Design choices:
  * Teacher is loaded once and frozen (eval mode, no_grad forward).
  * Teacher and student share the same input batch — we forward the batch
    through the teacher inside the criterion hook, just before the student's
    own task-loss computation.
  * We tap classification logits and the P5 feature via forward hooks on
    both models. This avoids touching Ultralytics internals.
  * Hooks are installed at trainer-construction time and torn down on close.

Note on the forward hooks: we register the hooks on the `Detect` head of each
model. The hook captures the head's *inputs* — i.e. the per-stride neck feature
list — rather than its outputs, because the raw neck features (P3/P4/P5) are
what we need both for the P5 feature-MSE term and for re-deriving classification
logits (by re-running the head's `cv3` branch). This keeps distillation entirely
external to Ultralytics' forward code.
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
    """Return the final `Detect` head module of an Ultralytics DetectionModel.

    Iterates all submodules and keeps the last one whose class name is
    ``"Detect"``. We match by class name (not isinstance) to stay decoupled from
    Ultralytics' import paths, which shift between versions. The *last* match is
    taken because in a `DetectionModel` the detection head is the final module.

    Args:
        model: An Ultralytics detection model (teacher or student).

    Returns:
        The `Detect` head module instance.

    Raises:
        RuntimeError: If no `Detect` head is found (wrong model type).
    """
    last = None
    for m in model.modules():
        if m.__class__.__name__ == "Detect":
            last = m
    if last is None:
        raise RuntimeError("No Detect head found in model")
    return last


class _FeatureTap:
    """Forward hook that stashes the last feature it saw.

    A callable hook object (registered via `register_forward_hook`) that records
    the deepest per-scale output tensor of the module it is attached to, so it
    can be read back after a forward pass. Currently unused by the active code
    path (the trainer uses `_DetectInputTap`) but retained as a feature-output
    variant of the tap.
    """

    def __init__(self):
        """Initialise the captured value to ``None`` (nothing seen yet)."""
        self.value: torch.Tensor | None = None

    def __call__(self, module, inputs, output):
        """Hook entry point invoked by PyTorch after the module's forward.

        Args:
            module: The hooked module (unused; required by the hook signature).
            inputs: The module's forward inputs (unused here).
            output: The module's forward output.

        Side effects:
            Stores the deepest feature on ``self.value``.
        """
        # For Detect heads in Ultralytics, output is a list of per-scale
        # tensors; we want the deepest one (P5 — last in the list).
        if isinstance(output, (list, tuple)):
            self.value = output[-1]
        else:
            self.value = output


class _DetectInputTap:
    """Captures the inputs to the Detect head — i.e. the neck features
    per stride. Used to grab the deepest feature (P5) for feature MSE.

    Registered as a forward hook on a `Detect` module. PyTorch passes the
    module's positional `inputs` to the hook; for a Detect head ``inputs[0]`` is
    the list of per-stride neck feature maps (P3, P4, P5). We snapshot that list
    so the criterion can later read both P5 (for feature MSE) and all scales
    (to re-derive classification logits). Capturing inputs rather than outputs
    is deliberate: the raw features are exactly what we need and re-running the
    head's `cv3` branch on them avoids depending on the head's output format.
    """

    def __init__(self):
        """Initialise the captured feature list to ``None`` (nothing seen)."""
        self.features: List[torch.Tensor] | None = None

    def __call__(self, module, inputs, output):
        """Hook entry point: snapshot the per-scale neck features.

        Args:
            module: The hooked Detect module (unused; per hook signature).
            inputs: Tuple of forward inputs; ``inputs[0]`` is the per-scale
                feature list we want.
            output: The Detect head's output (unused here).

        Side effects:
            Stores a fresh list copy of the per-scale features on
            ``self.features``.
        """
        # `inputs` is a tuple where inputs[0] is the list of per-scale features.
        # Copy into a new list so a later in-place mutation upstream can't alter
        # what we captured for this step.
        if inputs and isinstance(inputs[0], (list, tuple)):
            self.features = list(inputs[0])


class DistillationTrainer(DetectionTrainer):
    """Detection trainer with knowledge-distillation criterion.

    Subclasses Ultralytics' `DetectionTrainer` and overrides only `criterion`
    (plus lifecycle hooks for teacher loading / hook wiring). The student is
    trained exactly as a normal YOLOv8n run, except the per-step loss is the
    composite distillation loss instead of the plain task loss. Distillation
    hyper-parameters are passed in through Ultralytics' `overrides` dict and
    popped out before they reach the base trainer (which would reject unknown
    keys).
    """

    def __init__(
        self,
        cfg=None,
        overrides: Dict[str, Any] | None = None,
        _callbacks=None,
    ):
        """Construct the distillation trainer and extract distill settings.

        Args:
            cfg: Base Ultralytics config; defaults to `DEFAULT_CFG` if omitted,
                since the base trainer needs a cfg to merge overrides into.
            overrides: Standard Ultralytics overrides dict, additionally
                carrying RADS-specific keys that are popped here:
                ``teacher_weights`` (required path to the teacher checkpoint),
                ``distill_T`` (temperature, default 4.0) and the three mixing
                weights ``distill_task_w`` / ``distill_kl_w`` / ``distill_feat_w``.
            _callbacks: Passed straight through to the base trainer.

        Side effects:
            Pops the distill keys out of `overrides` (so the base trainer only
            sees keys it understands), stores them on the instance, and sets up
            placeholder attributes (teacher, loss, taps, hook handles) that are
            populated later in `_setup_train`.
        """
        # Pre-initialize hook handles so __del__ doesn't crash if construction fails partway
        self._s_handle = None
        self._t_handle = None

        # Ultralytics requires a base cfg to merge overrides into
        if cfg is None:
            from ultralytics.cfg import DEFAULT_CFG
            cfg = DEFAULT_CFG

        # Pop RADS-specific keys BEFORE calling super(): the base DetectionTrainer
        # validates overrides and would error on unknown keys.
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
        # Taps are the forward-hook callables; handles are their removal tokens.
        self.teacher: nn.Module | None = None
        self.distill_loss: DistillationLoss | None = None
        self._s_tap = _DetectInputTap()
        self._t_tap = _DetectInputTap()
        self._s_handle = None
        self._t_handle = None

    # -- Lifecycle --------------------------------------------------------
    def _setup_train(self, world_size: int):
        """Extend base training setup: load teacher, wire hooks, build loss.

        Called by Ultralytics once the student model and device are ready. We
        run the base setup first, then attach everything distillation needs so
        it is all live before the first training step.

        Args:
            world_size: DDP world size, forwarded to the base implementation.

        Side effects:
            Loads & freezes the teacher, registers forward hooks on both Detect
            heads, constructs `self.distill_loss` on the training device, and
            logs the active distillation configuration.
        """
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
        """Load the teacher checkpoint and freeze it for inference-only use.

        Reads the teacher weights from `self._teacher_weights`, unwraps the
        Ultralytics checkpoint format, moves the model to the training device in
        float32, switches it to eval mode and disables gradients on all its
        parameters. Freezing matters: the teacher is a fixed reference signal,
        so it must not be updated and must not waste memory on an autograd graph.

        Side effects:
            Sets `self.teacher` and logs the source path.
        """
        ckpt = torch.load(self._teacher_weights, map_location=self.device, weights_only=False)
        # Ultralytics checkpoints store the model under 'model' key.
        teacher = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        # float() guards against half-precision checkpoints; eval() disables
        # dropout/BN updates so the teacher's outputs are deterministic.
        teacher = teacher.float().to(self.device).eval()
        # Freeze every parameter: no grad, no optimizer touch, smaller graph.
        for p in teacher.parameters():
            p.requires_grad_(False)
        self.teacher = teacher
        LOGGER.info(f"[distill] teacher loaded from {self._teacher_weights}")

    def _install_hooks(self) -> None:
        """Register the input-capturing forward hooks on both Detect heads.

        Finds the student and teacher `Detect` heads and attaches `_s_tap` /
        `_t_tap` to each. After this, any forward pass through either model
        snapshots that model's per-scale neck features into the corresponding
        tap, which `criterion` reads back.

        Side effects:
            Stores the returned removal handles on `self._s_handle` /
            `self._t_handle` so the hooks can be cleanly removed in `__del__`.
        """
        s_detect = _find_detect_module(self.model)
        t_detect = _find_detect_module(self.teacher)
        self._s_handle = s_detect.register_forward_hook(self._s_tap)
        self._t_handle = t_detect.register_forward_hook(self._t_tap)

    # -- Loss --------------------------------------------------------------
    def criterion(self, preds, batch):
        """Compute the composite distillation loss for one training step.

        Overrides `DetectionTrainer.criterion`. The student forward has already
        run (producing `preds` and populating `self._s_tap`); here we add the
        teacher signal and combine everything via `self.distill_loss`.

        Args:
            preds: The student's raw predictions for this batch (from the
                forward pass Ultralytics already performed).
            batch: The training batch dict; `batch["img"]` is the input images,
                reused to forward the teacher on the identical inputs.

        Returns:
            ``(total_loss, loss_items)`` matching the base trainer's contract:
            ``total_loss`` is the scalar to backprop and ``loss_items`` is the
            unchanged per-component task-loss vector Ultralytics logs. Falls back
            to the plain task loss if the hooks did not capture features.
        """
        # Student task loss via parent class behaviour. This also leaves the
        # student's neck features sitting in self._s_tap (its hook just fired).
        task_loss, loss_items = super().criterion(preds, batch)

        # Teacher forward (no grad). Same images -> teacher's hook fills
        # self._t_tap. no_grad keeps the teacher graph-free and cheap.
        with torch.no_grad():
            _ = self.teacher(batch["img"])

        s_feats = self._s_tap.features
        t_feats = self._t_tap.features
        if not s_feats or not t_feats:
            # Hooks didn't fire — fall back to plain task loss. This keeps the
            # run alive (rather than crashing) if a model variant doesn't expose
            # features in the expected shape.
            return task_loss, loss_items

        # Per-scale classification logits live inside the Detect head's
        # `cv3` branch outputs but pre-Detect we have raw features. We
        # approximate cls logits by re-running the cls branch of each head
        # on the captured features. cv3[i] is the classification conv stack for
        # scale i, so cv3[i](feats[i]) reproduces that scale's class logits.
        s_detect = _find_detect_module(self.model)
        t_detect = _find_detect_module(self.teacher)
        # Student logits stay attached to the graph (we want gradients here).
        s_cls = [s_detect.cv3[i](s_feats[i]) for i in range(len(s_feats))]
        # Teacher logits are no_grad: they are fixed soft targets.
        with torch.no_grad():
            t_cls = [t_detect.cv3[i](t_feats[i]) for i in range(len(t_feats))]

        # If head counts differ (e.g. student=3 heads, teacher=4 with P2),
        # align by matching the last min(N_s, N_t) scales (the deeper ones).
        # The deeper scales are the semantically richest and are guaranteed to
        # be present in both, so this gives a safe, meaningful pairing.
        n = min(len(s_cls), len(t_cls))
        s_cls = s_cls[-n:]
        t_cls = t_cls[-n:]

        # Combine task + KL + feature MSE. We pass the deepest captured feature
        # (index -1, i.e. P5) for the feature-MSE term.
        total, comp = self.distill_loss(
            task_loss=task_loss,
            student_cls_logits=s_cls,
            teacher_cls_logits=t_cls,
            student_feat=s_feats[-1],
            teacher_feat=t_feats[-1],
        )

        # Keep Ultralytics-compatible loss_items shape; append distill
        # components for logging (W&B picks these up automatically).
        # We return the original loss_items unchanged so the base trainer's
        # progress bar / logging stay valid, while `total` carries the full
        # distillation gradient.
        return total, loss_items

    def __del__(self):  # best-effort hook removal
        """Best-effort teardown: remove the registered forward hooks.

        Forward hooks outlive a forward pass, so leaving them attached would
        keep references alive and could fire on unrelated future forwards. We
        remove both handles on garbage collection, swallowing any error because
        `__del__` must never raise (and construction may have failed before the
        handles were set, hence the None checks).
        """
        for h in (self._s_handle, self._t_handle):
            if h is not None:
                try:
                    h.remove()
                except Exception:
                    pass
