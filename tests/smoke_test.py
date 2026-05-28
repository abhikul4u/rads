#!/usr/bin/env python
"""Fast smoke test for RADS Layer 3.

Target: catch the most common regressions in under 2 minutes on a CPU box
(faster on GPU). No network access, no real training, no real dataset.

What it checks:
    1. All 6 YAML configs parse and build via YOLO(cfg)
    2. Each model forward-passes a random 640x640 tensor
    3. Param counts stay within expected ranges (catches width-scale regressions)
    4. CBAM module: shape preservation + param budget sanity
    5. Size-aware loss installs/uninstalls cleanly + computes finite values
    6. Composite distillation loss: forward + backward on dummy tensors
    7. One micro training step on a synthetic batch (3 samples, 1 batch)
    8. ONNX export of the student (smallest model, fastest export)
    9. All 7 scripts respond to --help

Exit code: 0 if all pass, 1 if any fail. Prints a summary at the end.

Usage:
    python tests/smoke_test.py                  # full suite
    python tests/smoke_test.py --quick          # skip ONNX export (saves ~20s)
    python tests/smoke_test.py --no-train-step  # skip training step too
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

# Repo root on sys.path (this file lives at tests/smoke_test.py)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Suppress the noise — we want our own clean output
warnings.filterwarnings("ignore")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("YOLO_VERBOSE", "False")


# --- Pretty output -----------------------------------------------------------

class TestResult:
    PASS = "\033[92m✓\033[0m"
    FAIL = "\033[91m✗\033[0m"
    SKIP = "\033[93m○\033[0m"


def _print_header(text: str) -> None:
    print(f"\n\033[1m── {text} ──\033[0m")


@contextmanager
def _timed(label: str):
    t0 = time.perf_counter()
    print(f"  {label}…", end=" ", flush=True)
    try:
        yield
        dt = time.perf_counter() - t0
        print(f"{TestResult.PASS} ({dt:.1f}s)")
    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"{TestResult.FAIL} ({dt:.1f}s)")
        print(f"      \033[91m{type(e).__name__}: {str(e)[:300]}\033[0m")
        raise


# --- Expected ranges ---------------------------------------------------------
# These are the param budgets we verified on the verified Ultralytics 8.3.40 +
# scale 'l' build. Tolerances are generous (±5%) — they exist to catch
# order-of-magnitude regressions, not to enforce exact reproduction.
EXPECTED_PARAMS_M = {
    "baseline.yaml":         (41.0, 46.0),  # nominal 43.63M
    "cbam.yaml":             (41.0, 46.5),  # nominal 43.74M
    "p2head.yaml":           (40.0, 45.5),  # nominal 42.85M
    "sizeaware.yaml":        (41.0, 46.0),  # nominal 43.63M (same as baseline)
    "combined.yaml":         (40.0, 46.0),  # nominal 42.97M
    "distill_student.yaml":  (2.5,   3.5),  # nominal 3.01M
}


# --- Test functions ----------------------------------------------------------

def test_imports() -> None:
    """Every module imports without errors."""
    from src import paths  # noqa: F401
    from src.modules.cbam import CBAM  # noqa: F401
    from src.modules.register import register_all  # noqa: F401
    from src.losses.size_aware_loss import (  # noqa: F401
        SizeAwareBboxLoss,
        install_size_aware_loss,
        uninstall_size_aware_loss,
    )
    from src.distill.composite_loss import DistillationLoss  # noqa: F401
    from src.distill.trainer import DistillationTrainer  # noqa: F401
    from src.eval.metrics import run_eval  # noqa: F401
    from src.eval.fps_bench import gpu_fps, write_mobile_bundle  # noqa: F401
    from src.eval.aggregate_seeds import aggregate  # noqa: F401
    from src.quantize.ptq_int8 import export_all  # noqa: F401
    from src.quantize.calib_loader import collect_calibration_images  # noqa: F401
    from src.data.roboflow_pull import pull  # noqa: F401


def test_cbam_module() -> None:
    """CBAM preserves shape and stays within param budget."""
    import torch
    from src.modules.cbam import CBAM

    # Shape preservation across typical YOLOv8l neck channel counts.
    for c in (128, 256, 512, 1024):
        m = CBAM(c, c)  # Conv-style ctor
        x = torch.randn(1, c, 16, 16)
        y = m(x)
        assert y.shape == x.shape, f"CBAM with c={c} changed shape: {y.shape}"

    # Param budget: total across 6 typical neck insertions should be under 0.5M.
    total = sum(
        sum(p.numel() for p in CBAM(c, c).parameters())
        for c in (128, 256, 512, 512, 256, 512)
    )
    total_m = total / 1e6
    assert total_m < 0.5, f"CBAM total params {total_m:.3f}M exceeds 0.5M budget"


def test_all_configs_build() -> None:
    """Build each YAML through Ultralytics and verify param counts."""
    import torch
    from ultralytics import YOLO

    from src.modules.register import register_all
    register_all()

    failures = []
    for cfg_name, (lo_m, hi_m) in EXPECTED_PARAMS_M.items():
        cfg = ROOT / "configs" / cfg_name
        try:
            model = YOLO(str(cfg))
            n_params = sum(p.numel() for p in model.model.parameters())
            n_m = n_params / 1e6

            # Param range check
            if not (lo_m <= n_m <= hi_m):
                failures.append(
                    f"{cfg_name}: {n_m:.2f}M params outside [{lo_m}, {hi_m}]"
                )
                continue

            # Forward-pass check at 640
            x = torch.randn(1, 3, 640, 640)
            model.model.eval()
            with torch.no_grad():
                _ = model.model(x)
        except Exception as e:
            failures.append(f"{cfg_name}: {type(e).__name__}: {str(e)[:120]}")

    if failures:
        raise AssertionError("Config build failures:\n  " + "\n  ".join(failures))


def test_size_aware_loss_install_uninstall() -> None:
    """Patch installs, uninstalls, and is idempotent."""
    from ultralytics.utils import loss as loss_mod

    from src.losses.size_aware_loss import (
        install_size_aware_loss,
        uninstall_size_aware_loss,
    )

    # Idempotent install
    install_size_aware_loss(alpha=1.0)
    install_size_aware_loss(alpha=1.0)  # double install must be a no-op
    assert getattr(loss_mod, "_RADS_SIZE_AWARE_INSTALLED", False)

    # Clean uninstall
    uninstall_size_aware_loss()
    assert not getattr(loss_mod, "_RADS_SIZE_AWARE_INSTALLED", False)

    # Idempotent uninstall
    uninstall_size_aware_loss()  # must not raise


def test_size_aware_loss_forward_backward() -> None:
    """Size-aware loss computes finite values and produces gradients."""
    import torch
    from ultralytics import YOLO

    from src.modules.register import register_all
    from src.losses.size_aware_loss import (
        install_size_aware_loss,
        uninstall_size_aware_loss,
    )

    register_all()
    install_size_aware_loss(alpha=1.0)
    try:
        model = YOLO(str(ROOT / "configs" / "baseline.yaml"))
        m = model.model.train()
        # v8DetectionLoss expects `m.args` to be a namespace; YOLO() leaves it as dict
        if isinstance(m.args, dict):
            m.args = SimpleNamespace(**m.args)

        batch = {
            "img": torch.rand(2, 3, 320, 320),
            "cls": torch.tensor([[0], [1], [2]], dtype=torch.float32),
            "bboxes": torch.tensor([
                [0.5, 0.5, 0.2, 0.2],
                [0.3, 0.3, 0.1, 0.1],
                [0.7, 0.7, 0.15, 0.15],
            ], dtype=torch.float32),
            "batch_idx": torch.tensor([0, 0, 1], dtype=torch.float32),
        }

        preds = m(batch["img"])
        from ultralytics.utils.loss import v8DetectionLoss
        crit = v8DetectionLoss(m)
        loss, loss_items = crit(preds, batch)

        total = loss.sum()
        assert torch.isfinite(total), f"loss is non-finite: {total.item()}"
        assert total.item() > 0, f"loss is non-positive: {total.item()}"
        # Box, cls, dfl components are all present
        assert len(loss_items) == 3, f"expected 3 loss items, got {len(loss_items)}"

        # Backward must produce gradients on at least one param
        total.backward()
        grad_found = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in m.parameters() if p.requires_grad
        )
        assert grad_found, "no gradients produced after backward"
    finally:
        uninstall_size_aware_loss()


def test_composite_distill_loss() -> None:
    """Distillation loss: forward + backward + correct component scaling."""
    import torch
    from src.distill.composite_loss import DistillationLoss

    loss_fn = DistillationLoss(task_w=0.4, kl_w=0.4, feat_w=0.2, T=4.0)

    # Synthetic teacher/student tensors. Student smaller feature channels →
    # tests the projector path.
    task_loss = torch.tensor(1.5, requires_grad=True)
    s_cls = [torch.randn(2, 3, 8, 8, requires_grad=True)]
    t_cls = [torch.randn(2, 3, 8, 8)]
    s_feat = torch.randn(2, 256, 4, 4, requires_grad=True)
    t_feat = torch.randn(2, 512, 4, 4)

    total, components = loss_fn(task_loss, s_cls, t_cls, s_feat, t_feat)

    assert torch.isfinite(total), f"distill total is non-finite: {total.item()}"
    for k in ("loss_task", "loss_kl", "loss_feat", "loss_total"):
        assert k in components, f"missing component: {k}"
        assert torch.isfinite(components[k]), f"{k} non-finite: {components[k]}"

    # Backward through total must work
    total.backward()
    assert s_cls[0].grad is not None, "no grad on student classification logits"
    assert s_feat.grad is not None, "no grad on student features"


def test_micro_training_step() -> None:
    """One AdamW step on the student. Verifies loss + optimizer + backward
    integrate without the full Ultralytics trainer.
    """
    import torch
    from torch.optim import AdamW
    from ultralytics import YOLO

    from src.modules.register import register_all
    register_all()

    model = YOLO(str(ROOT / "configs" / "distill_student.yaml"))
    m = model.model.train()
    if isinstance(m.args, dict):
        m.args = SimpleNamespace(**m.args)

    opt = AdamW(m.parameters(), lr=1e-4)

    batch = {
        "img": torch.rand(2, 3, 320, 320),
        "cls": torch.tensor([[0], [1]], dtype=torch.float32),
        "bboxes": torch.tensor([
            [0.5, 0.5, 0.2, 0.2],
            [0.3, 0.3, 0.1, 0.1],
        ], dtype=torch.float32),
        "batch_idx": torch.tensor([0, 1], dtype=torch.float32),
    }

    # Capture initial param signature
    init_param = next(p.clone() for p in m.parameters() if p.requires_grad)

    from ultralytics.utils.loss import v8DetectionLoss
    crit = v8DetectionLoss(m)

    preds = m(batch["img"])
    loss, _ = crit(preds, batch)
    total = loss.sum()
    assert torch.isfinite(total), f"micro-step loss non-finite: {total.item()}"

    opt.zero_grad()
    total.backward()
    opt.step()

    # At least one param should have changed
    new_param = next(p for p in m.parameters() if p.requires_grad)
    assert not torch.allclose(init_param, new_param), \
        "no parameter changed after optimizer step"


def test_onnx_export_student() -> None:
    """Export the smallest model (student) to FP32 ONNX.

    Skipped under --quick. ONNX export catches a lot of "model graph isn't
    statically traceable" bugs that won't show up in training.
    """
    from ultralytics import YOLO

    from src.modules.register import register_all
    register_all()

    model = YOLO(str(ROOT / "configs" / "distill_student.yaml"))
    onnx_path = model.export(format="onnx", imgsz=320, opset=13, simplify=False)
    onnx_p = Path(onnx_path)
    assert onnx_p.exists(), f"ONNX file not created at {onnx_path}"
    size_mb = onnx_p.stat().st_size / 1e6
    assert 5 < size_mb < 30, f"ONNX file size {size_mb:.1f}MB outside plausible range"

    # Cleanup: ONNX may emit both <name>.onnx and <name>.onnx.data (external
    # weights for large models). Remove anything in the export directory.
    for sibling in onnx_p.parent.glob(onnx_p.stem + ".*"):
        try:
            sibling.unlink()
        except Exception:
            pass


def test_scripts_help() -> None:
    """Every script responds to --help (catches import errors fast)."""
    scripts = [
        "scripts/01_train_baseline.py",
        "scripts/02_train_ablation.py",
        "scripts/03_train_distill.py",
        "scripts/04_quantize_export.py",
        "scripts/05_evaluate.py",
        "scripts/07_aggregate_results.py",
    ]
    failures = []
    for s in scripts:
        path = ROOT / s
        result = subprocess.run(
            [sys.executable, str(path), "--help"],
            capture_output=True, text=True, timeout=30,
            cwd=ROOT,
        )
        if result.returncode != 0:
            failures.append(f"{s}: rc={result.returncode}: {result.stderr[-200:]}")
    if failures:
        raise AssertionError("Script --help failures:\n  " + "\n  ".join(failures))


def test_register_idempotent() -> None:
    """Calling register_all() twice doesn't double-patch the parser."""
    import ultralytics.nn.tasks as tasks_mod

    from src.modules.register import register_all
    register_all()
    parser_after_first = tasks_mod.parse_model
    register_all()  # second call should be a no-op
    parser_after_second = tasks_mod.parse_model
    assert parser_after_first is parser_after_second, \
        "register_all is not idempotent (parser swapped on second call)"


# --- Runner ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Fast smoke test for RADS Layer 3")
    ap.add_argument("--quick", action="store_true",
                    help="Skip ONNX export (saves ~20s)")
    ap.add_argument("--no-train-step", action="store_true",
                    help="Skip the micro training step")
    args = ap.parse_args()

    t_total = time.perf_counter()

    # Test plan: (name, function, skip_if)
    plan = [
        ("Imports",                          test_imports,                        False),
        ("CBAM module unit",                 test_cbam_module,                    False),
        ("register_all() idempotent",        test_register_idempotent,            False),
        ("All 6 configs build + forward",    test_all_configs_build,              False),
        ("Size-aware loss install/uninstall",test_size_aware_loss_install_uninstall, False),
        ("Size-aware loss forward+backward", test_size_aware_loss_forward_backward,  False),
        ("Distillation composite loss",      test_composite_distill_loss,         False),
        ("Micro training step",              test_micro_training_step,            args.no_train_step),
        ("ONNX export (student, 320px)",     test_onnx_export_student,            args.quick),
        ("Scripts respond to --help",        test_scripts_help,                   False),
    ]

    _print_header("RADS Layer 3 — Smoke Test")
    print(f"Python {sys.version.split()[0]} · repo at {ROOT}")

    passed, failed, skipped = 0, 0, 0
    failed_names = []

    for name, fn, skip in plan:
        if skip:
            print(f"  {name}… {TestResult.SKIP} (skipped)")
            skipped += 1
            continue
        try:
            with _timed(name):
                fn()
            passed += 1
        except Exception:
            failed += 1
            failed_names.append(name)
            # Continue with other tests rather than stopping

    elapsed = time.perf_counter() - t_total
    _print_header(f"Summary  ({elapsed:.1f}s)")
    print(f"  {TestResult.PASS} {passed} passed")
    if skipped:
        print(f"  {TestResult.SKIP} {skipped} skipped")
    if failed:
        print(f"  {TestResult.FAIL} {failed} failed:")
        for n in failed_names:
            print(f"      - {n}")
        print()
        sys.exit(1)
    print()
    sys.exit(0)


if __name__ == "__main__":
    main()
