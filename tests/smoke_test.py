#!/usr/bin/env python
"""Fast smoke test for RADS Layer 3.

Author: Rutuja Kulkarni

Role in the RADS Layer 3 pipeline
---------------------------------
This is the *pre-flight safety check* for the entire Layer 3 model-development
pipeline. Real training runs cost hours of A100 time on RunPod, so before kicking
off any of them — and on every push, via ``.github/workflows/smoke.yml`` — this
suite runs a ~40 second batch of cheap, CPU-friendly checks that exercise every
custom piece of the codebase end-to-end without touching the network, a GPU, real
data, or real training. If a refactor silently breaks model construction, the
custom CBAM attention module, the monkey-patched size-aware loss, the distillation
loss/trainer, or ONNX export, this catches it in seconds instead of after a
multi-hour run produces garbage (or crashes at hour three).

It is intentionally self-contained: it puts the repo root on ``sys.path``,
disables W&B and Ultralytics verbosity, and builds models straight from the six
YAML configs under ``configs/``. Every check is a plain ``test_*`` function;
:func:`main` runs them as a plan, prints a colourised pass/fail/skip line per
test with timing, keeps going after failures (so one break doesn't mask others),
and exits non-zero if anything failed — which is what makes it usable as a CI gate
and a manual go/no-go before training.

The six configs and three classes (MH/PH/WLPH) referenced throughout match the
rest of the pipeline; the param budgets in :data:`EXPECTED_PARAMS_M` are the
guard-rails that catch architecture/width-scale regressions.

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
    """Namespace of ANSI-coloured status glyphs for the console report.

    Pure presentation: green tick for pass, red cross for fail, yellow circle for
    skip. Kept as a tiny class purely so the glyphs read as ``TestResult.PASS``
    etc. at the call sites; the ``\\033[..m`` sequences are terminal colour codes.
    """

    PASS = "\033[92m✓\033[0m"
    FAIL = "\033[91m✗\033[0m"
    SKIP = "\033[93m○\033[0m"


def _print_header(text: str) -> None:
    """Print a bold section header (helps visually separate suite phases)."""
    print(f"\n\033[1m── {text} ──\033[0m")


@contextmanager
def _timed(label: str):
    """Context manager that prints a timed PASS/FAIL line around a test body.

    Wrapping each test in this gives uniform output: it prints the label, runs the
    body, and on success reports the elapsed time with a tick. On any exception it
    reports a cross plus a truncated error message, then re-raises so the runner in
    :func:`main` can record the failure. Re-raising (rather than swallowing) is
    deliberate — the runner needs the exception to count the test as failed.

    Args:
        label: Human-readable name of the check being timed.
    """
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
    """Guards against import-time breakage anywhere in ``src``.

    Imports every public entry point the pipeline relies on (paths, CBAM, the
    registration hook, the size-aware and distillation losses, the distillation
    trainer, eval/quantize/data modules). Catches the failure mode where a syntax
    error, bad relative import, or renamed symbol in any one module would crash the
    real training scripts on startup — surfacing it here in milliseconds instead.
    """
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
    """Guards the CBAM attention module's two hard contracts: shape and size.

    Two regressions are guarded here. (1) Shape preservation: CBAM is dropped into
    the YOLOv8l neck, so its output tensor MUST have the same shape as its input
    for the surrounding graph to stay valid — checked across the typical neck
    channel counts. (2) Parameter budget: CBAM is meant to be a near-free attention
    add-on, so the total parameters across a representative set of six neck
    insertions must stay under 0.5M; a refactor that accidentally inflated its cost
    (e.g. a too-large reduction ratio or extra conv) would break the thesis claim
    that CBAM adds negligible parameters, and is caught here.
    """
    import torch
    from src.modules.cbam import CBAM

    # Shape preservation across typical YOLOv8l neck channel counts.
    for c in (128, 256, 512, 1024):
        m = CBAM(c, c)  # Conv-style ctor (in_channels, out_channels)
        x = torch.randn(1, c, 16, 16)
        y = m(x)
        assert y.shape == x.shape, f"CBAM with c={c} changed shape: {y.shape}"

    # Param budget: total across 6 typical neck insertions should be under 0.5M.
    total = sum(
        sum(p.numel() for p in CBAM(c, c).parameters())
        for c in (128, 256, 512, 512, 256, 512)
    )
    total_m = total / 1e6  # convert raw param count to millions
    assert total_m < 0.5, f"CBAM total params {total_m:.3f}M exceeds 0.5M budget"


def test_all_configs_build() -> None:
    """Guards that all six configs still build, size correctly, and forward-pass.

    This is the broadest architectural guard. After installing the custom modules
    via ``register_all()``, it builds every one of the six YAML configs through
    Ultralytics and asserts two things per config: (1) the parameter count falls
    inside its ``EXPECTED_PARAMS_M`` window — catching width/scale regressions or a
    custom block silently changing size — and (2) the model survives a forward pass
    on a real 640x640 tensor in eval mode, catching graph-wiring bugs (mismatched
    channels, bad concat indices) that a pure param count would miss. Failures are
    collected across all configs and reported together, so one broken config
    doesn't hide the others.
    """
    import torch
    from ultralytics import YOLO

    from src.modules.register import register_all
    register_all()  # ensure custom blocks (CBAM, etc.) are known to the YAML parser

    failures = []
    for cfg_name, (lo_m, hi_m) in EXPECTED_PARAMS_M.items():
        cfg = ROOT / "configs" / cfg_name
        try:
            model = YOLO(str(cfg))
            n_params = sum(p.numel() for p in model.model.parameters())
            n_m = n_params / 1e6  # params in millions, to compare against the budget

            # Param range check — must sit within the per-config tolerance window.
            if not (lo_m <= n_m <= hi_m):
                failures.append(
                    f"{cfg_name}: {n_m:.2f}M params outside [{lo_m}, {hi_m}]"
                )
                continue  # no point forward-passing a wrongly-sized model

            # Forward-pass check at the training resolution (640), grad-free.
            x = torch.randn(1, 3, 640, 640)
            model.model.eval()
            with torch.no_grad():
                _ = model.model(x)
        except Exception as e:
            # Record (truncated) error and keep testing the remaining configs.
            failures.append(f"{cfg_name}: {type(e).__name__}: {str(e)[:120]}")

    if failures:
        raise AssertionError("Config build failures:\n  " + "\n  ".join(failures))


def test_size_aware_loss_install_uninstall() -> None:
    """Guards the lifecycle of the monkey-patched size-aware loss.

    The size-aware loss is installed by monkey-patching Ultralytics' loss module.
    That makes its install/uninstall lifecycle a correctness hazard: a
    non-idempotent install could patch-over-a-patch (double-wrapping the loss), and
    a failed/partial uninstall could leak the patch into an unrelated run. This
    test asserts that installing twice is a no-op (the ``_RADS_SIZE_AWARE_INSTALLED``
    sentinel flips on exactly once), that uninstall cleanly clears the sentinel, and
    that a redundant uninstall does not raise. It uses the module-level sentinel as
    the observable proof of state.
    """
    from ultralytics.utils import loss as loss_mod

    from src.losses.size_aware_loss import (
        install_size_aware_loss,
        uninstall_size_aware_loss,
    )

    # Idempotent install — calling twice must leave exactly one patch in place.
    install_size_aware_loss(alpha=1.0)
    install_size_aware_loss(alpha=1.0)  # double install must be a no-op
    assert getattr(loss_mod, "_RADS_SIZE_AWARE_INSTALLED", False)

    # Clean uninstall — the sentinel must be cleared so other tests/runs are unaffected.
    uninstall_size_aware_loss()
    assert not getattr(loss_mod, "_RADS_SIZE_AWARE_INSTALLED", False)

    # Idempotent uninstall — uninstalling when not installed must not raise.
    uninstall_size_aware_loss()  # must not raise


def test_size_aware_loss_forward_backward() -> None:
    """Guards that the patched size-aware loss is numerically sound and trainable.

    Installing the patch is not enough — it must actually produce a usable training
    signal. With the patch active, this builds a model, runs a real
    ``v8DetectionLoss`` over a tiny synthetic batch, and asserts the total loss is
    finite and strictly positive (catching NaN/inf or a degenerate zero loss from a
    bad reweighting), that all three loss components (box/cls/dfl) are present, and
    that ``backward()`` produces a non-zero gradient on at least one parameter
    (proving the size-aware term is actually wired into the autograd graph). The
    ``finally`` block always uninstalls so this test cannot pollute later tests.
    """
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
        m = model.model.train()  # train() mode so the detection loss path is active
        # v8DetectionLoss expects `m.args` to be a namespace; YOLO() leaves it as dict
        if isinstance(m.args, dict):
            m.args = SimpleNamespace(**m.args)

        # Minimal synthetic batch: 2 images, 3 boxes total spread across both
        # (batch_idx maps each box row to its image), one per RADS class.
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
        crit = v8DetectionLoss(m)  # picks up the patched size-aware behaviour
        loss, loss_items = crit(preds, batch)

        total = loss.sum()
        assert torch.isfinite(total), f"loss is non-finite: {total.item()}"
        assert total.item() > 0, f"loss is non-positive: {total.item()}"
        # Box, cls, dfl components are all present
        assert len(loss_items) == 3, f"expected 3 loss items, got {len(loss_items)}"

        # Backward must produce gradients on at least one param — proves the loss
        # is differentiable and connected to the model's trainable weights.
        total.backward()
        grad_found = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in m.parameters() if p.requires_grad
        )
        assert grad_found, "no gradients produced after backward"
    finally:
        # Always undo the patch, even on failure, so other tests start clean.
        uninstall_size_aware_loss()


def test_composite_distill_loss() -> None:
    """Guards the composite distillation loss: finite, complete, and trainable.

    The distillation loss blends three terms (task + KL on class logits + feature
    mimicry). This test feeds it synthetic teacher/student tensors where the
    student has FEWER feature channels (256 vs the teacher's 512) — deliberately
    exercising the projector path that aligns mismatched channel counts, a common
    place for shape bugs. It asserts the total is finite, that all four reported
    components (``loss_task``/``loss_kl``/``loss_feat``/``loss_total``) exist and
    are finite, and that backward propagates gradients to BOTH the student class
    logits and the student features — proving every branch of the loss contributes
    to learning rather than silently detaching.
    """
    import torch
    from src.distill.composite_loss import DistillationLoss

    loss_fn = DistillationLoss(task_w=0.4, kl_w=0.4, feat_w=0.2, T=4.0)

    # Synthetic teacher/student tensors. Student has smaller feature channels
    # (256 vs 512) → forces the loss through its channel-projection path.
    task_loss = torch.tensor(1.5, requires_grad=True)
    s_cls = [torch.randn(2, 3, 8, 8, requires_grad=True)]  # student class logits
    t_cls = [torch.randn(2, 3, 8, 8)]                      # teacher class logits (no grad)
    s_feat = torch.randn(2, 256, 4, 4, requires_grad=True)  # student feature map
    t_feat = torch.randn(2, 512, 4, 4)                      # teacher feature map (wider)

    total, components = loss_fn(task_loss, s_cls, t_cls, s_feat, t_feat)

    assert torch.isfinite(total), f"distill total is non-finite: {total.item()}"
    for k in ("loss_task", "loss_kl", "loss_feat", "loss_total"):
        assert k in components, f"missing component: {k}"
        assert torch.isfinite(components[k]), f"{k} non-finite: {components[k]}"

    # Backward through total must reach both student-side inputs (logits + features).
    total.backward()
    assert s_cls[0].grad is not None, "no grad on student classification logits"
    assert s_feat.grad is not None, "no grad on student features"


def test_micro_training_step() -> None:
    """Guards the full train loop in miniature: loss -> backward -> optimizer step.

    The previous tests check the loss in isolation; this one verifies the three
    moving parts integrate into an actual weight update WITHOUT relying on the heavy
    Ultralytics trainer. It snapshots a parameter, runs one forward/backward/AdamW
    step on a synthetic batch, and asserts the loss was finite and that at least one
    parameter actually changed. The failure mode it guards: a model that builds and
    computes a loss but doesn't learn — e.g. params frozen by accident, gradients
    not flowing, or the optimizer not seeing the parameters.
    """
    import torch
    from torch.optim import AdamW
    from ultralytics import YOLO

    from src.modules.register import register_all
    register_all()

    model = YOLO(str(ROOT / "configs" / "distill_student.yaml"))
    m = model.model.train()  # train mode: enables the detection-loss forward path
    if isinstance(m.args, dict):
        m.args = SimpleNamespace(**m.args)  # see note in size-aware test re: m.args

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

    # Snapshot the first trainable param BEFORE the step (clone so it isn't
    # mutated in place) so we can prove it changed afterwards.
    init_param = next(p.clone() for p in m.parameters() if p.requires_grad)

    from ultralytics.utils.loss import v8DetectionLoss
    crit = v8DetectionLoss(m)

    preds = m(batch["img"])
    loss, _ = crit(preds, batch)
    total = loss.sum()
    assert torch.isfinite(total), f"micro-step loss non-finite: {total.item()}"

    # Standard PyTorch update triad.
    opt.zero_grad()
    total.backward()
    opt.step()

    # The same (first) param must now differ — the optimizer actually moved weights.
    new_param = next(p for p in m.parameters() if p.requires_grad)
    assert not torch.allclose(init_param, new_param), \
        "no parameter changed after optimizer step"


def test_onnx_export_student() -> None:
    """Guards deployability: the student must export to a sane FP32 ONNX graph.

    ONNX export is the bridge to the deployed/quantised model, and it is far
    stricter than training: it traces the whole graph statically, so any
    dynamic/Python-control-flow construct or non-traceable custom op fails here even
    though training passed. This exports the smallest model (the distilled student,
    fastest to export — hence why this and not a 43M-param variant) at 320px and
    asserts the file exists and its size is in a plausible band (5–30 MB), catching
    both export crashes and a suspiciously tiny/huge graph. Skipped under
    ``--quick`` because it is the slowest check (~20s).
    """
    from ultralytics import YOLO

    from src.modules.register import register_all
    register_all()

    model = YOLO(str(ROOT / "configs" / "distill_student.yaml"))
    # simplify=False keeps the test independent of onnxsim being installed.
    onnx_path = model.export(format="onnx", imgsz=320, opset=13, simplify=False)
    onnx_p = Path(onnx_path)
    assert onnx_p.exists(), f"ONNX file not created at {onnx_path}"
    size_mb = onnx_p.stat().st_size / 1e6
    # Plausibility band: catches a corrupt empty graph (too small) or weights
    # accidentally written external/duplicated (too large).
    assert 5 < size_mb < 30, f"ONNX file size {size_mb:.1f}MB outside plausible range"

    # Cleanup: ONNX may emit both <name>.onnx and <name>.onnx.data (external
    # weights for large models). Remove anything in the export directory so the
    # test leaves no artifacts behind. Unlink failures are non-fatal (best-effort).
    for sibling in onnx_p.parent.glob(onnx_p.stem + ".*"):
        try:
            sibling.unlink()
        except Exception:
            pass


def test_scripts_help() -> None:
    """Guards that every pipeline entry-point script still starts up.

    Runs each of the six top-level scripts with ``--help`` in a subprocess and
    requires a zero return code. ``--help`` forces Python to import the whole
    module (and its argparse setup) but stops before doing any real work, so this
    is a fast, side-effect-free way to catch import errors, typos, or broken
    argument definitions in the scripts a user actually invokes — before they fail
    minutes into a real run. Failures across scripts are aggregated into one report.
    """
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
        # Fresh subprocess per script (isolated import) with a 30s safety timeout;
        # cwd=ROOT so the scripts' repo-relative imports resolve.
        result = subprocess.run(
            [sys.executable, str(path), "--help"],
            capture_output=True, text=True, timeout=30,
            cwd=ROOT,
        )
        if result.returncode != 0:
            # Keep the tail of stderr — that's where the traceback/usage error is.
            failures.append(f"{s}: rc={result.returncode}: {result.stderr[-200:]}")
    if failures:
        raise AssertionError("Script --help failures:\n  " + "\n  ".join(failures))


def test_register_idempotent() -> None:
    """Guards that register_all() is idempotent — no double-patching the parser.

    ``register_all()`` works by swapping in a patched Ultralytics ``parse_model``.
    Several scripts (and several tests above) call it, so calling it twice MUST be
    safe. This compares the ``parse_model`` reference before and after a second
    call and asserts it is the *same object* — proving the second call short-circuits
    rather than wrapping the already-patched parser again, which would compound the
    patch and could corrupt model construction.
    """
    import ultralytics.nn.tasks as tasks_mod

    from src.modules.register import register_all
    register_all()
    parser_after_first = tasks_mod.parse_model
    register_all()  # second call should be a no-op
    parser_after_second = tasks_mod.parse_model
    # Identity check (is), not equality: the reference must be unchanged.
    assert parser_after_first is parser_after_second, \
        "register_all is not idempotent (parser swapped on second call)"


# --- Runner ------------------------------------------------------------------

def main():
    """Run the full smoke-test plan and exit 0 (all pass) or 1 (any fail).

    Builds an ordered test plan of ``(name, function, skip_flag)`` tuples, honours
    the ``--quick`` (skip ONNX) and ``--no-train-step`` flags via those skip flags,
    runs each non-skipped test inside :func:`_timed`, and — crucially — keeps going
    after a failure so a single break doesn't mask later ones. Prints a final
    pass/skip/fail summary with total elapsed time and sets the process exit code
    accordingly so CI and pre-training shell gates can branch on it.
    """
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
            with _timed(name):  # prints the timed PASS/FAIL line for this test
                fn()
            passed += 1
        except Exception:
            # _timed already printed the error; record and press on so one failure
            # doesn't hide the rest of the suite.
            failed += 1
            failed_names.append(name)

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
