"""INT8 post-training quantization + multi-format export.

Author: Rutuja Kulkarni

Role in the RADS Layer 3 (Model Development) pipeline
-----------------------------------------------------
This is the deployment-export driver. Once a RADS variant has been trained and
evaluated in float32, it must be shrunk and converted into the formats the
target platforms actually run: ONNX for cross-platform / server inference and
INT8 TFLite for the on-device Android (Snapdragon) deployment. INT8
post-training quantization roughly quarters the model size and speeds up mobile
inference, at the cost of a small, measured accuracy drop — which the thesis
reports alongside the FP32 baseline.

Strategy: rather than hand-rolling the quantization, this module leans on
Ultralytics' OFFICIAL ``model.export()`` for every format. For TFLite it drives
the maintained ``ONNX → TF SavedModel → TFLite`` chain via ``onnx2tf``; for INT8
it passes the dataset YAML so Ultralytics' exporter performs calibration over
representative images. We stage a fixed 200-image calibration sample (sourced
from ``calib_loader.collect_calibration_images``) on disk where the exporter
expects it, then produce three artifacts (FP32 ONNX, INT8 ONNX, INT8 TFLite) and
print their on-disk sizes for the thesis comparison table.

Strategy: lean on Ultralytics' official `model.export()` for both ONNX and
TFLite. It handles the ONNX → TF SavedModel → TFLite chain via `onnx2tf`,
which is the maintained path. INT8 calibration uses our 200-image sample.

Outputs (per run name):
    artifacts/exports/<name>/
        ├── <name>.onnx            (FP32 — for cross-platform/server inference)
        ├── <name>.int8.onnx       (INT8 dynamic — ONNX Runtime)
        └── <name>_saved_model/
            └── <name>_full_integer_quant.tflite   (INT8 — Android)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from src.modules.register import register_all
from src.paths import CALIB_DIR, EXPORTS_DIR, RADS_DATA
from src.quantize.calib_loader import collect_calibration_images


def export_all(
    weights: Path,
    name: str,
    imgsz: int = 640,
    calib_n: int = 200,
    dataset_root: Path | None = None,
) -> Path:
    """Export ``weights`` to FP32 ONNX, INT8 ONNX, and INT8 TFLite.

    Drives the full deployment-export flow for a single checkpoint: resolve the
    dataset (needed for INT8 calibration), stage a calibration image sample,
    then call Ultralytics' exporter three times to produce the three artifacts,
    copying each into a per-name export directory.

    Args:
        weights: Path to the trained ``.pt`` checkpoint.
        name: Export name; becomes the output sub-directory under EXPORTS_DIR
            and the filename stem of every artifact.
        imgsz: Square export resolution (mobile target uses 640).
        calib_n: Number of calibration images to sample (~200).
        dataset_root: Dataset root to calibrate against; if ``None`` the newest
            versioned dataset under ``RADS_DATA`` is auto-selected.

    Returns:
        The export directory containing all produced artifacts.

    Raises:
        FileNotFoundError: if no dataset can be located when one is needed.

    Side effects:
        Creates/overwrites the calibration staging dir and the export dir, and
        prints progress plus a file-size report.
    """
    # Register RADS custom modules so the custom-architecture checkpoint loads.
    register_all()

    out_dir = EXPORTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve which dataset to calibrate against. When unspecified, pick the
    # most recent versioned export (names sort so the last is newest, e.g.
    # "rads-v3" > "rads-v2"). INT8 calibration NEEDS this data.
    if dataset_root is None:
        candidates = sorted(RADS_DATA.glob("*-v*"))
        if not candidates:
            raise FileNotFoundError(
                f"No dataset under {RADS_DATA}. Run 00_pull_dataset.py first."
            )
        dataset_root = candidates[-1]

    # Stage calibration images on disk where Ultralytics' exporter expects them.
    # We copy the seeded sample into a clean per-name staging dir (wiping any
    # previous one) so each export run starts from a known, isolated set.
    calib_paths = collect_calibration_images(dataset_root, n_samples=calib_n)
    calib_stage = CALIB_DIR / name
    if calib_stage.exists():
        shutil.rmtree(calib_stage)
    calib_stage.mkdir(parents=True)
    for p in calib_paths:
        shutil.copy(p, calib_stage / p.name)
    print(f"[calib] staged {len(calib_paths)} images at {calib_stage}")

    # NOTE: each format reloads the checkpoint into a fresh YOLO instance. This
    # is deliberate — export() mutates / fuses the in-memory model, so reusing a
    # single instance across formats risks one export corrupting the next.

    # --- 1. FP32 ONNX (cross-platform reference) ---
    # Static shape (dynamic=False) + graph simplification; opset 13 for broad
    # runtime compatibility. This is the unquantized reference model.
    model = YOLO(str(weights))
    onnx_fp32 = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=13,
        dynamic=False,
        simplify=True,
    )
    shutil.copy(onnx_fp32, out_dir / f"{name}.onnx")
    print(f"[ok] FP32 ONNX → {out_dir / f'{name}.onnx'}")

    # --- 2. INT8 ONNX ---
    # int8=True triggers calibrated quantization; passing the dataset YAML lets
    # Ultralytics feed representative images through to fit the INT8 scales.
    # Target: ONNX Runtime INT8 inference.
    model = YOLO(str(weights))
    onnx_int8 = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=13,
        int8=True,
        simplify=True,
        data=str(dataset_root / "data.yaml"),
    )
    shutil.copy(onnx_int8, out_dir / f"{name}.int8.onnx")
    print(f"[ok] INT8 ONNX → {out_dir / f'{name}.int8.onnx'}")

    # --- 3. INT8 TFLite (Android target) ---
    # The TFLite export runs the ONNX -> TF SavedModel -> TFLite chain (onnx2tf)
    # internally and performs full-integer calibration over the dataset images.
    # This is the artifact that actually ships to the Snapdragon device.
    model = YOLO(str(weights))
    tflite_path = model.export(
        format="tflite",
        imgsz=imgsz,
        int8=True,
        data=str(dataset_root / "data.yaml"),
    )
    shutil.copy(tflite_path, out_dir / f"{name}.int8.tflite")
    print(f"[ok] INT8 TFLite → {out_dir / f'{name}.int8.tflite'}")

    # Report file sizes for the thesis table — the size reduction from FP32 ONNX
    # to the INT8 variants is itself a reported quantization result.
    print("\n[sizes]")
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            mb = f.stat().st_size / 1e6  # bytes -> megabytes (decimal MB)
            print(f"  {f.name:35s} {mb:7.2f} MB")
    return out_dir


if __name__ == "__main__":
    # CLI entry point: given a checkpoint and an export name, run the full
    # FP32/INT8 export flow with the requested resolution and calibration size.
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path, help="Path to .pt weights")
    ap.add_argument("--name", required=True, help="Export name (becomes subdir)")
    ap.add_argument("--imgsz", type=int, default=640)        # export resolution
    ap.add_argument("--calib-n", type=int, default=200)      # calibration sample size
    args = ap.parse_args()
    export_all(args.weights, args.name, args.imgsz, args.calib_n)
