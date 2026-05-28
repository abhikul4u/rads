"""INT8 post-training quantization + multi-format export.

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
    """Export `weights` to FP32 ONNX, INT8 ONNX, and INT8 TFLite.

    Returns the export directory.
    """
    register_all()

    out_dir = EXPORTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage calibration images on disk where Ultralytics' exporter expects them.
    if dataset_root is None:
        candidates = sorted(RADS_DATA.glob("*-v*"))
        if not candidates:
            raise FileNotFoundError(
                f"No dataset under {RADS_DATA}. Run 00_pull_dataset.py first."
            )
        dataset_root = candidates[-1]

    calib_paths = collect_calibration_images(dataset_root, n_samples=calib_n)
    calib_stage = CALIB_DIR / name
    if calib_stage.exists():
        shutil.rmtree(calib_stage)
    calib_stage.mkdir(parents=True)
    for p in calib_paths:
        shutil.copy(p, calib_stage / p.name)
    print(f"[calib] staged {len(calib_paths)} images at {calib_stage}")

    # --- 1. FP32 ONNX (cross-platform reference) ---
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
    model = YOLO(str(weights))
    tflite_path = model.export(
        format="tflite",
        imgsz=imgsz,
        int8=True,
        data=str(dataset_root / "data.yaml"),
    )
    shutil.copy(tflite_path, out_dir / f"{name}.int8.tflite")
    print(f"[ok] INT8 TFLite → {out_dir / f'{name}.int8.tflite'}")

    # Report file sizes for the thesis table.
    print("\n[sizes]")
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            mb = f.stat().st_size / 1e6
            print(f"  {f.name:35s} {mb:7.2f} MB")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path, help="Path to .pt weights")
    ap.add_argument("--name", required=True, help="Export name (becomes subdir)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--calib-n", type=int, default=200)
    args = ap.parse_args()
    export_all(args.weights, args.name, args.imgsz, args.calib_n)
