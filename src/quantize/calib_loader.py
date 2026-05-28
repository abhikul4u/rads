"""Calibration dataset utilities for INT8 post-training quantization.

We sample ~200 images from the training split (per Chapter 4.3.7) and provide
two flavors of loader:
  * `iter_numpy_chw`: yields preprocessed NCHW float32 tensors (TFLite needs HWC,
    but onnx2tf handles transposition — we keep CHW for consistency).
  * `tflite_representative_dataset`: generator factory matching the contract
    expected by TFLiteConverter.representative_dataset.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Iterator, List

import cv2
import numpy as np


def collect_calibration_images(
    dataset_root: Path,
    n_samples: int = 200,
    seed: int = 42,
) -> List[Path]:
    """Sample up to `n_samples` images from `<root>/train/images`."""
    train_dir = dataset_root / "train" / "images"
    if not train_dir.exists():
        raise FileNotFoundError(f"train/images not found under {dataset_root}")
    pool = sorted(p for p in train_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n_samples]


def _letterbox(im: np.ndarray, size: int = 640, color=(114, 114, 114)) -> np.ndarray:
    """Resize-and-pad to `size`x`size`, preserving aspect ratio."""
    h, w = im.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im_r = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = im_r
    return canvas


def iter_numpy_chw(
    paths: List[Path],
    imgsz: int = 640,
) -> Iterator[np.ndarray]:
    """Yield (1, 3, H, W) float32 in [0, 1], BGR→RGB."""
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            continue
        im = _letterbox(im, imgsz)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(np.float32) / 255.0
        im = np.transpose(im, (2, 0, 1))[None, ...]  # (1, 3, H, W)
        yield im


def tflite_representative_dataset(
    paths: List[Path],
    imgsz: int = 640,
) -> Callable[[], Iterator[List[np.ndarray]]]:
    """Return a generator factory matching TFLiteConverter's contract.

    TFLite expects HWC float32 in (1, H, W, 3).
    """
    def _gen() -> Iterator[List[np.ndarray]]:
        for p in paths:
            im = cv2.imread(str(p))
            if im is None:
                continue
            im = _letterbox(im, imgsz)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im.astype(np.float32) / 255.0
            im = im[None, ...]  # (1, H, W, 3)
            yield [im]

    return _gen
