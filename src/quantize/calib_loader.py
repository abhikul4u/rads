"""Calibration dataset utilities for INT8 post-training quantization.

Author: Rutuja Kulkarni

Role in the RADS Layer 3 (Model Development) pipeline
-----------------------------------------------------
INT8 post-training quantization (PTQ) needs a small "representative dataset" so
the quantizer can observe the real distribution of activations and choose the
per-tensor scale/zero-point that minimises information loss when float32 values
are mapped onto 8-bit integers. Get this calibration set wrong (too few images,
wrong preprocessing) and the quantized model's accuracy collapses.

This module is the data side of that calibration. It (a) samples a fixed,
seeded subset of training images so the calibration is reproducible across runs,
and (b) preprocesses each image EXACTLY the way the model expects at inference
time — letterbox resize preserving aspect ratio, BGR→RGB channel swap, and
[0, 1] normalisation — because the quantizer must see the same numeric
distribution the deployed model will see. ``ptq_int8.py`` consumes the sampled
paths from here when driving the INT8 export.

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
    """Sample up to ``n_samples`` images from ``<root>/train/images``.

    Draws a reproducible random subset of the training images to serve as the
    PTQ calibration set. We sample from TRAIN (not val/test) so calibration
    never touches the data the model is finally evaluated on.

    Args:
        dataset_root: Dataset root; must contain ``train/images``.
        n_samples: Maximum number of images to return (~200 per Chapter 4.3.7).
        seed: RNG seed making the sample deterministic — important so the same
            calibration set is reused across export runs and is auditable.

    Returns:
        A list of image paths (length ``min(n_samples, available)``).

    Raises:
        FileNotFoundError: if ``train/images`` does not exist under the root.
    """
    train_dir = dataset_root / "train" / "images"
    if not train_dir.exists():
        raise FileNotFoundError(f"train/images not found under {dataset_root}")
    # sorted() first gives a deterministic base ordering regardless of the
    # filesystem's directory enumeration order...
    pool = sorted(p for p in train_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    # ...then a SEEDED shuffle makes the random draw itself reproducible.
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n_samples]


def _letterbox(im: np.ndarray, size: int = 640, color=(114, 114, 114)) -> np.ndarray:
    """Resize-and-pad to ``size``x``size`` preserving aspect ratio.

    This reproduces YOLO's standard "letterbox" preprocessing: scale the image
    down by a single factor so the longer side fits, then pad the remainder with
    a neutral grey (114) to reach a square canvas. Using the identical transform
    here ensures the calibration activations match those at inference time.

    Args:
        im: Input image as an HWC uint8 array (OpenCV BGR).
        size: Target square side length.
        color: Padding colour (default YOLO grey 114,114,114).

    Returns:
        A ``(size, size, 3)`` uint8 canvas with the resized image centred.
    """
    h, w = im.shape[:2]
    # Single scale factor = min over both axes -> aspect ratio preserved, image
    # never upscaled past the box on either dimension.
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im_r = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    # Pre-fill the full square with the padding colour...
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    # ...then centre the resized image (equal padding on opposing sides).
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = im_r
    return canvas


def iter_numpy_chw(
    paths: List[Path],
    imgsz: int = 640,
) -> Iterator[np.ndarray]:
    """Yield ``(1, 3, H, W)`` float32 tensors in [0, 1], BGR→RGB.

    Channels-first (NCHW) flavour of the calibration loader, matching ONNX /
    PyTorch tensor layout. Each image is letterboxed, colour-corrected and
    normalised exactly as at inference time.

    Args:
        paths: Image paths to iterate (typically from
            ``collect_calibration_images``).
        imgsz: Square input resolution.

    Yields:
        One ``(1, 3, imgsz, imgsz)`` float32 array per readable image.
        Unreadable files are silently skipped.
    """
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            # cv2 returns None on a decode failure; skip rather than crash.
            continue
        im = _letterbox(im, imgsz)
        # OpenCV loads BGR; the model was trained on RGB, so swap channels.
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # Scale 0-255 ints to the [0, 1] float range the network expects.
        im = im.astype(np.float32) / 255.0
        # HWC -> CHW and prepend a batch dim of 1 -> (1, 3, H, W).
        im = np.transpose(im, (2, 0, 1))[None, ...]  # (1, 3, H, W)
        yield im


def tflite_representative_dataset(
    paths: List[Path],
    imgsz: int = 640,
) -> Callable[[], Iterator[List[np.ndarray]]]:
    """Return a generator factory matching TFLiteConverter's contract.

    ``tf.lite.TFLiteConverter.representative_dataset`` expects a *callable* that,
    when invoked, returns a fresh generator yielding a LIST of input tensors per
    sample (one entry per model input). We therefore return ``_gen`` itself, not
    the generator, so the converter can iterate it as many times as it needs.

    Unlike the NCHW loader, TFLite expects channels-LAST HWC float32 in
    ``(1, H, W, 3)``.

    Args:
        paths: Calibration image paths.
        imgsz: Square input resolution.

    Returns:
        A zero-argument factory producing the representative-dataset generator.
    """
    def _gen() -> Iterator[List[np.ndarray]]:
        for p in paths:
            im = cv2.imread(str(p))
            if im is None:
                # Skip undecodable images.
                continue
            # Identical preprocessing to the NCHW loader, except we keep HWC
            # layout because TFLite's input tensor is channels-last.
            im = _letterbox(im, imgsz)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im.astype(np.float32) / 255.0
            im = im[None, ...]  # (1, H, W, 3)
            # The converter contract wants a list of inputs per step; our model
            # has a single image input, hence a one-element list.
            yield [im]

    return _gen
