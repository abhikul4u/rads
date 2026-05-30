#!/usr/bin/env python
"""Annotation quality audit for YOLO-format datasets.

Author: Rutuja Kulkarni

Role in the RADS Layer 3 pipeline
---------------------------------
This script is the *data-hygiene gate* that runs before any expensive model
development on the RADS (Road Anomaly Detection System) dataset. The Layer 3
pipeline trains YOLOv8 variants for several hours on RunPod A100s, so feeding it
mislabelled or malformed annotations wastes both money and time and quietly
corrupts the thesis results. Rather than discover label problems via mysteriously
poor mAP after a multi-hour run, this auditor surfaces them up front with a cheap,
model-free static scan of the YOLO ``.txt`` label files.

It is deliberately dependency-light (only the standard library, plus PIL purely as
an *optional* extra for image-readability checks) so it can run anywhere — on the
pod, on a laptop, or inside CI — without importing torch or Ultralytics.

The dataset is the 3-class RADS detection set: ``MH`` (manhole), ``PH`` (pothole),
``WLPH`` (water-logged pothole). The class order matters because YOLO labels store
the class as a bare integer index, so the auditor must be told the class names in
the exact YAML order to map indices back to names and to flag out-of-range IDs.

Each finding is recorded as an :class:`Issue` with a severity:
    * ``critical`` — almost certainly a real error that will hurt training
      (out-of-range class IDs, unparsable lines, duplicate/zero-size boxes,
      orphaned labels). The process exits non-zero if any exist, which is what
      makes this usable as a pre-training / CI gate.
    * ``warn``     — should be reviewed by a human but may be legitimate
      (tiny/huge boxes, boxes extending slightly outside the frame, images with
      no label file, extreme aspect ratios, heavy same-class overlaps).
    * ``info``     — purely informational, no action required (e.g. small boxes
      that are normal for distant objects in road scenes).

Catches:
    1. Invalid class IDs (out of range)
    2. Boxes outside [0, 1] normalised coordinates (clipping issues)
    3. Suspiciously tiny boxes (<0.001 of image, likely noise or mis-annotation)
    4. Suspiciously huge boxes (>0.9 of image, likely a misclick)
    5. Duplicate boxes within the same image (annotator double-clicked)
    6. Severely overlapping boxes of the same class (IoU>0.9)
    7. Images with no labels at all (might be missed annotations or true negatives)
    8. Aspect-ratio outliers (extreme stretching, likely error)
    9. Per-class statistics (instance counts, avg box area per class)
    10. Per-image stats (max boxes per image, instance distribution)

Usage:
    python tools/audit_annotations.py --dataset /workspace/data/road_anamolies_yolov8-phref-v6
    python tools/audit_annotations.py --dataset <path> --fix-easy     # auto-fix safe issues
    python tools/audit_annotations.py --dataset <path> --export problems.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Best-effort: don't crash if PIL isn't installed
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Repo root on sys.path so we can import src
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass
class Issue:
    """A single problem found in one annotation line (or whole file/image).

    One :class:`Issue` is the atomic unit of the audit report. Both the
    human-readable summary and the CSV export iterate over a flat list of these,
    so every check in the pipeline funnels its findings into this one shape. The
    fields are intentionally string/int primitives so the object serialises
    trivially to a CSV row.

    Attributes:
        severity: One of ``'critical'``, ``'warn'`` or ``'info'``. Drives both
            the summary counts and whether the process exits non-zero.
        image: Path to the offending image, or ``'-'`` when the issue is about a
            label file with no matching image (e.g. an orphan label).
        label_file: Path to the offending ``.txt`` label file, or ``'-'`` when
            the issue is about an image with no label.
        line_no: 1-based line number inside the label file, or ``0`` for
            file/image-level issues that have no specific line.
        kind: Short machine-readable issue category (e.g. ``'invalid_class'``,
            ``'tiny_box'``). Used to group counts in the summary and to decide
            which issues :func:`fix_easy` is allowed to touch.
        detail: Human-readable explanation, usually with the offending values.
    """

    severity: str   # 'critical' | 'warn' | 'info'
    image: str
    label_file: str
    line_no: int
    kind: str
    detail: str


@dataclass
class AuditReport:
    """Mutable accumulator that every audit function writes into.

    A single instance is created in :func:`main` and threaded through
    :func:`audit_split` for each split, then read by :func:`summarize`,
    :func:`export_csv` and :func:`fix_easy`. Keeping all state in one object (vs.
    returning tuples) means the per-split, per-image and per-line checks can all
    append into the same place without complex plumbing.

    Attributes:
        splits: Reserved/unused placeholder for future per-split metadata.
        issues: Flat list of every :class:`Issue` found across all splits.
        class_counts: ``{split_name: Counter({class_name: instance_count})}`` —
            powers the class-distribution table and the imbalance warning.
        area_stats: ``{class_name: [box_area, ...]}`` of normalised box areas
            (w*h), used for the per-class area statistics table.
        aspect_stats: ``{class_name: [aspect_ratio, ...]}`` collected for
            potential reporting (currently used to detect outliers inline).
        empty_images: Images that carried no usable annotations (missing or
            empty label file) — useful for spotting missed annotations.
        images_per_split: ``{split_name: image_count}`` for the summary header.
    """

    splits: dict = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)
    # defaultdict(Counter): first key is the split, second key is the class name,
    # so missing splits/classes auto-initialise to zero on first access.
    class_counts: dict = field(default_factory=lambda: defaultdict(Counter))
    area_stats: dict = field(default_factory=lambda: defaultdict(list))
    aspect_stats: dict = field(default_factory=lambda: defaultdict(list))
    empty_images: list = field(default_factory=list)
    images_per_split: dict = field(default_factory=dict)


def _iou_xywh(a, b):
    """Intersection-over-Union for two boxes in normalised (cx, cy, w, h) format.

    Used by the duplicate / heavy-overlap detector to decide whether two
    same-class boxes in one image are really the same annotation (annotator
    double-clicked) or just a genuine near-miss. YOLO labels store boxes as
    centre-x, centre-y, width, height (all normalised to [0, 1]), so this first
    converts each box to its corner coordinates before computing IoU.

    Args:
        a: First box as ``(cx, cy, w, h)``.
        b: Second box as ``(cx, cy, w, h)``.

    Returns:
        The IoU as a float in [0, 1]. Returns ``0.0`` when the boxes do not
        overlap (or when the union degenerates to zero area).
    """
    # Convert each centre-format box to (x1, y1)-(x2, y2) corner coordinates.
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2

    # Intersection rectangle: the overlap region is bounded by the inner edges.
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    # Clamp width/height at 0 so non-overlapping boxes give zero, not negative.
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    # Union = sum of both areas minus the double-counted intersection.
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def audit_split(
    split_dir: Path,
    class_names: list[str],
    report: AuditReport,
    split_name: str,
    check_image_sizes: bool = False,
):
    """Audit one dataset split (train / valid / test), accumulating into ``report``.

    This is the heart of the auditor. It expects the standard YOLO layout where a
    split directory contains parallel ``images/`` and ``labels/`` folders, with one
    ``<stem>.txt`` label per ``<stem>.<ext>`` image. It performs three tiers of
    checks: (1) cross-referencing image/label filename stems to find orphans and
    missing labels, (2) per-line parsing and geometric validation of every box,
    and (3) per-image cross-box duplicate / overlap detection.

    All findings are appended to ``report`` in place (the function returns nothing)
    so the caller can aggregate every split into one report object.

    Args:
        split_dir: Path to the split root (the folder holding ``images/`` and
            ``labels/``).
        class_names: Class names in YAML index order, used to validate class IDs
            and to map integer IDs to readable names for the statistics tables.
        report: The shared :class:`AuditReport` accumulator, mutated in place.
        split_name: Human-readable split label (``"train"``/``"valid"``/``"test"``)
            used as the key into per-split stats.
        check_image_sizes: If True (and PIL is available), open each image to
            confirm it is readable, flagging corrupt files. Slower, hence opt-in.
    """
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    # A split is unusable if either half of the image/label pair is missing;
    # flag it as critical and bail rather than producing misleading partial stats.
    if not images_dir.exists() or not labels_dir.exists():
        report.issues.append(Issue(
            severity="critical",
            image="-",
            label_file=str(split_dir),
            line_no=0,
            kind="missing_split",
            detail=f"Either images/ or labels/ missing in {split_dir}",
        ))
        return

    # Index both folders by filename stem so images and labels can be matched up.
    # Only known image extensions count as images; only .txt files count as labels.
    image_files = {p.stem: p for p in images_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}}
    label_files = {p.stem: p for p in labels_dir.iterdir() if p.suffix == ".txt"}
    report.images_per_split[split_name] = len(image_files)

    # Images without labels (could be legitimate empty backgrounds OR missed annotations)
    images_missing_labels = set(image_files) - set(label_files)
    for stem in images_missing_labels:
        report.issues.append(Issue(
            severity="warn",
            image=str(image_files[stem]),
            label_file="-",
            line_no=0,
            kind="image_no_label",
            detail="Image has no .txt label file (could be background or missed annotation)",
        ))
        report.empty_images.append(str(image_files[stem]))

    # Labels without images (orphans — definite error). Unlike a missing label
    # (which might be an intentional background), a label with no image can never
    # be trained on and signals a broken/incomplete export, so it is critical.
    labels_orphaned = set(label_files) - set(image_files)
    for stem in labels_orphaned:
        report.issues.append(Issue(
            severity="critical",
            image="-",
            label_file=str(label_files[stem]),
            line_no=0,
            kind="orphan_label",
            detail="Label file has no matching image",
        ))

    # Per-image, per-line checks. Iterate only over stems present in BOTH folders
    # (the intersection) — orphans and missing labels were already reported above.
    # Sorted for deterministic, reproducible report ordering.
    for stem in sorted(set(image_files) & set(label_files)):
        img_path = image_files[stem]
        lbl_path = label_files[stem]

        # Optional: read image dims (slower, but enables more checks).
        # Note: img_w/img_h are currently only used to verify readability; the
        # geometric checks below operate purely in normalised [0, 1] space.
        img_w, img_h = None, None
        if check_image_sizes and HAS_PIL:
            try:
                with Image.open(img_path) as im:
                    img_w, img_h = im.size
            except Exception as e:
                report.issues.append(Issue(
                    severity="critical",
                    image=str(img_path),
                    label_file=str(lbl_path),
                    line_no=0,
                    kind="corrupt_image",
                    detail=f"Cannot open: {e}",
                ))
                continue

        try:
            text = lbl_path.read_text()
        except Exception as e:
            report.issues.append(Issue(
                severity="critical",
                image=str(img_path),
                label_file=str(lbl_path),
                line_no=0,
                kind="unreadable_label",
                detail=str(e),
            ))
            continue

        # An empty label file is a valid "background" image in YOLO, but it is
        # worth a warning because it often means an annotation was missed.
        if not text.strip():
            report.issues.append(Issue(
                severity="warn",
                image=str(img_path),
                label_file=str(lbl_path),
                line_no=0,
                kind="empty_label",
                detail="Label file exists but is empty",
            ))
            report.empty_images.append(str(img_path))
            continue

        boxes_in_image = []  # collected per image so we can compare pairs for duplicates/overlap

        # 1-based line numbering matches what a human sees in a text editor.
        for line_no, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue  # tolerate blank lines within a label file

            # A valid YOLO line is exactly: "<class_id> <cx> <cy> <w> <h>".
            parts = line.split()
            if len(parts) != 5:
                report.issues.append(Issue(
                    severity="critical",
                    image=str(img_path),
                    label_file=str(lbl_path),
                    line_no=line_no,
                    kind="malformed_line",
                    detail=f"Expected 5 fields, got {len(parts)}: '{raw_line[:80]}'",
                ))
                continue

            # Class is an integer index; the four box coords are floats.
            try:
                cls_id = int(parts[0])
                cx, cy, w, h = (float(p) for p in parts[1:])
            except ValueError:
                report.issues.append(Issue(
                    severity="critical",
                    image=str(img_path),
                    label_file=str(lbl_path),
                    line_no=line_no,
                    kind="numeric_parse_error",
                    detail=f"Can't parse numeric values: '{raw_line[:80]}'",
                ))
                continue

            # 1. Class ID validity
            if cls_id < 0 or cls_id >= len(class_names):
                report.issues.append(Issue(
                    severity="critical",
                    image=str(img_path),
                    label_file=str(lbl_path),
                    line_no=line_no,
                    kind="invalid_class",
                    detail=f"class_id={cls_id} not in [0, {len(class_names)-1}]",
                ))
                continue

            # ID is valid -> resolve to a name and tally it for the distribution table.
            class_name = class_names[cls_id]
            report.class_counts[split_name][class_name] += 1

            # 2. Coordinates outside [0, 1] — YOLO coords are normalised, so any
            #    value beyond this range is a hard error (bad export or scaling bug).
            for fld, val in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
                if val < 0 or val > 1:
                    report.issues.append(Issue(
                        severity="critical",
                        image=str(img_path),
                        label_file=str(lbl_path),
                        line_no=line_no,
                        kind="oob_coordinate",
                        detail=f"{fld}={val:.4f} outside [0, 1]",
                    ))

            # 3. Box extending outside image. Even with an in-range centre, a wide
            #    box can spill past the frame edge. The 1e-6 epsilon tolerates
            #    floating-point rounding so boxes that touch the edge aren't flagged.
            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = cx + w / 2, cy + h / 2
            if x1 < -1e-6 or y1 < -1e-6 or x2 > 1 + 1e-6 or y2 > 1 + 1e-6:
                report.issues.append(Issue(
                    severity="warn",
                    image=str(img_path),
                    label_file=str(lbl_path),
                    line_no=line_no,
                    kind="box_extends_outside",
                    detail=f"Box ({x1:.3f},{y1:.3f})-({x2:.3f},{y2:.3f}) extends outside image",
                ))

            # 4. Box area sanity. Area here is the normalised fraction of the
            #    whole image (w and h are each in [0, 1], so w*h is the area
            #    fraction). Recorded for every box to build the per-class stats.
            area = w * h
            report.area_stats[class_name].append(area)
            # < 0.01% of the image is almost certainly noise or a stray click.
            if area < 0.0001:
                report.issues.append(Issue(
                    severity="warn",
                    image=str(img_path),
                    label_file=str(lbl_path),
                    line_no=line_no,
                    kind="tiny_box",
                    detail=f"Box area = {area*100:.4f}% of image (very small, may be noise)",
                ))
            elif area < 0.001:
                # Still small but typical for distant objects in road scenes — info-only
                report.issues.append(Issue(
                    severity="info",
                    image=str(img_path),
                    label_file=str(lbl_path),
                    line_no=line_no,
                    kind="small_box",
                    detail=f"Box area = {area*100:.3f}% (typical for distant objects)",
                ))

            # 5. Box area too large (> 90% of image — likely a misclick or a box
            #    accidentally drawn around the entire frame).
            if area > 0.9:
                report.issues.append(Issue(
                    severity="warn",
                    image=str(img_path),
                    label_file=str(lbl_path),
                    line_no=line_no,
                    kind="huge_box",
                    detail=f"Box covers {area*100:.1f}% of image (likely a misclick)",
                ))

            # 6. Aspect ratio outliers. max(w/h, h/w) gives the longer-to-shorter
            #    side ratio regardless of orientation, so 1.0 == square. Guarded by
            #    w>0 and h>0 to avoid division by zero (zero-size handled below).
            if w > 0 and h > 0:
                ar = max(w / h, h / w)
                report.aspect_stats[class_name].append(ar)
                if ar > 10:  # >10:1 stretch is implausible for road anomalies
                    report.issues.append(Issue(
                        severity="warn",
                        image=str(img_path),
                        label_file=str(lbl_path),
                        line_no=line_no,
                        kind="aspect_ratio_outlier",
                        detail=f"Aspect ratio = {ar:.1f}:1 (extremely stretched)",
                    ))

            # 7. Zero-size boxes — a degenerate box trains on nothing and can
            #    destabilise the loss; skip it (continue) so it never enters
            #    boxes_in_image for the overlap pass.
            if w <= 0 or h <= 0:
                report.issues.append(Issue(
                    severity="critical",
                    image=str(img_path),
                    label_file=str(lbl_path),
                    line_no=line_no,
                    kind="zero_size_box",
                    detail=f"w={w}, h={h} — box has zero or negative dimension",
                ))
                continue

            # Box passed all per-line checks: keep it for the pairwise overlap pass.
            boxes_in_image.append((cls_id, cx, cy, w, h, line_no))

        # 8. Duplicates / heavy overlaps within image. Compare every unique pair
        #    (j starts at i+1 to avoid self- and double-comparison). Unpacking
        #    `c, *box, l` splits each tuple into class id, the 4 box coords, and
        #    its line number so we can report exactly which lines collide.
        for i in range(len(boxes_in_image)):
            for j in range(i + 1, len(boxes_in_image)):
                c1, *box1, l1 = boxes_in_image[i]
                c2, *box2, l2 = boxes_in_image[j]
                if c1 != c2:
                    continue  # two different classes overlapping is legitimate
                iou = _iou_xywh(box1, box2)
                if iou > 0.95:  # near-identical -> almost certainly a duplicate click
                    report.issues.append(Issue(
                        severity="critical",
                        image=str(img_path),
                        label_file=str(lbl_path),
                        line_no=l1,
                        kind="duplicate_box",
                        detail=f"Lines {l1} & {l2} have IoU={iou:.3f} (duplicate annotation)",
                    ))
                elif iou > 0.7:  # heavy but not identical -> flag for human review
                    report.issues.append(Issue(
                        severity="warn",
                        image=str(img_path),
                        label_file=str(lbl_path),
                        line_no=l1,
                        kind="heavy_overlap",
                        detail=f"Lines {l1} & {l2} have IoU={iou:.3f} (heavy same-class overlap)",
                    ))


def summarize(report: AuditReport, class_names: list[str]) -> None:
    """Print a human-readable triage summary of a completed audit to stdout.

    This is the report a researcher actually reads at the console: severity
    counts, an issue-type breakdown, the per-split class distribution, an
    imbalance warning, per-class box-area statistics, and a preview of the first
    critical issues. It is purely a presentation layer — it reads from ``report``
    and prints, mutating nothing.

    Args:
        report: The populated :class:`AuditReport` to summarise.
        class_names: Class names in YAML order, so the distribution and area
            tables list every class (even ones with zero instances).
    """
    # Pre-aggregate counts by severity and by issue kind for the headline tables.
    sev_counts = Counter(i.severity for i in report.issues)
    kind_counts = Counter(i.kind for i in report.issues)

    print("\n" + "═" * 70)
    print("  ANNOTATION AUDIT SUMMARY")
    print("═" * 70)

    print(f"\n  Total images checked: {sum(report.images_per_split.values())}")
    for split, n in report.images_per_split.items():
        print(f"    {split:8s}: {n} images")

    print(f"\n  Issues found: {len(report.issues)}")
    print(f"    🔴 Critical: {sev_counts.get('critical', 0)} (must fix)")
    print(f"    🟡 Warn    : {sev_counts.get('warn', 0)} (should review)")
    print(f"    🔵 Info    : {sev_counts.get('info', 0)} (FYI, no action needed)")

    if kind_counts:
        print("\n  Issues by type:")
        # Sort descending by count (-x[1]) so the most common problems lead.
        for kind, n in sorted(kind_counts.items(), key=lambda x: -x[1]):
            print(f"    {kind:30s} {n}")

    print("\n  Class distribution:")
    print(f"  {'Class':10s} {'train':>10s} {'valid':>10s} {'test':>10s} {'TOTAL':>10s}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for cls in class_names:
        train_n = report.class_counts.get("train", {}).get(cls, 0)
        valid_n = report.class_counts.get("valid", {}).get(cls, 0)
        test_n = report.class_counts.get("test", {}).get(cls, 0)
        total = train_n + valid_n + test_n
        print(f"  {cls:10s} {train_n:>10} {valid_n:>10} {test_n:>10} {total:>10}")

    # Class imbalance flag. A skewed training set biases the detector toward the
    # majority class, so the max/min instance ratio is a quick health signal.
    train_counts = report.class_counts.get("train", {})
    if train_counts:
        max_n = max(train_counts.values())
        min_n = min(train_counts.values())
        # Guard against a class with zero instances (ratio would be undefined).
        ratio = max_n / min_n if min_n > 0 else float("inf")
        if ratio > 10:
            print(f"\n  ⚠️  Severe class imbalance in train: max/min ratio = {ratio:.1f}")
        elif ratio > 3:
            print(f"\n  ⚠️  Moderate class imbalance in train: max/min ratio = {ratio:.1f}")

    # Per-class box area stats
    print("\n  Box area statistics (% of image):")
    print(f"  {'Class':10s} {'mean':>10s} {'median':>10s} {'min':>10s} {'max':>10s}")
    for cls in class_names:
        areas = report.area_stats.get(cls, [])
        if not areas:
            continue
        # Multiply by 100 to present areas as a percentage of the image. Median
        # is the middle element of the sorted list (approximate for even counts).
        areas_sorted = sorted(areas)
        mean = sum(areas) / len(areas) * 100
        median = areas_sorted[len(areas_sorted) // 2] * 100
        amin = areas_sorted[0] * 100
        amax = areas_sorted[-1] * 100
        print(f"  {cls:10s} {mean:>10.3f} {median:>10.3f} {amin:>10.4f} {amax:>10.3f}")

    # Top 10 critical issues — a preview so the user can act without opening the
    # CSV; full detail is available via --export.
    crit = [i for i in report.issues if i.severity == "critical"]
    if crit:
        print("\n  First 10 critical issues:")
        for i in crit[:10]:
            print(f"    [{i.kind}] {Path(i.label_file).name}:{i.line_no} — {i.detail}")
        if len(crit) > 10:
            print(f"    ... and {len(crit) - 10} more (use --export for full list)")

    print("\n" + "═" * 70)


def export_csv(report: AuditReport, out_path: Path) -> None:
    """Write the full issue list to a CSV for offline triage / spreadsheets.

    The console summary only previews the first few critical issues; this dumps
    every :class:`Issue` (one row each) so problems can be sorted, filtered and
    fixed systematically. Parent directories are created as needed.

    Args:
        report: The populated audit report whose ``issues`` are written.
        out_path: Destination CSV path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" is the documented way to avoid blank rows from csv on Windows.
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["severity", "kind", "image", "label_file", "line_no", "detail"])
        for i in report.issues:
            w.writerow([i.severity, i.kind, i.image, i.label_file, i.line_no, i.detail])
    print(f"\n  → Full report: {out_path}")


def fix_easy(report: AuditReport) -> int:
    """Auto-fix the subset of issues that are safe to repair mechanically.

    Only three issue kinds are *reversible and unambiguous* enough to fix without
    human judgement, so those are the only ones touched:
      - ``oob_coordinate``: clip box coordinates to [0, 1] (fixes rounding spill)
      - ``duplicate_box``:  remove the duplicate line (keep the first occurrence)
      - ``zero_size_box``:  remove the degenerate, untrainable box

    It deliberately does NOT touch class IDs, missing labels, or any geometric
    "looks wrong" issue, because guessing the correct value there risks silently
    corrupting ground truth. Every modified file is backed up to ``<name>.bak``
    (once) before being rewritten, so the operation is recoverable.

    The function rebuilds each affected label file from scratch: lines flagged for
    removal are dropped, surviving coordinate lines are re-emitted with clamped,
    fixed-precision values, and anything it can't parse is passed through verbatim.

    Args:
        report: A completed audit report; only its ``issues`` list is consulted.

    Returns:
        The total number of individual fixes applied (removed lines + clamps).
    """
    fixes = 0
    # Group fixable issues by the file they belong to, so each file is rewritten
    # exactly once. Issues on images-without-labels (label_file == "-") are skipped.
    files_to_fix: dict[str, list[Issue]] = defaultdict(list)
    for i in report.issues:
        if i.kind in {"oob_coordinate", "duplicate_box", "zero_size_box"} and i.label_file != "-":
            files_to_fix[i.label_file].append(i)

    for label_file_str, issues in files_to_fix.items():
        label_file = Path(label_file_str)
        if not label_file.exists():
            continue  # file may have been moved/deleted since the audit ran
        lines = label_file.read_text().splitlines()
        new_lines = []
        seen_boxes = set()
        # Lines we will drop entirely (duplicates and zero-size boxes), by line number.
        skip_lines = {i.line_no for i in issues if i.kind in {"duplicate_box", "zero_size_box"}}

        for line_no, raw in enumerate(lines, start=1):
            if line_no in skip_lines:
                fixes += 1  # dropping a line counts as one fix
                continue
            parts = raw.strip().split()
            if len(parts) != 5:
                new_lines.append(raw)  # leave malformed lines untouched (not our job)
                continue
            try:
                cls = int(parts[0])
                vals = [float(p) for p in parts[1:]]
            except ValueError:
                new_lines.append(raw)  # unparsable numerics -> pass through unchanged
                continue
            # Clamp each coordinate into [0, 1]; nested min/max == hard clip.
            clamped = [max(0.0, min(1.0, v)) for v in vals]
            if clamped != vals:
                fixes += 1  # only count when clamping actually changed something
            # Re-emit with fixed 6-decimal precision for consistent formatting.
            new_lines.append(f"{cls} {clamped[0]:.6f} {clamped[1]:.6f} {clamped[2]:.6f} {clamped[3]:.6f}")

        # Backup the original before overwriting — but only once, so re-running
        # the fixer never clobbers the pristine first backup with already-fixed data.
        backup = label_file.with_suffix(label_file.suffix + ".bak")
        if not backup.exists():
            backup.write_text(label_file.read_text())
        label_file.write_text("\n".join(new_lines) + "\n")

    return fixes


def main():
    """CLI entry point: parse args, audit every split, report, and set exit code.

    Orchestrates the whole tool: validates the dataset path, runs
    :func:`audit_split` over the standard train/valid/test splits (whichever
    exist), prints the :func:`summarize` report, optionally exports a CSV and/or
    applies easy fixes, then exits non-zero if any critical issue was found. That
    non-zero exit is what lets the script act as a fail-fast gate in CI or a
    pre-training shell pipeline.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path,
                    help="Dataset root (contains train/, valid/, test/ subdirs)")
    ap.add_argument("--classes", nargs="+", default=["MH", "PH", "WLPH"],
                    help="Class names in YAML order (default: MH PH WLPH)")
    ap.add_argument("--check-image-sizes", action="store_true",
                    help="Open each image to check it's readable (slower)")
    ap.add_argument("--fix-easy", action="store_true",
                    help="Auto-fix safe issues (clamp coords, remove duplicates). Creates .bak files.")
    ap.add_argument("--export", type=Path, default=None,
                    help="Export full issue list to CSV at this path")
    args = ap.parse_args()

    if not args.dataset.exists():
        print(f"ERROR: dataset path does not exist: {args.dataset}", file=sys.stderr)
        sys.exit(2)

    # One shared report accumulates findings from every split that exists on disk.
    report = AuditReport()
    for split in ("train", "valid", "test"):
        split_dir = args.dataset / split
        if split_dir.exists():  # silently skip splits that aren't present
            print(f"Auditing {split}...")
            audit_split(split_dir, args.classes, report, split, args.check_image_sizes)

    summarize(report, args.classes)

    if args.export:
        export_csv(report, args.export)

    if args.fix_easy:
        print("\n  Applying easy fixes (creating .bak files)...")
        n_fixes = fix_easy(report)
        print(f"  → Fixed {n_fixes} issues. Originals backed up as .bak files.")

    # Exit non-zero if critical issues exist, so callers (CI / shell pipelines)
    # can block training on a dirty dataset. Warn/info never affect the exit code.
    n_critical = sum(1 for i in report.issues if i.severity == "critical")
    sys.exit(1 if n_critical > 0 else 0)


if __name__ == "__main__":
    main()
