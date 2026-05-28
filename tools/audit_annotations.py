#!/usr/bin/env python
"""Annotation quality audit for YOLO-format datasets.

Runs cheap automated checks and produces a triage report. No model needed.
Designed for the RADS 3-class dataset (MH, PH, WLPH).

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
    severity: str   # 'critical' | 'warn' | 'info'
    image: str
    label_file: str
    line_no: int
    kind: str
    detail: str


@dataclass
class AuditReport:
    splits: dict = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)
    class_counts: dict = field(default_factory=lambda: defaultdict(Counter))
    area_stats: dict = field(default_factory=lambda: defaultdict(list))
    aspect_stats: dict = field(default_factory=lambda: defaultdict(list))
    empty_images: list = field(default_factory=list)
    images_per_split: dict = field(default_factory=dict)


def _iou_xywh(a, b):
    """IoU for two boxes in normalised (cx, cy, w, h) format."""
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def audit_split(
    split_dir: Path,
    class_names: list[str],
    report: AuditReport,
    split_name: str,
    check_image_sizes: bool = False,
):
    """Audit one split (train/valid/test). Modifies `report` in place."""
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

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

    # Labels without images (orphans — definite error)
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

    # Per-image, per-line checks
    for stem in sorted(set(image_files) & set(label_files)):
        img_path = image_files[stem]
        lbl_path = label_files[stem]

        # Optional: read image dims (slower, but enables more checks)
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

        boxes_in_image = []  # for duplicate/overlap detection

        for line_no, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue

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

            class_name = class_names[cls_id]
            report.class_counts[split_name][class_name] += 1

            # 2. Coordinates outside [0, 1]
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

            # 3. Box extending outside image (cx+w/2 > 1 or cx-w/2 < 0 etc.)
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

            # 4. Box area too small (< 0.001 of image area = 0.1%)
            area = w * h
            report.area_stats[class_name].append(area)
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

            # 5. Box area too large (> 90% of image — likely error)
            if area > 0.9:
                report.issues.append(Issue(
                    severity="warn",
                    image=str(img_path),
                    label_file=str(lbl_path),
                    line_no=line_no,
                    kind="huge_box",
                    detail=f"Box covers {area*100:.1f}% of image (likely a misclick)",
                ))

            # 6. Aspect ratio outliers
            if w > 0 and h > 0:
                ar = max(w / h, h / w)
                report.aspect_stats[class_name].append(ar)
                if ar > 10:
                    report.issues.append(Issue(
                        severity="warn",
                        image=str(img_path),
                        label_file=str(lbl_path),
                        line_no=line_no,
                        kind="aspect_ratio_outlier",
                        detail=f"Aspect ratio = {ar:.1f}:1 (extremely stretched)",
                    ))

            # 7. Zero-size boxes
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

            boxes_in_image.append((cls_id, cx, cy, w, h, line_no))

        # 8. Duplicates / heavy overlaps within image
        for i in range(len(boxes_in_image)):
            for j in range(i + 1, len(boxes_in_image)):
                c1, *box1, l1 = boxes_in_image[i]
                c2, *box2, l2 = boxes_in_image[j]
                if c1 != c2:
                    continue  # different classes overlapping is fine
                iou = _iou_xywh(box1, box2)
                if iou > 0.95:
                    report.issues.append(Issue(
                        severity="critical",
                        image=str(img_path),
                        label_file=str(lbl_path),
                        line_no=l1,
                        kind="duplicate_box",
                        detail=f"Lines {l1} & {l2} have IoU={iou:.3f} (duplicate annotation)",
                    ))
                elif iou > 0.7:
                    report.issues.append(Issue(
                        severity="warn",
                        image=str(img_path),
                        label_file=str(lbl_path),
                        line_no=l1,
                        kind="heavy_overlap",
                        detail=f"Lines {l1} & {l2} have IoU={iou:.3f} (heavy same-class overlap)",
                    ))


def summarize(report: AuditReport, class_names: list[str]) -> None:
    """Print a human-readable summary of the audit."""
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

    # Class imbalance flag
    train_counts = report.class_counts.get("train", {})
    if train_counts:
        max_n = max(train_counts.values())
        min_n = min(train_counts.values())
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
        areas_sorted = sorted(areas)
        mean = sum(areas) / len(areas) * 100
        median = areas_sorted[len(areas_sorted) // 2] * 100
        amin = areas_sorted[0] * 100
        amax = areas_sorted[-1] * 100
        print(f"  {cls:10s} {mean:>10.3f} {median:>10.3f} {amin:>10.4f} {amax:>10.3f}")

    # Top 10 critical issues
    crit = [i for i in report.issues if i.severity == "critical"]
    if crit:
        print("\n  First 10 critical issues:")
        for i in crit[:10]:
            print(f"    [{i.kind}] {Path(i.label_file).name}:{i.line_no} — {i.detail}")
        if len(crit) > 10:
            print(f"    ... and {len(crit) - 10} more (use --export for full list)")

    print("\n" + "═" * 70)


def export_csv(report: AuditReport, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["severity", "kind", "image", "label_file", "line_no", "detail"])
        for i in report.issues:
            w.writerow([i.severity, i.kind, i.image, i.label_file, i.line_no, i.detail])
    print(f"\n  → Full report: {out_path}")


def fix_easy(report: AuditReport) -> int:
    """Auto-fix issues we can fix safely:
      - Clip box coordinates to [0, 1] if they're slightly outside (rounding)
      - Remove duplicate boxes (keep first occurrence)
      - Remove zero-size boxes

    Does NOT fix: class IDs, missing labels, oriented errors.
    Returns count of fixes applied.
    """
    fixes = 0
    files_to_fix: dict[str, list[Issue]] = defaultdict(list)
    for i in report.issues:
        if i.kind in {"oob_coordinate", "duplicate_box", "zero_size_box"} and i.label_file != "-":
            files_to_fix[i.label_file].append(i)

    for label_file_str, issues in files_to_fix.items():
        label_file = Path(label_file_str)
        if not label_file.exists():
            continue
        lines = label_file.read_text().splitlines()
        new_lines = []
        seen_boxes = set()
        skip_lines = {i.line_no for i in issues if i.kind in {"duplicate_box", "zero_size_box"}}

        for line_no, raw in enumerate(lines, start=1):
            if line_no in skip_lines:
                fixes += 1
                continue
            parts = raw.strip().split()
            if len(parts) != 5:
                new_lines.append(raw)
                continue
            try:
                cls = int(parts[0])
                vals = [float(p) for p in parts[1:]]
            except ValueError:
                new_lines.append(raw)
                continue
            # Clamp to [0, 1] with epsilon
            clamped = [max(0.0, min(1.0, v)) for v in vals]
            if clamped != vals:
                fixes += 1
            new_lines.append(f"{cls} {clamped[0]:.6f} {clamped[1]:.6f} {clamped[2]:.6f} {clamped[3]:.6f}")

        # Backup original
        backup = label_file.with_suffix(label_file.suffix + ".bak")
        if not backup.exists():
            backup.write_text(label_file.read_text())
        label_file.write_text("\n".join(new_lines) + "\n")

    return fixes


def main():
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

    report = AuditReport()
    for split in ("train", "valid", "test"):
        split_dir = args.dataset / split
        if split_dir.exists():
            print(f"Auditing {split}...")
            audit_split(split_dir, args.classes, report, split, args.check_image_sizes)

    summarize(report, args.classes)

    if args.export:
        export_csv(report, args.export)

    if args.fix_easy:
        print("\n  Applying easy fixes (creating .bak files)...")
        n_fixes = fix_easy(report)
        print(f"  → Fixed {n_fixes} issues. Originals backed up as .bak files.")

    # Exit non-zero if critical issues exist
    n_critical = sum(1 for i in report.issues if i.severity == "critical")
    sys.exit(1 if n_critical > 0 else 0)


if __name__ == "__main__":
    main()
