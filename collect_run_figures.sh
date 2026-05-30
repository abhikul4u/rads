#!/bin/bash
# =============================================================================
# RADS Layer 3 — Model Development Pipeline
# collect_run_figures.sh — gather per-run figures into a thesis-ready tree
#
# Author: Rutuja Kulkarni
#
# WHAT THIS SCRIPT DOES
#   Each Ultralytics training run writes a set of diagnostic figures into its
#   run folder. This script harvests those PNG/JPG figures from every
#   variant×seed run and reorganises them into a clean, downloadable tree under
#   /workspace/thesis_run_figures/ — grouped by figure TYPE rather than by run —
#   so they can be embedded directly into the thesis.
#
#   Each Ultralytics run folder contains:
#   - confusion_matrix.png         (raw counts)
#   - confusion_matrix_normalized.png (proportions, better for thesis)
#   - PR_curve.png                 (precision-recall curve)
#   - F1_curve.png                 (F1 vs confidence)
#   - P_curve.png                  (precision vs confidence)
#   - R_curve.png                  (recall vs confidence)
#   - results.png                  (Ultralytics' own loss/mAP composite plot)
#   - labels.jpg, labels_correlogram.jpg  (dataset visualization)
#
#   In addition to the by-type folders, it builds a "thesis_picks" folder
#   containing only the seed=42 versions of the most important figures
#   (confusion matrices, PR curves, F1 curves) — the canonical figures meant for
#   direct inclusion in the write-up.
#
# WHEN TO RUN IT
#   After trainings have produced figures (during or after the full pipeline).
#   Read-only with respect to the source runs — it only copies files out.
#
# ARGUMENTS / ENV
#   None. Paths and the variant/seed matrix are hard-coded to the standard
#   /workspace/artifacts/runs layout.
#
# HOW IT FITS THE PIPELINE
#   The figures-only companion to backup_artifacts.sh: it curates plots for the
#   thesis, whereas the backup script preserves the full artifact tree.
#
# Usage: bash collect_run_figures.sh
# =============================================================================

# -u catches typos in variable names; no -e because missing per-run figures are
# expected and handled by explicit `if [ -f ... ]` checks.
set -u

# ANSI colours only when attached to a terminal; blanked otherwise.
if [ -t 1 ]; then
    GREEN='\033[1;32m'; YELLOW='\033[1;33m'; BLUE='\033[1;34m'; RESET='\033[0m'
else
    GREEN=''; YELLOW=''; BLUE=''; RESET=''
fi

# Source (Ultralytics run outputs), destination (curated tree), and the full
# variant×seed matrix to iterate over.
RUNS_DIR="/workspace/artifacts/runs"
OUT_DIR="/workspace/thesis_run_figures"
VARIANTS=("baseline" "cbam" "p2" "sizeaware" "combined" "distill")
SEEDS=(42 1337 2024)

# Figures to collect (filename → output subdir name)
# Associative array mapping each source figure filename to the by-type
# subdirectory it should be collected into under OUT_DIR.
declare -A FIGURE_TYPES=(
    ["confusion_matrix_normalized.png"]="confusion_matrix_normalized"
    ["confusion_matrix.png"]="confusion_matrix"
    ["PR_curve.png"]="PR_curve"
    ["F1_curve.png"]="F1_curve"
    ["P_curve.png"]="P_curve"
    ["R_curve.png"]="R_curve"
    ["results.png"]="results_composite"
    ["labels.jpg"]="labels_visualization"
    ["labels_correlogram.jpg"]="labels_correlogram"
)

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BLUE}║       Collect Ultralytics Per-Run Figures                        ║${RESET}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${RESET}"
echo ""
echo "  Source:  $RUNS_DIR"
echo "  Output:  $OUT_DIR"
echo ""

mkdir -p "$OUT_DIR"

# Create per-type subdirs
# One destination folder per figure type (the array values), so all confusion
# matrices land together, all PR curves together, etc.
for type_dir in "${FIGURE_TYPES[@]}"; do
    mkdir -p "$OUT_DIR/$type_dir"
done

# Also create a "thesis_picks" folder with just the seed=42 versions for primary thesis figures
# seed=42 is the canonical seed; these three figure types are the ones embedded
# directly in the thesis, so they get their own curated subtree.
mkdir -p "$OUT_DIR/thesis_picks/confusion_matrices"
mkdir -p "$OUT_DIR/thesis_picks/PR_curves"
mkdir -p "$OUT_DIR/thesis_picks/F1_curves"

# Iterate every run in the matrix and copy whichever figures it actually produced.
echo -e "${BLUE}─── Copying figures ─────────────────────────────────────${RESET}"
total_copied=0

for variant in "${VARIANTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run="${variant}_seed${seed}"
        run_dir="$RUNS_DIR/$run"

        # Skip runs that don't exist yet (matrix not fully trained).
        if [ ! -d "$run_dir" ]; then
            echo -e "  ${YELLOW}skip $run (not found)${RESET}"
            continue
        fi

        copied_this_run=0
        # Walk the figure-type keys; copy each one that exists in this run.
        for fig_file in "${!FIGURE_TYPES[@]}"; do
            src="$run_dir/$fig_file"
            type_dir="${FIGURE_TYPES[$fig_file]}"

            if [ -f "$src" ]; then
                # Copy with variant_seedN prefix
                # Rebuild the name as <run>_<base>.<ext> so files from different
                # runs don't collide once flattened into one by-type folder.
                ext="${fig_file##*.}"     # extension after the last dot
                base="${fig_file%.*}"     # filename without the extension
                out_name="${run}_${base}.${ext}"
                cp "$src" "$OUT_DIR/$type_dir/$out_name"
                copied_this_run=$((copied_this_run + 1))
                total_copied=$((total_copied + 1))

                # If this is seed 42 and a key thesis figure, also copy to thesis_picks
                # Drop a second copy under thesis_picks using a short, paper-friendly
                # name (variant only — seed is implied as 42).
                if [ "$seed" -eq 42 ]; then
                    case "$fig_file" in
                        "confusion_matrix_normalized.png")
                            cp "$src" "$OUT_DIR/thesis_picks/confusion_matrices/${variant}_cm_normalized.png"
                            ;;
                        "PR_curve.png")
                            cp "$src" "$OUT_DIR/thesis_picks/PR_curves/${variant}_PR.png"
                            ;;
                        "F1_curve.png")
                            cp "$src" "$OUT_DIR/thesis_picks/F1_curves/${variant}_F1.png"
                            ;;
                    esac
                fi
            fi
        done

        # Per-run line: green + with a count if anything was copied, else a
        # yellow - to flag a run dir that had none of the expected figures.
        if [ "$copied_this_run" -gt 0 ]; then
            echo -e "  ${GREEN}+${RESET} $run ($copied_this_run files)"
        else
            echo -e "  ${YELLOW}-${RESET} $run (no figures found)"
        fi
    done
done

# Summary: total copied, plus per-type and thesis-picks counts/sizes so the
# operator can confirm the harvest looks complete before downloading.
echo ""
echo -e "${BLUE}─── Summary ─────────────────────────────────────────────${RESET}"
echo "  Total files copied: $total_copied"
echo ""
echo "  Per-type counts:"
# Report how many figures (and how much disk) ended up in each by-type folder.
for type_dir in "${FIGURE_TYPES[@]}"; do
    count=$(ls "$OUT_DIR/$type_dir" 2>/dev/null | wc -l)
    size=$(du -sh "$OUT_DIR/$type_dir" 2>/dev/null | cut -f1)
    printf "    %-30s %3d files  (%s)\n" "$type_dir" "$count" "$size"
done

echo ""
echo "  Thesis picks (seed 42 only — for direct thesis embedding):"
# Counts for the curated seed-42 subset.
for sub in confusion_matrices PR_curves F1_curves; do
    count=$(ls "$OUT_DIR/thesis_picks/$sub" 2>/dev/null | wc -l)
    printf "    %-30s %3d files\n" "thesis_picks/$sub" "$count"
done

total_size=$(du -sh "$OUT_DIR" 2>/dev/null | cut -f1)
echo ""
echo -e "${GREEN}  Total size: $total_size${RESET}"
echo ""

# Operator guidance: how to tarball, download, and unpack the curated figures.
echo -e "${BLUE}─── Next steps ──────────────────────────────────────────${RESET}"
echo "  1. Tar it up:"
echo "       tar -czf /workspace/thesis_run_figures.tar.gz -C /workspace thesis_run_figures"
echo "  2. Download via Jupyter Lab → right-click thesis_run_figures.tar.gz → Download"
echo "  3. On laptop, extract into repo:"
echo "       cd C:\\D-Drive\\Abhikul\\FinalProject\\radsTraining"
echo "       tar -xzf <downloaded>.tar.gz"
echo ""
