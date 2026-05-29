#!/bin/bash
# collect_run_figures.sh — gather auto-generated PNGs from each training run
#
# Each Ultralytics run folder contains:
#   - confusion_matrix.png         (raw counts)
#   - confusion_matrix_normalized.png (proportions, better for thesis)
#   - PR_curve.png                 (precision-recall curve)
#   - F1_curve.png                 (F1 vs confidence)
#   - P_curve.png                  (precision vs confidence)
#   - R_curve.png                  (recall vs confidence)
#   - results.png                  (Ultralytics' own loss/mAP composite plot)
#   - labels.jpg, labels_correlogram.jpg  (dataset visualization)
#
# This script copies them into a clean structure under
# /workspace/thesis_run_figures/ for easy download and thesis embedding.
#
# Usage: bash collect_run_figures.sh

set -u

if [ -t 1 ]; then
    GREEN='\033[1;32m'; YELLOW='\033[1;33m'; BLUE='\033[1;34m'; RESET='\033[0m'
else
    GREEN=''; YELLOW=''; BLUE=''; RESET=''
fi

RUNS_DIR="/workspace/artifacts/runs"
OUT_DIR="/workspace/thesis_run_figures"
VARIANTS=("baseline" "cbam" "p2" "sizeaware" "combined" "distill")
SEEDS=(42 1337 2024)

# Figures to collect (filename → output subdir name)
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
for type_dir in "${FIGURE_TYPES[@]}"; do
    mkdir -p "$OUT_DIR/$type_dir"
done

# Also create a "thesis_picks" folder with just the seed=42 versions for primary thesis figures
mkdir -p "$OUT_DIR/thesis_picks/confusion_matrices"
mkdir -p "$OUT_DIR/thesis_picks/PR_curves"
mkdir -p "$OUT_DIR/thesis_picks/F1_curves"

echo -e "${BLUE}─── Copying figures ─────────────────────────────────────${RESET}"
total_copied=0

for variant in "${VARIANTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run="${variant}_seed${seed}"
        run_dir="$RUNS_DIR/$run"

        if [ ! -d "$run_dir" ]; then
            echo -e "  ${YELLOW}skip $run (not found)${RESET}"
            continue
        fi

        copied_this_run=0
        for fig_file in "${!FIGURE_TYPES[@]}"; do
            src="$run_dir/$fig_file"
            type_dir="${FIGURE_TYPES[$fig_file]}"

            if [ -f "$src" ]; then
                # Copy with variant_seedN prefix
                ext="${fig_file##*.}"
                base="${fig_file%.*}"
                out_name="${run}_${base}.${ext}"
                cp "$src" "$OUT_DIR/$type_dir/$out_name"
                copied_this_run=$((copied_this_run + 1))
                total_copied=$((total_copied + 1))

                # If this is seed 42 and a key thesis figure, also copy to thesis_picks
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

        if [ "$copied_this_run" -gt 0 ]; then
            echo -e "  ${GREEN}+${RESET} $run ($copied_this_run files)"
        else
            echo -e "  ${YELLOW}-${RESET} $run (no figures found)"
        fi
    done
done

echo ""
echo -e "${BLUE}─── Summary ─────────────────────────────────────────────${RESET}"
echo "  Total files copied: $total_copied"
echo ""
echo "  Per-type counts:"
for type_dir in "${FIGURE_TYPES[@]}"; do
    count=$(ls "$OUT_DIR/$type_dir" 2>/dev/null | wc -l)
    size=$(du -sh "$OUT_DIR/$type_dir" 2>/dev/null | cut -f1)
    printf "    %-30s %3d files  (%s)\n" "$type_dir" "$count" "$size"
done

echo ""
echo "  Thesis picks (seed 42 only — for direct thesis embedding):"
for sub in confusion_matrices PR_curves F1_curves; do
    count=$(ls "$OUT_DIR/thesis_picks/$sub" 2>/dev/null | wc -l)
    printf "    %-30s %3d files\n" "thesis_picks/$sub" "$count"
done

total_size=$(du -sh "$OUT_DIR" 2>/dev/null | cut -f1)
echo ""
echo -e "${GREEN}  Total size: $total_size${RESET}"
echo ""

echo -e "${BLUE}─── Next steps ──────────────────────────────────────────${RESET}"
echo "  1. Tar it up:"
echo "       tar -czf /workspace/thesis_run_figures.tar.gz -C /workspace thesis_run_figures"
echo "  2. Download via Jupyter Lab → right-click thesis_run_figures.tar.gz → Download"
echo "  3. On laptop, extract into repo:"
echo "       cd C:\\D-Drive\\Abhikul\\FinalProject\\radsTraining"
echo "       tar -xzf <downloaded>.tar.gz"
echo ""
