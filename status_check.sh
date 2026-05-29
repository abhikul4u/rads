#!/bin/bash
# Post-sleep status check for RADS Layer 3 full training run
# Usage:  bash status_check.sh
# Safe to run repeatedly — only reads state, never modifies.

set -u  # error on undefined vars (but no -e — we want graceful errors)

# Colors (degrade gracefully if not a tty)
if [ -t 1 ]; then
    BLUE='\033[1;34m'; GREEN='\033[1;32m'; YELLOW='\033[1;33m'; RED='\033[1;31m'; RESET='\033[0m'
else
    BLUE=''; GREEN=''; YELLOW=''; RED=''; RESET=''
fi

EXPECTED_VARIANTS=("baseline" "cbam" "p2" "sizeaware" "combined" "distill")
EXPECTED_SEEDS=(42 1337 2024)
EXPECTED_TOTAL=$((${#EXPECTED_VARIANTS[@]} * ${#EXPECTED_SEEDS[@]}))
RUNS_DIR="/workspace/artifacts/runs"
RESULTS_DIR="/workspace/artifacts/results"

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BLUE}║         RADS Layer 3 — Post-Sleep Pipeline Status                ║${RESET}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${RESET}"
echo "Current time:   $(date)"
echo "Pod hostname:   $(hostname)"
echo ""

# --- 1. Pipeline activity ---
echo -e "${BLUE}─── 1. Pipeline Activity ────────────────────────────────────────${RESET}"

if tmux has-session -t rads-full 2>/dev/null; then
    echo -e "  Tmux session 'rads-full':       ${GREEN}✓ ALIVE${RESET}"
else
    echo -e "  Tmux session 'rads-full':       ${YELLOW}○ ENDED${RESET}  (training may be complete)"
fi

PIPELINE_PROCS=$(ps aux | grep "06_run_full_pipeline.sh" | grep -v grep | wc -l)
TRAINING_PROCS=$(ps aux | grep -E "scripts/0[1-4]_" | grep -v grep | wc -l)

if [ "$PIPELINE_PROCS" -gt 0 ]; then
    echo -e "  Pipeline orchestrator:           ${GREEN}✓ RUNNING${RESET}"
else
    echo -e "  Pipeline orchestrator:           ${YELLOW}○ NOT RUNNING${RESET}  (likely finished)"
fi

if [ "$TRAINING_PROCS" -gt 0 ]; then
    echo -e "  Training processes:              ${GREEN}✓ $TRAINING_PROCS active${RESET}"
    # Show which variant
    CURRENT=$(ps aux | grep -oE "scripts/0[1-4]_[a-z_]+\.py" | head -1 | sed 's|scripts/||; s|\.py||')
    echo -e "  Currently executing:             ${CURRENT:-unknown}"
else
    echo -e "  Training processes:              ${YELLOW}○ NONE${RESET}"
fi
echo ""

# --- 2. GPU & disk ---
echo -e "${BLUE}─── 2. Hardware Health ──────────────────────────────────────────${RESET}"

if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.free,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$GPU_INFO" ]; then
        IFS=',' read -r gpu_name gpu_util gpu_mem_used gpu_mem_free gpu_temp <<< "$GPU_INFO"
        gpu_util_int=$(echo $gpu_util | tr -d ' ')
        echo "  GPU:                            $gpu_name"
        if [ "$gpu_util_int" -gt 80 ]; then
            echo -e "  GPU utilization:                ${GREEN}${gpu_util}%${RESET}  (training active)"
        elif [ "$gpu_util_int" -gt 0 ]; then
            echo -e "  GPU utilization:                ${YELLOW}${gpu_util}%${RESET}  (between variants)"
        else
            echo -e "  GPU utilization:                ${YELLOW}${gpu_util}%${RESET}  (idle — done?)"
        fi
        echo "  GPU memory used:                ${gpu_mem_used} MiB"
        echo "  GPU memory free:                ${gpu_mem_free} MiB"
        echo "  GPU temperature:                ${gpu_temp}°C"
    fi
else
    echo -e "  ${RED}nvidia-smi not found${RESET}"
fi

DISK_OUTPUT=$(df -h /workspace 2>/dev/null | tail -1)
if [ -n "$DISK_OUTPUT" ]; then
    DISK_USED_PCT=$(echo "$DISK_OUTPUT" | awk '{print $5}' | tr -d '%')
    DISK_AVAIL=$(echo "$DISK_OUTPUT" | awk '{print $4}')
    DISK_TOTAL=$(echo "$DISK_OUTPUT" | awk '{print $2}')
    if [ "$DISK_USED_PCT" -lt 70 ]; then
        echo -e "  Disk on /workspace:             ${GREEN}${DISK_USED_PCT}% used${RESET}  ($DISK_AVAIL free of $DISK_TOTAL)"
    elif [ "$DISK_USED_PCT" -lt 85 ]; then
        echo -e "  Disk on /workspace:             ${YELLOW}${DISK_USED_PCT}% used${RESET}  ($DISK_AVAIL free of $DISK_TOTAL)"
    else
        echo -e "  Disk on /workspace:             ${RED}${DISK_USED_PCT}% used${RESET}  ($DISK_AVAIL free of $DISK_TOTAL) — WARNING"
    fi
fi
echo ""

# --- 3. Completed variants ---
echo -e "${BLUE}─── 3. Training Progress ────────────────────────────────────────${RESET}"

ALL_RUNS=($(ls "$RUNS_DIR" 2>/dev/null | sort))
COMPLETED_RUNS=0
IN_PROGRESS_RUNS=0

# A "completed" run has best.pt; in-progress has only partial state
for run in "${ALL_RUNS[@]}"; do
    if [ -f "$RUNS_DIR/$run/weights/best.pt" ]; then
        ((COMPLETED_RUNS++))
    else
        ((IN_PROGRESS_RUNS++))
    fi
done

PROGRESS_PCT=$(( COMPLETED_RUNS * 100 / EXPECTED_TOTAL ))
echo "  Expected total:                  $EXPECTED_TOTAL trainings"
echo "  Folders present:                 ${#ALL_RUNS[@]}"
echo -e "  Completed (has best.pt):         ${GREEN}$COMPLETED_RUNS${RESET}"
echo -e "  In progress (no best.pt yet):    ${YELLOW}$IN_PROGRESS_RUNS${RESET}"
echo "  Progress:                        $PROGRESS_PCT%"
echo ""

if [ ${#ALL_RUNS[@]} -gt 0 ]; then
    echo "  Per-variant completion:"
    for variant in "${EXPECTED_VARIANTS[@]}"; do
        done_seeds=()
        active_seed=""
        for seed in "${EXPECTED_SEEDS[@]}"; do
            run="${variant}_seed${seed}"
            if [ -f "$RUNS_DIR/$run/weights/best.pt" ]; then
                done_seeds+=("$seed")
            elif [ -d "$RUNS_DIR/$run" ]; then
                active_seed="$seed"
            fi
        done
        n_done=${#done_seeds[@]}
        if [ "$n_done" -eq 3 ]; then
            echo -e "    ${variant}:        ${GREEN}3/3 ✓ done${RESET}  (seeds: ${done_seeds[*]})"
        elif [ "$n_done" -gt 0 ] || [ -n "$active_seed" ]; then
            if [ -n "$active_seed" ]; then
                echo -e "    ${variant}:        ${YELLOW}${n_done}/3 done${RESET}, seed $active_seed in progress"
            else
                echo -e "    ${variant}:        ${YELLOW}${n_done}/3 done${RESET}"
            fi
        else
            echo -e "    ${variant}:        ○ not started"
        fi
    done
fi
echo ""

# --- 4. Test set results ---
echo -e "${BLUE}─── 4. Test Set Results ─────────────────────────────────────────${RESET}"

if [ -d "$RESULTS_DIR" ]; then
    RESULT_FILES=("$RESULTS_DIR"/*_test.json)
    if [ -f "${RESULT_FILES[0]}" ]; then
        echo "  $(ls $RESULTS_DIR/*_test.json 2>/dev/null | wc -l) test evaluations complete"
        echo ""
        printf "  %-25s %-9s %-12s %-9s %-9s\n" "Variant" "mAP50" "mAP50-95" "P" "R"
        printf "  %-25s %-9s %-12s %-9s %-9s\n" "-------" "-----" "--------" "-" "-"
        for f in "$RESULTS_DIR"/*_test.json; do
            [ -f "$f" ] || continue
            name=$(basename "$f" .json)
            python3 -c "
import json, sys
try:
    d = json.load(open('$f'))
    print(f'  {\"$name\":<25} {d.get(\"map50\", 0):<9.4f} {d.get(\"map50_95\", 0):<12.4f} {d.get(\"precision\", 0):<9.4f} {d.get(\"recall\", 0):<9.4f}')
except Exception as e:
    print(f'  $name: parse error ({e})', file=sys.stderr)
" 2>/dev/null || echo "  $name: (couldn't parse)"
        done
    else
        echo "  No test results yet — eval runs after each variant completes training"
    fi
else
    echo -e "  ${YELLOW}No results directory yet${RESET}"
fi
echo ""

# --- 5. Best.pt summary per variant (per-seed val mAP) ---
echo -e "${BLUE}─── 5. Per-Run Validation mAP (from training) ──────────────────${RESET}"

if [ ${#ALL_RUNS[@]} -gt 0 ]; then
    for run in "${ALL_RUNS[@]}"; do
        csv="$RUNS_DIR/$run/results.csv"
        if [ -f "$csv" ]; then
            # Find the row with highest val mAP50
            BEST=$(python3 -c "
import csv as c, sys
try:
    rows = list(c.DictReader(open('$csv')))
    if not rows:
        sys.exit()
    best = max(rows, key=lambda r: float(r.get('       metrics/mAP50(B)', r.get('metrics/mAP50(B)', 0) or 0) or 0))
    epoch = best.get('                  epoch', best.get('epoch', '?')).strip()
    map50 = float(best.get('       metrics/mAP50(B)', best.get('metrics/mAP50(B)', 0)) or 0)
    map5095 = float(best.get('    metrics/mAP50-95(B)', best.get('metrics/mAP50-95(B)', 0)) or 0)
    last_epoch = rows[-1].get('                  epoch', rows[-1].get('epoch', '?')).strip()
    print(f'  {\"$run\":<25} best epoch {epoch:<3} mAP50={map50:.4f} mAP50-95={map5095:.4f}  (trained through epoch {last_epoch})')
except Exception as e:
    print(f'  $run: parse error', file=sys.stderr)
" 2>/dev/null)
            echo "$BEST"
        fi
    done
else
    echo "  No run folders yet."
fi
echo ""

# --- 6. Recent log activity ---
echo -e "${BLUE}─── 6. Recent Log Activity ──────────────────────────────────────${RESET}"
LOG_FILE="/workspace/artifacts/full_run.log"
if [ -f "$LOG_FILE" ]; then
    LOG_SIZE=$(du -h "$LOG_FILE" | cut -f1)
    LOG_MTIME=$(stat -c %y "$LOG_FILE" | cut -d. -f1)
    echo "  Log file:                       $LOG_FILE ($LOG_SIZE)"
    echo "  Last modified:                  $LOG_MTIME"
    SECONDS_SINCE=$(( $(date +%s) - $(stat -c %Y "$LOG_FILE") ))
    if [ "$SECONDS_SINCE" -lt 300 ]; then
        echo -e "  Status:                         ${GREEN}✓ Active (updated ${SECONDS_SINCE}s ago)${RESET}"
    elif [ "$SECONDS_SINCE" -lt 1800 ]; then
        echo -e "  Status:                         ${YELLOW}○ Quiet (updated ${SECONDS_SINCE}s ago — between variants?)${RESET}"
    else
        echo -e "  Status:                         ${YELLOW}○ Idle ($(($SECONDS_SINCE / 60)) min since last write)${RESET}"
    fi
    echo ""
    echo "  Last 5 meaningful log lines:"
    tail -200 "$LOG_FILE" 2>/dev/null | grep -E "(Epoch|mAP50|Validating|Results saved|task=detect)" | tail -5 | sed 's/^/    /'
else
    echo -e "  ${YELLOW}Log file not found: $LOG_FILE${RESET}"
fi
echo ""

# --- 7. Summary verdict ---
echo -e "${BLUE}─── 7. Verdict ──────────────────────────────────────────────────${RESET}"

if [ "$COMPLETED_RUNS" -eq "$EXPECTED_TOTAL" ]; then
    echo -e "  ${GREEN}🎉 FULL RUN COMPLETE!${RESET} All $EXPECTED_TOTAL trainings done."
    echo "     Next step: triage results, fix TFLite, build thesis plots."
elif [ "$TRAINING_PROCS" -gt 0 ]; then
    echo -e "  ${GREEN}✓ Pipeline is healthy and progressing${RESET}"
    echo "     $COMPLETED_RUNS/$EXPECTED_TOTAL trainings complete ($PROGRESS_PCT%)"
    if [ "$IN_PROGRESS_RUNS" -gt 0 ]; then
        echo "     Currently training one variant; let it continue."
    fi
elif [ "$COMPLETED_RUNS" -gt 0 ] && [ "$TRAINING_PROCS" -eq 0 ]; then
    echo -e "  ${YELLOW}⚠ Pipeline appears stopped${RESET} but only $COMPLETED_RUNS/$EXPECTED_TOTAL done."
    echo "     Possible causes: crashed mid-run, OOM, or quantize stage failure."
    echo "     Check: tail -100 $LOG_FILE"
    echo "     Or:    tmux attach -t rads-full"
else
    echo -e "  ${RED}? Unexpected state${RESET} — investigate manually."
fi
echo ""
echo -e "${BLUE}══════════════════════════════════════════════════════════════════${RESET}"
echo "Quick actions:"
echo "  tmux attach -t rads-full   # see live training (Ctrl+B D to detach)"
echo "  tail -f $LOG_FILE          # follow log live (Ctrl+C to exit)"
echo "  bash $0                    # re-run this status check"
echo "  W&B dashboard:  https://wandb.ai/abhikul4u-hvpm-college-of-engineering-technology-amravati"
echo -e "${BLUE}══════════════════════════════════════════════════════════════════${RESET}"
