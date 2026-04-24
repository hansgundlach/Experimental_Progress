#!/bin/bash

# Sync offline wandb runs for this project.
#
# Runs are written as `./wandb/offline-run-*` relative to the script's
# current working directory. In this repo, offline runs end up in THREE
# possible places:
#   - ./wandb                        (transformer runs launched via experiments.py)
#   - LSTM_model/wandb               (LSTM runs launched via lstm_experiments.py)
#   - learning_rate_analysis/wandb   (LR sweep runs from run_lr_sweep.py)
#
# Usage:
#   ./sync_wandb_runs.sh                # sync 5 most recent runs across ALL dirs
#   ./sync_wandb_runs.sh 20             # sync 20 most recent runs across ALL dirs
#   ./sync_wandb_runs.sh 20 lstm        # sync 20 most recent runs from LSTM_model/wandb only
#   ./sync_wandb_runs.sh 20 root        # sync 20 most recent from ./wandb only
#   ./sync_wandb_runs.sh 20 lr          # sync 20 most recent from learning_rate_analysis/wandb only
#   ./sync_wandb_runs.sh all            # sync EVERY offline run across all dirs
#
# After syncing, runs appear at https://wandb.ai/<your-entity>/<project>
# where <project> is the experiment folder name (e.g. "x2_lstm_layer2").

N_RUNS=${1:-5}
WHICH=${2:-all}

# Resolve repo root (directory containing this script)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pick which wandb directories to scan
case "$WHICH" in
    root)  DIRS=("$REPO_ROOT/wandb") ;;
    lstm)  DIRS=("$REPO_ROOT/LSTM_model/wandb") ;;
    lr)    DIRS=("$REPO_ROOT/learning_rate_analysis/wandb") ;;
    all)   DIRS=("$REPO_ROOT/wandb" "$REPO_ROOT/LSTM_model/wandb" "$REPO_ROOT/learning_rate_analysis/wandb") ;;
    *)
        echo "Unknown selector: $WHICH. Use root|lstm|lr|all." >&2
        exit 1
        ;;
esac

# Gather candidate offline runs across all selected dirs, newest first.
#
# We sort by the mtime of the inner `run-*.wandb` file rather than the run
# directory's own mtime. While a run is actively training, wandb appends to
# `run-*.wandb` every few seconds, but the directory's mtime does not update
# (subdirs are created once at init and not touched again). Meanwhile,
# unrelated tools (rsync, backups, permission changes) can bump the mtime
# on OLD run dirs, making them appear "newer" than live runs and pushing
# them to the top of the sort. Using the `.wandb` file mtime avoids that.
tmp=$(mktemp)
for d in "${DIRS[@]}"; do
    [ -d "$d" ] || continue
    # Prefer the run-*.wandb mtime; fall back to the dir mtime if somehow absent.
    for run_dir in "$d"/offline-run-*; do
        [ -d "$run_dir" ] || continue
        wandb_file=$(find "$run_dir" -maxdepth 1 -name "run-*.wandb" -print -quit)
        if [ -n "$wandb_file" ]; then
            ts=$(stat -c '%Y' "$wandb_file")
        else
            ts=$(stat -c '%Y' "$run_dir")
        fi
        printf '%s %s\n' "$ts" "$run_dir" >> "$tmp"
    done
done

if [ ! -s "$tmp" ]; then
    echo "No offline runs found in:"
    printf '   %s\n' "${DIRS[@]}"
    rm -f "$tmp"
    exit 1
fi

if [ "$N_RUNS" = "all" ]; then
    recent_runs=$(sort -rn "$tmp" | cut -d' ' -f2-)
else
    if ! [[ "$N_RUNS" =~ ^[0-9]+$ ]] || [ "$N_RUNS" -lt 1 ]; then
        echo "First arg must be a positive integer or 'all'. Got: $N_RUNS" >&2
        rm -f "$tmp"
        exit 1
    fi
    recent_runs=$(sort -rn "$tmp" | head -"$N_RUNS" | cut -d' ' -f2-)
fi
rm -f "$tmp"

total=$(echo "$recent_runs" | wc -l)
echo "Syncing $total run(s) from: ${DIRS[*]}"
echo

count=0
success=0
while IFS= read -r run_dir; do
    [ -z "$run_dir" ] && continue
    count=$((count + 1))
    run_name=$(basename "$run_dir")
    parent=$(dirname "$run_dir")

    # Try to surface the project name from config.yaml for visibility
    project=""
    if [ -f "$run_dir/files/config.yaml" ]; then
        project=$(grep -m1 "project:" "$run_dir/files/config.yaml" | sed 's/.*project:[[:space:]]*//' | tr -d '"'"'")
    fi

    echo "[$count/$total] $run_name  (in $parent)"
    [ -n "$project" ] && echo "    project: $project"

    # wandb sync accepts a path; run from the parent so relative paths resolve cleanly
    if (cd "$parent" && wandb sync "$run_name"); then
        success=$((success + 1))
    else
        echo "    sync failed"
    fi
    echo
done <<< "$recent_runs"

echo "Done: $success/$count synced."
