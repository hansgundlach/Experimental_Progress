#!/bin/bash
#
# Submit an LR sweep for a given experiment group.
#
# Usage:
#   bash submit_lr.sh modern_transformer
#   bash submit_lr.sh sgd_transformer
#   bash submit_lr.sh --list                  # list available groups
#   bash submit_lr.sh modern_transformer -c 4 # limit to 4 concurrent GPUs
#
# What happens:
#   1. Submits a SLURM array job (1 GPU per task, up to 8 concurrent)
#      that runs all (hidden_dim, lr) combos for the group.
#   2. Submits a dependency job that runs analyze_lr.py after all
#      sweep tasks finish — produces plots and the fitted LR table
#      automatically.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONCURRENT_JOBS=8  # max GPUs to use simultaneously (you have 8 V100s)

# ---- Parse arguments ----
GROUP_NAME=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --list|-l)
            cd "$SCRIPT_DIR"
            python -c "from lr_experiment_groups import list_groups; list_groups()"
            exit 0
            ;;
        -c|--concurrent)
            CONCURRENT_JOBS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 <group_name> [-c concurrent_gpus]"
            echo "       $0 --list"
            echo ""
            echo "Options:"
            echo "  <group_name>       Name of the LR experiment group"
            echo "  -c, --concurrent N Max concurrent GPU jobs (default: 8)"
            echo "  -l, --list         List available experiment groups"
            exit 0
            ;;
        *)
            GROUP_NAME="$1"
            shift
            ;;
    esac
done

if [[ -z "$GROUP_NAME" ]]; then
    echo "Error: No group name specified."
    echo "Usage: $0 <group_name>"
    echo "Run '$0 --list' to see available groups."
    exit 1
fi

# ---- Count experiments for this group ----
cd "$SCRIPT_DIR"
ARRAY_SIZE=$(python -c "
from lr_experiment_groups import LR_EXPERIMENT_GROUPS
g = LR_EXPERIMENT_GROUPS.get('${GROUP_NAME}')
if g is None:
    print('ERROR')
else:
    print(len(g['hidden_dims']) * len(g['learning_rates']))
")

if [[ "$ARRAY_SIZE" == "ERROR" ]]; then
    echo "Error: Group '${GROUP_NAME}' not found."
    python -c "from lr_experiment_groups import list_groups; list_groups()"
    exit 1
fi

MAX_INDEX=$((ARRAY_SIZE - 1))
ARRAY_SPEC="0-${MAX_INDEX}%${CONCURRENT_JOBS}"

echo "============================================================"
echo "  LR Sweep Submission: ${GROUP_NAME}"
echo "============================================================"
echo "  Total experiments:   ${ARRAY_SIZE}"
echo "  Max concurrent GPUs: ${CONCURRENT_JOBS}"
echo "  SLURM array spec:   ${ARRAY_SPEC}"
echo "============================================================"
echo ""

# ---- Submit from project root (where SLURM expects to run) ----
cd "$PROJECT_DIR"

# Step 1: Submit the sweep array job
SWEEP_JOB_ID=$(sbatch --parsable \
    --array="${ARRAY_SPEC}" \
    learning_rate_analysis/lr_job.sh "${GROUP_NAME}")

echo "Sweep job submitted: ${SWEEP_JOB_ID} (${ARRAY_SIZE} tasks, ${CONCURRENT_JOBS} concurrent)"

# Step 2: Submit the analysis job, dependent on all sweep tasks completing
ANALYZE_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${SWEEP_JOB_ID} \
    learning_rate_analysis/analyze_job.sh "${GROUP_NAME}")

echo "Analysis job submitted: ${ANALYZE_JOB_ID} (runs after sweep completes)"
echo ""
echo "Monitor with:  squeue -u $(whoami)"
echo "Cancel with:   scancel ${SWEEP_JOB_ID} ${ANALYZE_JOB_ID}"
echo ""
echo "Results will appear in:"
echo "  CSVs:  learning_rate_analysis/results/${GROUP_NAME}/"
echo "  Plots: learning_rate_analysis/plots/${GROUP_NAME}/"
