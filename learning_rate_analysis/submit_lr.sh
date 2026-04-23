#!/bin/bash
#
# Submit an LR sweep for a given experiment group.
#
# Usage:
#   bash submit_lr.sh modern_transformer
#   bash submit_lr.sh sgd_transformer
#   bash submit_lr.sh --list                  # list available groups
#   bash submit_lr.sh --all                   # submit every defined group
#   bash submit_lr.sh modern_transformer -c 4 # limit to 4 concurrent GPUs
#
# What happens (per group):
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
RUN_ALL=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --list|-l)
            cd "$SCRIPT_DIR"
            python -c "from lr_experiment_groups import list_groups; list_groups()"
            exit 0
            ;;
        --all|-a)
            RUN_ALL=1
            shift
            ;;
        -c|--concurrent)
            CONCURRENT_JOBS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 <group_name> [-c concurrent_gpus]"
            echo "       $0 --all [-c concurrent_gpus]"
            echo "       $0 --list"
            echo ""
            echo "Options:"
            echo "  <group_name>       Name of the LR experiment group"
            echo "  -a, --all          Submit every group defined in LR_EXPERIMENT_GROUPS"
            echo "  -c, --concurrent N Max concurrent GPU jobs per group (default: 8)"
            echo "  -l, --list         List available experiment groups"
            exit 0
            ;;
        *)
            GROUP_NAME="$1"
            shift
            ;;
    esac
done

if [[ "$RUN_ALL" -eq 1 && -n "$GROUP_NAME" ]]; then
    echo "Error: specify either a group name or --all, not both."
    exit 1
fi

if [[ "$RUN_ALL" -eq 0 && -z "$GROUP_NAME" ]]; then
    echo "Error: No group name specified."
    echo "Usage: $0 <group_name>"
    echo "       $0 --all"
    echo "Run '$0 --list' to see available groups."
    exit 1
fi

submit_group() {
    local group="$1"

    cd "$SCRIPT_DIR"
    local array_size
    array_size=$(python -c "
from lr_experiment_groups import LR_EXPERIMENT_GROUPS
g = LR_EXPERIMENT_GROUPS.get('${group}')
if g is None:
    print('ERROR')
else:
    print(len(g['hidden_dims']) * len(g['learning_rates']))
")

    if [[ "$array_size" == "ERROR" ]]; then
        echo "Error: Group '${group}' not found."
        python -c "from lr_experiment_groups import list_groups; list_groups()"
        return 1
    fi

    local max_index=$((array_size - 1))
    local array_spec="0-${max_index}%${CONCURRENT_JOBS}"

    echo "============================================================"
    echo "  LR Sweep Submission: ${group}"
    echo "============================================================"
    echo "  Total experiments:   ${array_size}"
    echo "  Max concurrent GPUs: ${CONCURRENT_JOBS}"
    echo "  SLURM array spec:   ${array_spec}"
    echo "============================================================"
    echo ""

    cd "$PROJECT_DIR"

    local sweep_job_id
    sweep_job_id=$(sbatch --parsable \
        --array="${array_spec}" \
        learning_rate_analysis/lr_job.sh "${group}")

    echo "Sweep job submitted: ${sweep_job_id} (${array_size} tasks, ${CONCURRENT_JOBS} concurrent)"

    local analyze_job_id
    analyze_job_id=$(sbatch --parsable \
        --dependency=afterok:${sweep_job_id} \
        learning_rate_analysis/analyze_job.sh "${group}")

    echo "Analysis job submitted: ${analyze_job_id} (runs after sweep completes)"
    echo ""
    echo "Results will appear in:"
    echo "  CSVs:  learning_rate_analysis/results/${group}/"
    echo "  Plots: learning_rate_analysis/plots/${group}/"
    echo ""
}

if [[ "$RUN_ALL" -eq 1 ]]; then
    cd "$SCRIPT_DIR"
    GROUPS_FILE=$(mktemp)
    python -c "
from lr_experiment_groups import LR_EXPERIMENT_GROUPS
for name in LR_EXPERIMENT_GROUPS:
    print(name)
" > "$GROUPS_FILE"

    echo "Submitting all groups:"
    cat "$GROUPS_FILE" | sed 's/^/  - /'
    echo ""

    while read -r g; do
        [[ -z "$g" ]] && continue
        submit_group "$g"
    done < "$GROUPS_FILE"

    rm -f "$GROUPS_FILE"
else
    submit_group "$GROUP_NAME"
fi

echo "Monitor with:  squeue -u $(whoami)"
