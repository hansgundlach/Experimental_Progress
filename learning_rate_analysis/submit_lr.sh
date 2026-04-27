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
# What happens (concrete group):
#   1. Submits a SLURM array job (1 GPU per task, up to 8 concurrent)
#      that runs all (hidden_dim, lr) combos for the group.
#   2. Submits a dependency job that runs analyze_lr.py after all
#      sweep tasks finish — produces plots and the fitted LR table
#      automatically.
#
# What happens (combined group, i.e. group with `combine: [...]`):
#   1. For each listed subgroup, submits a SLURM array job as above.
#   2. Submits a single analyze job for the combined group name,
#      depending on all sub-arrays. analyze_lr.py reads every subgroup's
#      CSVs and fits one LR scaling law across the pooled data, writing
#      plots/CSVs to plots/<combined_name>/.

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

describe_group() {
    # Returns "concrete <N>" or "combined <sub1,sub2,...>" or "ERROR".
    local group="$1"
    cd "$SCRIPT_DIR"
    python -c "
from lr_experiment_groups import LR_EXPERIMENT_GROUPS, is_combined_group, expand_subgroups
g = LR_EXPERIMENT_GROUPS.get('${group}')
if g is None:
    print('ERROR')
elif is_combined_group('${group}'):
    print('combined ' + ','.join(expand_subgroups('${group}')))
else:
    print('concrete ' + str(len(g['hidden_dims']) * len(g['learning_rates'])))
"
}

submit_concrete_array() {
    # Submits the SLURM array job for a concrete group. Echoes the sweep job id.
    local group="$1"
    local array_size="$2"
    local max_index=$((array_size - 1))
    local array_spec="0-${max_index}%${CONCURRENT_JOBS}"

    echo "  ${group}: ${array_size} experiments, array=${array_spec}" >&2

    cd "$PROJECT_DIR"
    sbatch --parsable \
        --array="${array_spec}" \
        learning_rate_analysis/lr_job.sh "${group}"
}

submit_group() {
    local group="$1"

    local desc
    desc=$(describe_group "$group")
    if [[ "$desc" == "ERROR" ]]; then
        echo "Error: Group '${group}' not found."
        python -c "from lr_experiment_groups import list_groups; list_groups()"
        return 1
    fi

    # All sbatch calls below use paths relative to PROJECT_DIR.
    # `--all` cd's to SCRIPT_DIR before this is called, and
    # submit_concrete_array runs in a $() subshell, so its own cd doesn't
    # propagate back here. Cd in the parent shell now to keep both the
    # array and analyze sbatch invocations consistent.
    cd "$PROJECT_DIR"

    echo "============================================================"
    echo "  LR Sweep Submission: ${group}"
    echo "============================================================"

    local kind="${desc%% *}"
    local rest="${desc#* }"

    if [[ "$kind" == "concrete" ]]; then
        local array_size="$rest"
        echo "  Type:                concrete"
        echo "  Total experiments:   ${array_size}"
        echo "  Max concurrent GPUs: ${CONCURRENT_JOBS}"
        echo "============================================================"
        echo ""

        local sweep_job_id
        sweep_job_id=$(submit_concrete_array "$group" "$array_size")
        echo "Sweep job submitted: ${sweep_job_id}"

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
        return 0
    fi

    # Combined group: submit each subgroup's array, then one analyze job
    # for the combined name dependent on all sub-arrays succeeding.
    local subs="$rest"
    echo "  Type:                combined"
    echo "  Subgroups:           ${subs}"
    echo "  Max concurrent GPUs: ${CONCURRENT_JOBS} (per subgroup array)"
    echo "============================================================"
    echo ""

    local dep_list=""
    IFS=',' read -ra sub_arr <<< "$subs"
    for sub in "${sub_arr[@]}"; do
        local sub_desc
        sub_desc=$(describe_group "$sub")
        if [[ "${sub_desc%% *}" != "concrete" ]]; then
            echo "Error: subgroup '${sub}' is not concrete." >&2
            return 1
        fi
        local sub_size="${sub_desc#* }"
        local sub_job_id
        sub_job_id=$(submit_concrete_array "$sub" "$sub_size")
        echo "Sweep job submitted: ${sub_job_id}  (subgroup ${sub})"
        if [[ -z "$dep_list" ]]; then
            dep_list="$sub_job_id"
        else
            dep_list="${dep_list}:${sub_job_id}"
        fi
    done

    local analyze_job_id
    analyze_job_id=$(sbatch --parsable \
        --dependency=afterok:${dep_list} \
        learning_rate_analysis/analyze_job.sh "${group}")
    echo "Analysis job submitted: ${analyze_job_id} (runs after all subgroup sweeps complete)"
    echo ""
    echo "Combined results will appear in:"
    echo "  Plots: learning_rate_analysis/plots/${group}/"
    for sub in "${sub_arr[@]}"; do
        echo "  Sub CSVs: learning_rate_analysis/results/${sub}/"
    done
    echo ""
}

if [[ "$RUN_ALL" -eq 1 ]]; then
    cd "$SCRIPT_DIR"
    GROUPS_FILE=$(mktemp)
    # Skip concrete groups that are already covered by a combined group --
    # otherwise --all would submit those subgroups twice (once standalone,
    # once via the combined group's expansion). Combined groups always run.
    python -c "
from lr_experiment_groups import LR_EXPERIMENT_GROUPS, is_combined_group, expand_subgroups
covered = set()
for name in LR_EXPERIMENT_GROUPS:
    if is_combined_group(name):
        covered.update(expand_subgroups(name))
for name in LR_EXPERIMENT_GROUPS:
    if name in covered and not is_combined_group(name):
        continue
    print(name)
" > "$GROUPS_FILE"

    echo "Submitting all groups (subgroups of combined groups are skipped to avoid duplicate runs):"
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
