#!/bin/bash

# Wrapper script to submit LSTM single-GPU jobs with dynamic array sizes
# Each LSTM experiment uses 1 GPU for single-GPU training (more experiments, less memory per job)
# Auto-counts experiments from GRAND_EXPERIMENT when -N is not provided.
# Usage: ./submit_lstm_job.sh              (auto-count, 8 concurrent)
# Usage: ./submit_lstm_job.sh -50          (50 experiments, 8 concurrent - using 8 GPUs total)
# Usage: ./submit_lstm_job.sh -n 50 -c 4   (50 experiments, 4 concurrent - using 4 GPUs total)
# Usage: ./submit_lstm_job.sh --array-size 24 --concurrent 8

# Default values (ARRAY_SIZE empty = auto-count)
ARRAY_SIZE=""
CONCURRENT_JOBS=8  # Default to use all 8 GPUs since each job needs only 1 GPU
SCRIPT_PATH="LSTM_model/lstm.sh"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--array-size)
            ARRAY_SIZE="$2"
            shift 2
            ;;
        -c|--concurrent)
            CONCURRENT_JOBS="$2"
            shift 2
            ;;
        -s|--script)
            SCRIPT_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Submit LSTM single-GPU training jobs. Each experiment uses 1 GPU for single-GPU training."
            echo ""
            echo "Options:"
            echo "  -n, --array-size N    Number of array jobs (default: auto-count from GRAND_EXPERIMENT)"
            echo "  -c, --concurrent N    Max concurrent jobs (default: 8)"
            echo "                        Note: Each job uses 1 GPU, so concurrent = total GPU usage"
            echo "  -s, --script PATH     Script to submit (default: lstm.sh)"
            echo "  -50                   Shorthand for --array-size 50"
            echo "  -h, --help           Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 -50                # 50 experiments, 8 concurrent (8 GPUs total)"
            echo "  $0 -n 100 -c 8       # 100 experiments, 8 concurrent (8 GPUs total)"
            echo "  $0 -24 -c 4          # 24 experiments, 4 concurrent (4 GPUs total)"
            exit 0
            ;;
        -*)
            # Check if it's a negative number (array size shorthand)
            if [[ $1 =~ ^-[0-9]+$ ]]; then
                ARRAY_SIZE="${1:1}"  # Remove the minus sign
                shift
            else
                echo "Unknown option $1"
                echo "Use -h or --help for usage information"
                exit 1
            fi
            ;;
        *)
            echo "Unknown argument $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Auto-count experiments if -n was not given.
# LSTM GRAND_EXPERIMENT is already expanded (create_multi_lr_experiments /
# create_multi_seed_lstm_experiments return flat lists).
if [[ -z "$ARRAY_SIZE" ]]; then
    # Cache the count by source-file mtime. Importing lstm_experiment_definitions
    # pulls in torch (via lstm_experiment_utils), which can take ~1 minute on a
    # cold LLGrid filesystem. Recount only when definitions change.
    CACHE_FILE=".auto_count_cache_lstm"
    DEFS_FILES=(LSTM_model/lstm_experiment_definitions.py LSTM_model/lstm_experiment_utils.py experiment_utils.py)
    CURRENT_KEY=$(stat -c '%Y' "${DEFS_FILES[@]}" 2>/dev/null | tr '\n' ':')

    if [[ -f "$CACHE_FILE" ]]; then
        CACHED_KEY=$(head -1 "$CACHE_FILE")
        CACHED_COUNT=$(tail -1 "$CACHE_FILE")
        if [[ "$CACHED_KEY" == "$CURRENT_KEY" && "$CACHED_COUNT" =~ ^[0-9]+$ ]]; then
            ARRAY_SIZE="$CACHED_COUNT"
            echo "Auto-detected ${ARRAY_SIZE} experiments (cached; run 'rm ${CACHE_FILE}' to force recount)."
        fi
    fi

    if [[ -z "$ARRAY_SIZE" ]]; then
        echo "Auto-counting LSTM experiments in GRAND_EXPERIMENT..."
        ARRAY_SIZE=$(cd LSTM_model && python -c "
from lstm_experiment_definitions import GRAND_EXPERIMENT
print('__COUNT__', sum(len(e['subexperiments']) for e in GRAND_EXPERIMENT))
" 2>/dev/null | grep '^__COUNT__' | awk '{print $2}')
        if [[ -z "$ARRAY_SIZE" || ! "$ARRAY_SIZE" =~ ^[0-9]+$ || "$ARRAY_SIZE" -eq 0 ]]; then
            echo "Error: auto-count failed. Pass -N explicitly, e.g. '-50'."
            exit 1
        fi
        printf '%s\n%s\n' "$CURRENT_KEY" "$ARRAY_SIZE" > "$CACHE_FILE"
        echo "Auto-detected ${ARRAY_SIZE} experiments."
    fi
fi

# Calculate max index (0-based, so subtract 1)
MAX_INDEX=$((ARRAY_SIZE - 1))
ARRAY_SPEC="0-${MAX_INDEX}%${CONCURRENT_JOBS}"

# Calculate total GPU usage
TOTAL_GPU_USAGE=${CONCURRENT_JOBS}  # Single-GPU training: 1 GPU per job

echo "Submitting LSTM single-GPU job array with:"
echo "  Array specification: ${ARRAY_SPEC}"
echo "  Total experiments: ${ARRAY_SIZE}"
echo "  Max concurrent jobs: ${CONCURRENT_JOBS}"
echo "  GPUs per job: 1 (single-GPU training)"
echo "  Total GPU usage: ${TOTAL_GPU_USAGE} GPUs"
echo "  Script: ${SCRIPT_PATH}"
echo ""

# Warn if high GPU usage
if [ $TOTAL_GPU_USAGE -gt 8 ]; then
    echo "⚠️  WARNING: This will use ${TOTAL_GPU_USAGE} GPUs concurrently!"
    echo "   Your cluster limit is 8 GPUs. Consider reducing concurrent jobs with -c option."
    echo ""
fi

# Submit the job with the array specification
echo "Submitting with command: sbatch --array=\"${ARRAY_SPEC}\" \"${SCRIPT_PATH}\""
echo "Checking if script exists: $(ls -la "${SCRIPT_PATH}" 2>/dev/null || echo "SCRIPT NOT FOUND")"
sbatch --array="${ARRAY_SPEC}" "${SCRIPT_PATH}"

if [ $? -eq 0 ]; then
    echo "✅ LSTM single-GPU job array submitted successfully!"
    echo "   Monitor with: squeue -u $(whoami)"
    echo "   Cancel with: scancel <job_id>"
else
    echo "❌ Error submitting job"
    exit 1
fi