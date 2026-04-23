#!/bin/bash

# Wrapper script to submit jobs with dynamic array sizes
# Auto-counts experiments from GRAND_EXPERIMENT when -N is not provided.
# Usage: ./submit_job.sh              (auto-count, 8 concurrent)
# Usage: ./submit_job.sh -24          (24 experiments, 8 concurrent)
# Usage: ./submit_job.sh -n 24 -c 4   (24 experiments, 4 concurrent)
# Usage: ./submit_job.sh --array-size 50 --concurrent 10

# Default values (ARRAY_SIZE empty = auto-count)
ARRAY_SIZE=""
CONCURRENT_JOBS=8
SCRIPT_PATH="main.sh"

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
            echo "Options:"
            echo "  -n, --array-size N    Number of array jobs (default: auto-count from GRAND_EXPERIMENT)"
            echo "  -c, --concurrent N    Max concurrent jobs (default: 8)"
            echo "  -s, --script PATH     Script to submit (default: main.sh)"
            echo "  -24                   Shorthand for --array-size 24"
            echo "  -h, --help           Show this help"
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
# GRAND_EXPERIMENT is already expanded (create_multi_lr_experiments /
# create_multi_seed_experiments return flat lists), so summing subexperiments
# across groups gives the true total.
if [[ -z "$ARRAY_SIZE" ]]; then
    # Cache the count by the mtime of the definitions file. Importing
    # experiment_definitions.py runs gen_experim() many times; on a cold FS
    # cache, that can take tens of seconds. Recount only when defs change.
    CACHE_FILE=".auto_count_cache_transformer"
    DEFS_FILES=(experiment_definitions.py experiment_utils.py)
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
        echo "Auto-counting experiments in GRAND_EXPERIMENT (first run or defs changed)..."
        ARRAY_SIZE=$(python -c "
from experiment_definitions import GRAND_EXPERIMENT
print('__COUNT__', sum(len(e['subexperiments']) for e in GRAND_EXPERIMENT))
" 2>/dev/null | grep '^__COUNT__' | awk '{print $2}')
        if [[ -z "$ARRAY_SIZE" || ! "$ARRAY_SIZE" =~ ^[0-9]+$ || "$ARRAY_SIZE" -eq 0 ]]; then
            echo "Error: auto-count failed. Pass -N explicitly, e.g. '-24'."
            exit 1
        fi
        printf '%s\n%s\n' "$CURRENT_KEY" "$ARRAY_SIZE" > "$CACHE_FILE"
        echo "Auto-detected ${ARRAY_SIZE} experiments."
    fi
fi

# Calculate max index (0-based, so subtract 1)
MAX_INDEX=$((ARRAY_SIZE - 1))
ARRAY_SPEC="0-${MAX_INDEX}%${CONCURRENT_JOBS}"

echo "Submitting job array with:"
echo "  Array specification: ${ARRAY_SPEC}"
echo "  Total experiments: ${ARRAY_SIZE}"
echo "  Max concurrent jobs: ${CONCURRENT_JOBS}"
echo "  Script: ${SCRIPT_PATH}"
echo ""

# Submit the job with the array specification
sbatch --array="${ARRAY_SPEC}" "${SCRIPT_PATH}"

if [ $? -eq 0 ]; then
    echo "Job submitted successfully!"
else
    echo "Error submitting job"
    exit 1
fi
