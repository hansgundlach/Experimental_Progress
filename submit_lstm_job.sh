#!/bin/bash

# Wrapper script to submit LSTM DDP jobs with dynamic array sizes
# Each LSTM experiment requires 2 GPUs for Distributed Data Parallel (DDP) training
# Usage: ./submit_lstm_job.sh -50          (50 experiments, 4 concurrent - using 8 GPUs total)
# Usage: ./submit_lstm_job.sh -n 50 -c 2   (50 experiments, 2 concurrent - using 4 GPUs total)
# Usage: ./submit_lstm_job.sh --array-size 24 --concurrent 8

# Default values
ARRAY_SIZE=24
CONCURRENT_JOBS=4  # Conservative default since each job needs 2 GPUs
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
            echo "Submit LSTM DDP training jobs. Each experiment uses 2 GPUs for distributed training."
            echo ""
            echo "Options:"
            echo "  -n, --array-size N    Number of array jobs (default: 24)"
            echo "  -c, --concurrent N    Max concurrent jobs (default: 4)"
            echo "                        Note: Each job uses 2 GPUs, so concurrent*2 = total GPU usage"
            echo "  -s, --script PATH     Script to submit (default: lstm.sh)"
            echo "  -50                   Shorthand for --array-size 50"
            echo "  -h, --help           Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 -50                # 50 experiments, 4 concurrent (8 GPUs total)"
            echo "  $0 -n 100 -c 8       # 100 experiments, 8 concurrent (16 GPUs total)"
            echo "  $0 -24 -c 2          # 24 experiments, 2 concurrent (4 GPUs total)"
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

# Calculate max index (0-based, so subtract 1)
MAX_INDEX=$((ARRAY_SIZE - 1))
ARRAY_SPEC="0-${MAX_INDEX}%${CONCURRENT_JOBS}"

# Calculate total GPU usage
TOTAL_GPU_USAGE=$((CONCURRENT_JOBS * 2))

echo "Submitting LSTM DDP job array with:"
echo "  Array specification: ${ARRAY_SPEC}"
echo "  Total experiments: ${ARRAY_SIZE}"
echo "  Max concurrent jobs: ${CONCURRENT_JOBS}"
echo "  GPUs per job: 2 (DDP training)"
echo "  Total GPU usage: ${TOTAL_GPU_USAGE} GPUs"
echo "  Script: ${SCRIPT_PATH}"
echo ""

# Warn if high GPU usage
if [ $TOTAL_GPU_USAGE -gt 16 ]; then
    echo "⚠️  WARNING: This will use ${TOTAL_GPU_USAGE} GPUs concurrently!"
    echo "   Make sure your cluster has sufficient GPU resources."
    echo "   Consider reducing concurrent jobs with -c option."
    echo ""
fi

# Submit the job with the array specification
sbatch --array="${ARRAY_SPEC}" "${SCRIPT_PATH}"

if [ $? -eq 0 ]; then
    echo "✅ LSTM DDP job array submitted successfully!"
    echo "   Monitor with: squeue -u $(whoami)"
    echo "   Cancel with: scancel <job_id>"
else
    echo "❌ Error submitting job"
    exit 1
fi