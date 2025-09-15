#!/bin/bash

# Wrapper script to submit LSTM single-GPU jobs with dynamic array sizes
# Each LSTM experiment uses 1 GPU for single-GPU training (more experiments, less memory per job)
# Usage: ./submit_lstm_job.sh -50          (50 experiments, 8 concurrent - using 8 GPUs total)
# Usage: ./submit_lstm_job.sh -n 50 -c 4   (50 experiments, 4 concurrent - using 4 GPUs total)
# Usage: ./submit_lstm_job.sh --array-size 24 --concurrent 8

# Default values
ARRAY_SIZE=24
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
            echo "  -n, --array-size N    Number of array jobs (default: 24)"
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