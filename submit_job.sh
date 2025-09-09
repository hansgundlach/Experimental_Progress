#!/bin/bash

# Wrapper script to submit jobs with dynamic array sizes
# Usage: ./submit_job.sh -24          (24 experiments, 8 concurrent)
# Usage: ./submit_job.sh -n 24 -c 4   (24 experiments, 4 concurrent)
# Usage: ./submit_job.sh --array-size 50 --concurrent 10

# Default values
ARRAY_SIZE=24
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
            echo "  -n, --array-size N    Number of array jobs (default: 24)"
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
