#!/bin/bash
# Usage: ./submit_variable_gpus.sh <NUM_GPUS> [partition] [mem] [gpu_type]
# Example: ./submit_variable_gpus.sh 3
# Example: ./submit_variable_gpus.sh 6 xeon-g6-volta 16G volta

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <NUM_GPUS> [partition] [mem] [gpu_type]"
  exit 1
fi

NUM_GPUS="$1"
PARTITION="${2:-xeon-g6-volta}"
MEM="${3:-16G}"
GPU_TYPE="${4:-volta}"

if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [[ "$NUM_GPUS" -lt 1 ]]; then
  echo "NUM_GPUS must be a positive integer"
  exit 1
fi

ARRAY_SPEC="0-$((NUM_GPUS - 1))"

echo "Submitting ${NUM_GPUS} GPU tasks (1 GPU per task) to partition '${PARTITION}'..."
sbatch \
  --partition="${PARTITION}" \
  --array="${ARRAY_SPEC}" \
  --gres="gpu:${GPU_TYPE}:1" \
  --mem="${MEM}" \
  job_array_worker.sh

