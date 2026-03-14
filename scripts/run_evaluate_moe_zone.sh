#!/bin/bash

# Evaluate Zone-Specialized MoE technique classification
# Expert 0: Onset specialist | Expert 1: Body specialist
# Expert 2: Offset specialist | Expert 3: Holistic expert

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="${PROJECT_ROOT}"

CHECKPOINT="${1:-${WORKSPACE}/checkpoints/your_zone_moe_checkpoint.pth}"
HDF5S_DIR="${2:-${WORKSPACE}/hdf5s/viotech}"
SPLIT="${3:-test}"
DEVICE="${4:-0}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Checkpoint not found: $CHECKPOINT"
    echo "Usage: $0 [checkpoint_path] [hdf5s_dir] [split] [device]"
    exit 1
fi

cd "${WORKSPACE}/piano_transcription"

python3 pytorch/evaluate_moe_zone.py \
    --checkpoint_path "$CHECKPOINT" \
    --hdf5s_dir "$HDF5S_DIR" \
    --split "$SPLIT" \
    --device "$DEVICE" \
    --batch_size 4 \
    --num_workers 4 \
    --max_iterations 500
