#!/bin/bash

# Evaluate MoE technique classification on viotech test set
# Prints per-class accuracy, confusion matrix, and F1 for:
#   - Tonal technique  (none / pizzicato / harmonics / openstring)
#   - Articulation     (none / release / staccato / spiccato)
#   - Legato           (bow_change / sustained)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="${PROJECT_ROOT}"

# ---- Configure these ----
CHECKPOINT="${1:-/root/VioPTT/checkpoints/main_contrast/vioptt_viotech_moe_technique_v0.1/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=4/10000_iterations.pth}"
HDF5S_DIR="${2:-${WORKSPACE}/hdf5s/viotech}"
SPLIT="${3:-validation}"
DEVICE="${4:-0}"

cd "${WORKSPACE}/piano_transcription"

echo "============================================"
echo "  MoE Technique Evaluation"
echo "============================================"
echo "  Checkpoint : ${CHECKPOINT}"
echo "  HDF5s dir  : ${HDF5S_DIR}"
echo "  Split      : ${SPLIT}"
echo "  Device     : ${DEVICE}"
echo "============================================"
echo ""

python3 pytorch/evaluate_moe_technique.py \
    --checkpoint_path "$CHECKPOINT" \
    --hdf5s_dir "$HDF5S_DIR" \
    --split "$SPLIT" \
    --device "$DEVICE" \
    --batch_size 4 \
    --num_workers 4 \
    --max_iterations 500
