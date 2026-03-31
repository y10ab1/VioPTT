#!/bin/bash

# Note-level evaluation of Frame-level Multi-Scale MoE technique classification
# End-to-end: uses model's OWN transcription to detect note boundaries (no GT)
# Then aggregates frame-level MoE predictions per detected note (avg logits → argmax)
# GT technique per note = majority vote over frame-level GT within the same span

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="${PROJECT_ROOT}"

CHECKPOINT="${1:-/root/VioPTT/checkpoints/main_contrast/vioptt_viotech_frame_moe_reduced_lregression_viotech_mixed_mosavpt/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=4/10000_iterations.pth}"
HDF5S_DIR="${2:-${WORKSPACE}/hdf5s/viotech}"
SPLIT="${3:-validation}"
DEVICE="${4:-0}"
RWC_H5="${5:-/mnt/hdd/rwc_processed_data.h5}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Checkpoint not found: $CHECKPOINT"
    echo "Usage: $0 [checkpoint_path] [hdf5s_dir] [split] [device] [rwc_h5_path]"
    exit 1
fi

cd "${WORKSPACE}/piano_transcription"

echo "============================================"
echo "  Frame MoE NOTE-LEVEL Evaluation"
echo "  (predicted note boundaries, no GT boundaries)"
echo "============================================"
echo "  Checkpoint : ${CHECKPOINT}"
echo "  HDF5s dir  : ${HDF5S_DIR}"
echo "  Split      : ${SPLIT}"
echo "  Device     : ${DEVICE}"
echo "  RWC H5     : ${RWC_H5}"
echo "============================================"
echo ""

# Build RWC argument if file exists
RWC_ARGS=""
if [ -f "$RWC_H5" ]; then
    RWC_ARGS="--rwc_h5_path $RWC_H5 --rwc_split test --rwc_fold 0"
    echo "RWC H5 found — will evaluate on RWC as well."
else
    echo "RWC H5 not found at ${RWC_H5} — skipping RWC evaluation."
fi
echo ""

python3 pytorch/evaluate_frame_moe_notelevel.py \
    --checkpoint_path "$CHECKPOINT" \
    --hdf5s_dir "$HDF5S_DIR" \
    --split "$SPLIT" \
    --device "$DEVICE" \
    --batch_size 4 \
    --num_workers 4 \
    --max_iterations 500 \
    --fmoe_spectral_expert=0 \
    $RWC_ARGS
