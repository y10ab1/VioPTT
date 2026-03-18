#!/usr/bin/env bash
set -euo pipefail

# Evaluate base VioTech technique classification on viotech test set (+ optional RWC)
# Reports per-class accuracy, confusion matrix, and F1 for:
#   - Tonal technique  (none / pizzicato / harmonics / openstring)
#   - Articulation     (none / release / staccato / spiccato)
#   - Legato           (bow_change / sustained)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="${PROJECT_ROOT}"

# ========== Configuration ==========

# Model checkpoint to evaluate
MODEL_TAG="vioptt_viotech_v0.1_wtech_parallel_b4_tech0.1_legatoImproved"
CHECKPOINT_PATH="${WORKSPACE}/checkpoints/main_contrast/${MODEL_TAG}/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=4/10000_iterations.pth"

# Viotech HDF5 dataset directory
HDF5S_DIR="${WORKSPACE}/hdf5s/viotech"

# Evaluation split
SPLIT="test"

# RWC H5 path (set to empty string or non-existent path to skip)
RWC_H5="/mnt/hdd/rwc_processed_data.h5"

DEVICE=0
BATCH_SIZE=4
NUM_WORKERS=4
MAX_ITERATIONS=500

# ===================================

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Checkpoint not found: $CHECKPOINT_PATH"
  exit 1
fi

if [[ ! -d "$HDF5S_DIR" ]]; then
  echo "HDF5s directory not found: $HDF5S_DIR"
  exit 1
fi

echo "========================================="
echo "VioTech Base Technique Evaluation"
echo "========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Dataset:    $HDF5S_DIR"
echo "Split:      $SPLIT"
echo "RWC H5:     $RWC_H5"
echo "========================================="

cd "${WORKSPACE}/piano_transcription"

# Build RWC arguments if file exists
RWC_ARGS=""
if [[ -n "$RWC_H5" ]] && [[ -f "$RWC_H5" ]]; then
  RWC_ARGS="--rwc_h5_path $RWC_H5 --rwc_split test --rwc_fold 0"
  echo "RWC H5 found — will evaluate cross-dataset."
else
  echo "RWC H5 not found at ${RWC_H5} — skipping RWC evaluation."
fi
echo ""

python3 pytorch/evaluate_viotech.py \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --hdf5s_dir "$HDF5S_DIR" \
  --split "$SPLIT" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --max_iterations "$MAX_ITERATIONS" \
  $RWC_ARGS
