#!/bin/bash

# Auto-detect project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="${PROJECT_ROOT}"

# Path to trained model checkpoint
# CHECKPOINT_PATH="${WORKSPACE}/checkpoints/transcriptor_model.pth"
CHECKPOINT_PATH="/root/VioPTT/checkpoints/main_contrast/vioptt_viotech_v0.1/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=4/2000_iterations.pth"

# Model type (adjust based on your model)
MODEL_TYPE="Regress_onset_offset_frame_velocity_CRNN"

# Post processor type
POST_PROCESSOR="regression"  # or "onsets_frames""

# Model tag
# MODEL_TAG="transcriptor_model"
MODEL_TAG="vioptt_viotech_v0.1_2000"

# Set environment variables
export BACH10_DIR="/mnt/hdd/Bach10"
export URMP_DIR="/mnt/hdd/URMP_Violin/Processed/Dataset"

# create logs directory if it doesn't exist
mkdir -p "${WORKSPACE}/logs"

# Run evaluation for Bach10
python "${WORKSPACE}/evaluate.py" \
    --model_tag $MODEL_TAG \
    --dataset bach10 \
    --checkpoint_path $CHECKPOINT_PATH \
    --model_type $MODEL_TYPE \
    --post_processor_type $POST_PROCESSOR \
    --device 0 \
    > "${WORKSPACE}/logs/bach10_eval_${MODEL_TAG}.log" 2>&1
echo "Bach10 evaluation completed, results saved in logs/bach10_eval_${MODEL_TAG}.log"

# Run evaluation for URMP
python "${WORKSPACE}/evaluate.py" \
    --model_tag $MODEL_TAG \
    --dataset urmp \
    --checkpoint_path $CHECKPOINT_PATH \
    --model_type $MODEL_TYPE \
    --post_processor_type $POST_PROCESSOR \
    --device 0 \
    > "${WORKSPACE}/logs/urmp_eval_${MODEL_TAG}.log" 2>&1
echo "URMP evaluation completed, results saved in logs/urmp_eval_${MODEL_TAG}.log"

# # Run evaluation for MOSAPT (ba3)
# python "${WORKSPACE}/evaluate.py" \
#     --model_tag $MODEL_TAG \
#     --dataset mosapt \
#     --checkpoint_path $CHECKPOINT_PATH \
#     --model_type $MODEL_TYPE \
#     --post_processor_type $POST_PROCESSOR \
#     --device 0 \
#     > "${WORKSPACE}/logs/mosapt_ba3_eval_${MODEL_TAG}.log" 2>&1
# echo "MOSAPT (ba3) evaluation completed, results saved in logs/mosapt_ba3_eval_${MODEL_TAG}.log"

