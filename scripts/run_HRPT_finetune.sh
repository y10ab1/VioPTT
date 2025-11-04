#!/bin/bash

# Auto-detect project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="${PROJECT_ROOT}"

# Training High Resolution Piano Transcription (HRPT)

# Notice:
# 1. The checkpoints will be saved in this directory as `./checkpoints` under WORKSPACE
# 2. Place dataset path under WORKSPACE, it should be {WORKSPACE}/hdf5s/mosa/...

# Tensorboard log directory
TB="${WORKSPACE}/tb"

# Piano Pretrained model path (uncomment to use)
# PRETRAIN_PATH="${WORKSPACE}/checkpoints/pretrained/high_resolution_MAESTRO_augmentations.pth"

MODEL_TAG="w_aug_w_technique_annotation_mixed_dataset_mosapt_ssv_local_technique_feature_10frames_volume_normalized_per_class_acc_window_size30_0907"
cd "${WORKSPACE}/piano_transcription"
# --- 1. Train note transcription system ---
python3 pytorch/main_contrast.py train \
    --workspace=$WORKSPACE \
    --logdir=$TB \
    --model_tag $MODEL_TAG \
    --model_type='Regress_onset_offset_frame_velocity_CRNN' \
    --loss_type='regress_onset_offset_frame_velocity_bce' \
    --augmentation='aug' \
    --max_note_shift=2 \
    --batch_size=5 \
    --learning_rate=5e-4 \
    --reduce_iteration=1000 \
    --resume_iteration=0 \
    --early_stop=10000 \
    --device 0 \
    --dataset mixed \
    --contrast_weight=0.0 \
    --ctc_weight=0.0 \
    --technique_weight=1.0 
    # --pretrain_path=$PRETRAIN_PATH 

