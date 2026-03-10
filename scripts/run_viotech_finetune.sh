#!/bin/bash

# Auto-detect project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="${PROJECT_ROOT}"

# Training High Resolution Piano Transcription (HRPT) for Viotech

# Notice:
# 1. The checkpoints will be saved in this directory as `./checkpoints` under WORKSPACE
# 2. Place dataset path under WORKSPACE, it should be {WORKSPACE}/hdf5s/viotech/...

# Create symlink for viotech dataset if it doesn't exist
if [ ! -d "${WORKSPACE}/hdf5s/viotech" ]; then
    mkdir -p "${WORKSPACE}/hdf5s"
    echo "Creating symlink for viotech dataset..."
    ln -s /mnt/hdd/viotech "${WORKSPACE}/hdf5s/viotech"
fi

# Tensorboard log directory
TB="${WORKSPACE}/tb"

# Pretrained Violin Transcriptor model path (uncomment to use)
PRETRAIN_PATH="${WORKSPACE}/checkpoints/transcriptor_model.pth"

MODEL_TAG="vioptt_viotech_v0.1_wtech_parallel_b4_tech0.1_legatoImproved"
cd "${WORKSPACE}/piano_transcription"

# --- 1. Train note transcription system ---
python3 pytorch/main_contrast.py train \
    --workspace=$WORKSPACE \
    --logdir=$TB \
    --pretrain_path=$PRETRAIN_PATH \
    --model_tag $MODEL_TAG \
    --model_type='Regress_onset_offset_frame_velocity_CRNN' \
    --loss_type='regress_onset_offset_frame_velocity_bce' \
    --augmentation='aug' \
    --max_note_shift=2 \
    --batch_size=4 \
    --learning_rate=5e-4 \
    --reduce_iteration=1000 \
    --resume_iteration=0 \
    --early_stop=10000 \
    --device 1 \
    --dataset viotech \
    --contrast_weight=0.0 \
    --ctc_weight=0.0 \
    --num_workers=8 \
    --technique_weight=0.1
