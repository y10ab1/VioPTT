#!/bin/bash

# Auto-detect project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="${PROJECT_ROOT}"

# Training with note-level MoE technique classification
# Uses the same transcription backbone + a new MoE technique branch
# that slices encoder features into onset/body/offset/context zones per note.

if [ ! -d "${WORKSPACE}/hdf5s/viotech" ]; then
    mkdir -p "${WORKSPACE}/hdf5s"
    echo "Creating symlink for viotech dataset..."
    ln -s /mnt/hdd/viotech "${WORKSPACE}/hdf5s/viotech"
fi

TB="${WORKSPACE}/tb"
PRETRAIN_PATH="${WORKSPACE}/checkpoints/transcriptor_model.pth"
MODEL_TAG="vioptt_viotech_moe_technique_v0.1"

cd "${WORKSPACE}/piano_transcription"

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
    --technique_weight=0.0 \
    --technique_moe_weight=0.1 \
    --moe_balance_coeff=0.01 \
    
