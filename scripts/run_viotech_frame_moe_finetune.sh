#!/bin/bash

# Auto-detect project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="${PROJECT_ROOT}"

# Training with Frame-level Multi-Scale MoE technique classification (end-to-end)
# Expert 0: Onset specialist  (~110ms + onset prob)
# Expert 1: Note specialist   (~510ms + frame activity)
# Expert 2: Phrase specialist  (~1.5s  + all cues)
# No GT note boundaries required — uses transcription branch cues.
# Per-task gating (tonal / articulation / legato), top-k=2 sparse, low balance.

if [ ! -d "${WORKSPACE}/hdf5s/viotech" ]; then
    mkdir -p "${WORKSPACE}/hdf5s"
    echo "Creating symlink for viotech dataset..."
    ln -s /mnt/hdd/viotech "${WORKSPACE}/hdf5s/viotech"
fi

TB="${WORKSPACE}/tb"
PRETRAIN_PATH="${WORKSPACE}/checkpoints/transcriptor_model.pth"
MODEL_TAG="vioptt_viotech_frame_moe_v2.0_viotech_mixed_mosavpt_reproduce"

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
    --device 3 \
    --dataset viotech_mixed_mosavpt \
    --contrast_weight=0.0 \
    --ctc_weight=0.0 \
    --num_workers=8 \
    --technique_weight=0.0 \
    --technique_moe_weight=0.0 \
    --technique_moe_zone_weight=0.0 \
    --technique_moe_zone_pt_weight=0.0 \
    --technique_frame_moe_weight=0.1 \
    --frame_moe_balance_coeff=0.001 \
    --fmoe_spectral_expert=0 \
    