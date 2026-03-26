#!/bin/bash

# Two-stage training:
#   Stage 1: Pretrain on MOSA_VPT until validation plateaus (patience-based early stop)
#   Stage 2: Fine-tune best checkpoint on VioTech
#
# Usage:
#   bash scripts/run_mosavpt_pretrain_then_viotech_finetune.sh [GPU_ID]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="${PROJECT_ROOT}"

# ── Shared config ─────────────────────────────────────────────────────
PRETRAIN_TRANSCRIPTOR="${WORKSPACE}/checkpoints/transcriptor_model.pth"
MODEL_TYPE='Regress_onset_offset_frame_velocity_CRNN'
LOSS_TYPE='regress_onset_offset_frame_velocity_bce'
BATCH_SIZE=4
LR=5e-4
NUM_WORKERS=8
FMOE_SPECTRAL=0
TECHNIQUE_FRAME_MOE_WEIGHT=0.1
FRAME_MOE_BALANCE_COEFF=0.001

# ── Stage 1: Pretrain on MOSA_VPT ────────────────────────────────────
STAGE1_TAG="stage1_mosavpt_pretrain"
STAGE1_MAX_ITER=10000
STAGE1_PATIENCE=2000

echo "============================================================"
echo "  Stage 1: Pretrain on MOSA_VPT"
echo "  max_iter=${STAGE1_MAX_ITER}, patience=${STAGE1_PATIENCE}"
echo "  GPU=${GPU}"
echo "============================================================"
echo ""

# Ensure mosapt symlink exists
if [ ! -d "${WORKSPACE}/hdf5s/mosapt" ]; then
    mkdir -p "${WORKSPACE}/hdf5s"
    echo "Creating symlink for mosapt dataset..."
    ln -s /mnt/hdd/mosavpt_hdf5_all_for_viotech_model "${WORKSPACE}/hdf5s/mosapt"
fi

cd "${WORKSPACE}/piano_transcription"

python3 pytorch/main_contrast.py train \
    --workspace=$WORKSPACE \
    --logdir="${WORKSPACE}/tb" \
    --pretrain_path=$PRETRAIN_TRANSCRIPTOR \
    --model_tag $STAGE1_TAG \
    --model_type=$MODEL_TYPE \
    --loss_type=$LOSS_TYPE \
    --augmentation='aug' \
    --max_note_shift=2 \
    --batch_size=$BATCH_SIZE \
    --learning_rate=$LR \
    --reduce_iteration=500 \
    --resume_iteration=0 \
    --early_stop=$STAGE1_MAX_ITER \
    --patience=$STAGE1_PATIENCE \
    --best_metric='fmoe_loss_technique' \
    --device 0 \
    --dataset mosapt \
    --contrast_weight=0.0 \
    --ctc_weight=0.0 \
    --num_workers=$NUM_WORKERS \
    --technique_weight=0.0 \
    --technique_moe_weight=0.0 \
    --technique_moe_zone_weight=0.0 \
    --technique_moe_zone_pt_weight=0.0 \
    --technique_frame_moe_weight=$TECHNIQUE_FRAME_MOE_WEIGHT \
    --frame_moe_balance_coeff=$FRAME_MOE_BALANCE_COEFF \
    --fmoe_spectral_expert=$FMOE_SPECTRAL

# Locate best checkpoint from Stage 1
STAGE1_CKPT_DIR="${WORKSPACE}/checkpoints/main_contrast/${STAGE1_TAG}/${MODEL_TYPE}/loss_type=${LOSS_TYPE}/augmentation=aug/max_note_shift=2/batch_size=${BATCH_SIZE}"
STAGE1_BEST="${STAGE1_CKPT_DIR}/best_model.pth"

if [ ! -f "$STAGE1_BEST" ]; then
    echo ""
    echo "ERROR: Stage 1 best checkpoint not found at ${STAGE1_BEST}"
    echo "Falling back to latest numbered checkpoint..."
    STAGE1_BEST=$(ls -t "${STAGE1_CKPT_DIR}"/*_iterations.pth 2>/dev/null | head -1)
    if [ -z "$STAGE1_BEST" ]; then
        echo "No checkpoint found. Aborting."
        exit 1
    fi
fi

echo ""
echo "============================================================"
echo "  Stage 1 complete. Best checkpoint:"
echo "    ${STAGE1_BEST}"
echo "============================================================"
echo ""

# ── Stage 2: Fine-tune on VioTech ────────────────────────────────────
STAGE2_TAG="stage2_viotech_finetune_from_mosavpt"
STAGE2_MAX_ITER=10000
STAGE2_LR=1e-4

echo "============================================================"
echo "  Stage 2: Fine-tune on VioTech"
echo "  pretrain_from=${STAGE1_BEST}"
echo "  max_iter=${STAGE2_MAX_ITER}, lr=${STAGE2_LR}"
echo "  GPU=${GPU}"
echo "============================================================"
echo ""

# Ensure viotech symlink exists
if [ ! -d "${WORKSPACE}/hdf5s/viotech" ]; then
    mkdir -p "${WORKSPACE}/hdf5s"
    echo "Creating symlink for viotech dataset..."
    ln -s /mnt/hdd/viotech "${WORKSPACE}/hdf5s/viotech"
fi

python3 pytorch/main_contrast.py train \
    --workspace=$WORKSPACE \
    --logdir="${WORKSPACE}/tb" \
    --pretrain_path="${STAGE1_BEST}" \
    --model_tag $STAGE2_TAG \
    --model_type=$MODEL_TYPE \
    --loss_type=$LOSS_TYPE \
    --augmentation='aug' \
    --max_note_shift=2 \
    --batch_size=$BATCH_SIZE \
    --learning_rate=$STAGE2_LR \
    --reduce_iteration=1000 \
    --resume_iteration=0 \
    --early_stop=$STAGE2_MAX_ITER \
    --patience=0 \
    --device 0 \
    --dataset viotech \
    --contrast_weight=0.0 \
    --ctc_weight=0.0 \
    --num_workers=$NUM_WORKERS \
    --technique_weight=0.0 \
    --technique_moe_weight=0.0 \
    --technique_moe_zone_weight=0.0 \
    --technique_moe_zone_pt_weight=0.0 \
    --technique_frame_moe_weight=$TECHNIQUE_FRAME_MOE_WEIGHT \
    --frame_moe_balance_coeff=$FRAME_MOE_BALANCE_COEFF \
    --fmoe_spectral_expert=$FMOE_SPECTRAL

echo ""
echo "============================================================"
echo "  Two-stage training complete."
echo "  Stage 1 best: ${STAGE1_BEST}"
echo "  Stage 2 checkpoints: ${WORKSPACE}/checkpoints/main_contrast/${STAGE2_TAG}/..."
echo "============================================================"
