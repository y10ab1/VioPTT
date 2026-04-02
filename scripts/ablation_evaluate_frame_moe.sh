#!/bin/bash
#
# Evaluate a Frame-MoE ablation checkpoint.
#
# Usage:
#   bash scripts/ablation_evaluate_frame_moe.sh <experiment_id> [checkpoint] [device]
#
# If checkpoint is omitted, automatically finds the latest checkpoint under
# checkpoints/main_contrast/ablation_fmoe_<tag>/...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="${PROJECT_ROOT}"

EXPERIMENT="${1:?Usage: $0 <experiment_id> [checkpoint] [device]}"
CHECKPOINT="${2:-}"
DEVICE="${3:-0}"

RWC_H5="/mnt/hdd/rwc_processed_data.h5"
LABEL_CFG="${WORKSPACE}/config/technique_label_config_reduced.json"
HDF5S_DIR="${WORKSPACE}/hdf5s/viotech"
SPLIT="validation"

# Map experiment ID → model tag + config flags
EXPERT_MASK=""
SPECTRAL=0
TOP_K=""
SHARED_GATE=0
UNIFORM=0

case "$EXPERIMENT" in
    baseline)       TAG="ablation_fmoe_baseline_e012";         EXPERT_MASK="0,1,2" ;;
    full)           TAG="ablation_fmoe_full_e0123_spectral";   EXPERT_MASK="0,1,2,3"; SPECTRAL=1 ;;
    no_onset)       TAG="ablation_fmoe_no_onset_e12";          EXPERT_MASK="1,2" ;;
    no_note)        TAG="ablation_fmoe_no_note_e02";           EXPERT_MASK="0,2" ;;
    no_phrase)      TAG="ablation_fmoe_no_phrase_e01";         EXPERT_MASK="0,1" ;;
    only_onset)     TAG="ablation_fmoe_only_onset_e0";         EXPERT_MASK="0" ;;
    only_note)      TAG="ablation_fmoe_only_note_e1";          EXPERT_MASK="1" ;;
    only_phrase)    TAG="ablation_fmoe_only_phrase_e2";         EXPERT_MASK="2" ;;
    topk1)          TAG="ablation_fmoe_topk1_e012";            EXPERT_MASK="0,1,2"; TOP_K=1 ;;
    dense)          TAG="ablation_fmoe_dense_e012";            EXPERT_MASK="0,1,2"; TOP_K=0 ;;
    uniform)        TAG="ablation_fmoe_uniform_e012";          EXPERT_MASK="0,1,2"; UNIFORM=1 ;;
    shared_gate)    TAG="ablation_fmoe_shared_gate_e012";      EXPERT_MASK="0,1,2"; SHARED_GATE=1 ;;
    bal_none)       TAG="ablation_fmoe_bal0_e012";             EXPERT_MASK="0,1,2" ;;
    bal_high)       TAG="ablation_fmoe_bal001_e012";           EXPERT_MASK="0,1,2" ;;
    bal_vhigh)      TAG="ablation_fmoe_bal01_e012";            EXPERT_MASK="0,1,2" ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        exit 1 ;;
esac

# Auto-find checkpoint if not given
if [ -z "$CHECKPOINT" ]; then
    CKPT_DIR="${WORKSPACE}/checkpoints/main_contrast/${TAG}/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=4"
    if [ -d "$CKPT_DIR" ]; then
        CHECKPOINT=$(ls -t "$CKPT_DIR"/*_iterations.pth 2>/dev/null | head -1)
    fi
    if [ -z "$CHECKPOINT" ]; then
        echo "No checkpoint found for ${TAG} under ${CKPT_DIR}"
        echo "Usage: $0 <experiment_id> <checkpoint_path> [device]"
        exit 1
    fi
fi

echo "============================================"
echo "  Ablation Evaluation: ${EXPERIMENT}"
echo "============================================"
echo "  Tag        : ${TAG}"
echo "  Checkpoint : ${CHECKPOINT}"
echo "  Expert mask: ${EXPERT_MASK}"
echo "  Spectral   : ${SPECTRAL}"
echo "  Device     : ${DEVICE}"
echo "============================================"
echo ""

EXTRA_ARGS=""
[ -n "$EXPERT_MASK" ]  && EXTRA_ARGS+=" --fmoe_expert_mask=${EXPERT_MASK}"
[ -n "$TOP_K" ]        && EXTRA_ARGS+=" --fmoe_top_k=${TOP_K}"

RWC_ARGS=""
if [ -f "$RWC_H5" ]; then
    RWC_ARGS="--rwc_h5_path $RWC_H5 --rwc_split all"
fi

cd "${WORKSPACE}/piano_transcription"

python3 pytorch/evaluate_frame_moe.py \
    --checkpoint_path "$CHECKPOINT" \
    --hdf5s_dir "$HDF5S_DIR" \
    --split "$SPLIT" \
    --device "$DEVICE" \
    --batch_size 4 \
    --num_workers 4 \
    --max_iterations 500 \
    --fmoe_spectral_expert=${SPECTRAL} \
    --fmoe_shared_gate=${SHARED_GATE} \
    --fmoe_uniform_routing=${UNIFORM} \
    $EXTRA_ARGS \
    $RWC_ARGS
