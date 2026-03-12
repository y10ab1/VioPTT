#!/usr/bin/env bash
set -euo pipefail

# Auto-detect project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="${PROJECT_ROOT}"

# ========== Configuration ==========

# Audio directory containing audio files (.wav, .mp3, .flac)
AUDIO_DIR="/mnt/hdd/ysaye"

# Mapping config (technique name -> index)
MAPPING_CONFIG="${WORKSPACE}/mapping_config.json"

# Model checkpoint (Regress_onset_offset_frame_velocity_CRNN with technique head)
# CHECKPOINT_PATH="${WORKSPACE}/checkpoints/transcriptor_model.pth"
CHECKPOINT_PATH="/root/VioPTT/checkpoints/main_contrast/vioptt_viotech_v0.1_wtech_parallel_b4_tech0.1_legatoImproved/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=4/2000_iterations.pth"

# Extract iteration number from CHECKPOINT_PATH, e.g., .../2000_iterations.pth -> 2000
ITERATION_NUMBER=$(basename "$CHECKPOINT_PATH" | grep -oP '^\d+(?=_iterations\.pth$)')
# If not found, break!
if [[ -z "$ITERATION_NUMBER" ]]; then
  echo "Iteration number not found in checkpoint path: $CHECKPOINT_PATH"
  exit 1
fi

# Output directory for MIDI, CSV, and visualization PNGs
OUTPUT_DIR="$(dirname "$CHECKPOINT_PATH")/inference_${ITERATION_NUMBER}"


# GPU device index (-1 for CPU)
DEVICE=0

# Model type
MODEL_TYPE="Regress_onset_offset_frame_velocity_CRNN"

# ===================================

if [[ ! -d "$AUDIO_DIR" ]]; then
  echo "Audio directory not found: $AUDIO_DIR"
  exit 1
fi

if [[ ! -f "$MAPPING_CONFIG" ]]; then
  echo "Mapping config not found: $MAPPING_CONFIG"
  exit 1
fi

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Checkpoint not found: $CHECKPOINT_PATH"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "VioTech Inference"
echo "========================================="
echo "Audio dir:      $AUDIO_DIR"
echo "Mapping config: $MAPPING_CONFIG"
echo "Checkpoint:     $CHECKPOINT_PATH"
echo "Output dir:     $OUTPUT_DIR"
echo "Device:         $DEVICE"
echo "========================================="

python "${WORKSPACE}/piano_transcription/pytorch/infer_viotech.py" \
  --audio_dir "$AUDIO_DIR" \
  --mapping_config "$MAPPING_CONFIG" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --model_type "$MODEL_TYPE"

echo "========================================="
echo "Done! Results saved to: $OUTPUT_DIR"
echo "  - .mid   : MIDI with technique-separated tracks"
echo "  - .csv   : Per-note technique labels"
echo "  - .png   : Piano roll + technique prediction visualization"
echo "========================================="
