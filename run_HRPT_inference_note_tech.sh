#!/usr/bin/env bash
set -euo pipefail

# ========== Configuration ==========

WORKSPACE="/home/yuehpo/coding/VioPTT"
# Audio directory containing audio files (.wav, .mp3, .flac)
AUDIO_DIR="/home/yuehpo/data/violin_transcription"

# MIDI directory containing corresponding MIDI files (.mid, .midi)
MIDI_DIR="${WORKSPACE}/output/violin_transcription"

# Output directory for technique CSV files
OUTPUT_DIR="${WORKSPACE}/output/violin_transcription"

# Note-level technique model checkpoint
NOTE_MODEL_CHECKPOINT="${WORKSPACE}/checkpoints/note_tech_model.pth"

# Transcriptor model checkpoint
TRANSCRIPTOR_CHECKPOINT="${WORKSPACE}/checkpoints/transcriptor_model.pth"

# Transcription features to use
TRANS_FEATURES_LIST="reg_onset_output reg_offset_output frame_output velocity_output"

# Debug mode (set to 1 to enable debug output)
DEBUG=0

# ===================================

if [[ ! -d "$AUDIO_DIR" ]]; then
  echo "Audio directory not found: $AUDIO_DIR"
  exit 1
fi

if [[ ! -d "$MIDI_DIR" ]]; then
  echo "MIDI directory not found: $MIDI_DIR"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

declare -A AUDIO_MAP
declare -A MIDI_MAP

# Collect audio files (mp3, wav, flac) recursively in AUDIO_DIR
while IFS= read -r -d '' AUDIO_FILE; do
  BASENAME="$(basename "${AUDIO_FILE%.*}")"
  if [[ -z "${AUDIO_MAP[$BASENAME]+x}" ]]; then
    AUDIO_MAP["$BASENAME"]="$AUDIO_FILE"
  fi
done < <(find -L "$AUDIO_DIR" -type f \( -iname '*.mp3' -o -iname '*.wav' -o -iname '*.flac' \) -print0)

# Collect MIDI files recursively in MIDI_DIR
while IFS= read -r -d '' MIDI_FILE; do
  BASENAME="$(basename "${MIDI_FILE%.*}")"
  if [[ -z "${MIDI_MAP[$BASENAME]+x}" ]]; then
    MIDI_MAP["$BASENAME"]="$MIDI_FILE"
  fi
done < <(find -L "$MIDI_DIR" -type f \( -iname '*.mid' -o -iname '*.midi' \) -print0)

# Debug listing
if [[ $DEBUG -eq 1 ]]; then
  set +u
  echo "Found ${#AUDIO_MAP[@]} audio basenames in $AUDIO_DIR"
  SHOW_N=10
  COUNT_SHOWN=0
  for k in "${!AUDIO_MAP[@]}"; do
    echo "  [audio] $k -> ${AUDIO_MAP[$k]}"
    COUNT_SHOWN=$((COUNT_SHOWN + 1))
    if [[ $COUNT_SHOWN -ge $SHOW_N ]]; then
      break
    fi
  done | sort

  echo "Found ${#MIDI_MAP[@]} MIDI basenames in $MIDI_DIR"
  COUNT_SHOWN=0
  for k in "${!MIDI_MAP[@]}"; do
    echo "  [midi]  $k -> ${MIDI_MAP[$k]}"
    COUNT_SHOWN=$((COUNT_SHOWN + 1))
    if [[ $COUNT_SHOWN -ge $SHOW_N ]]; then
      break
    fi
  done | sort

  # Intersection preview
  INTERSECT_COUNT=0
  echo "Preview of matching pairs (up to $SHOW_N):"
  for BASENAME in "${!AUDIO_MAP[@]}"; do
    if [[ -n "${MIDI_MAP[$BASENAME]+x}" ]]; then
      echo "  [pair]  $BASENAME"
      INTERSECT_COUNT=$((INTERSECT_COUNT + 1))
      if [[ $INTERSECT_COUNT -ge $SHOW_N ]]; then
        break
      fi
    fi
  done | sort
  echo "Total matching basenames: $INTERSECT_COUNT"
  set -u
fi

COUNT=0
for BASENAME in "${!AUDIO_MAP[@]}"; do
  if [[ -n "${MIDI_MAP[$BASENAME]+x}" ]]; then
    AUDIO_PATH="${AUDIO_MAP[$BASENAME]}"
    MIDI_PATH="${MIDI_MAP[$BASENAME]}"
    OUTPUT_CSV="$OUTPUT_DIR/${BASENAME}_techniques.csv"

    echo "Processing: $BASENAME"
    python /home/yuehpo/coding/VioPTT/piano_transcription/pytorch/infer_note_technique_from_midi.py \
      --audio_path "$AUDIO_PATH" \
      --midi_path "$MIDI_PATH" \
      --note_model_checkpoint "$NOTE_MODEL_CHECKPOINT" \
      --output_csv "$OUTPUT_CSV" \
      --use_trans_features \
      --transcriptor_checkpoint "$TRANSCRIPTOR_CHECKPOINT" \
      --trans_features_list $TRANS_FEATURES_LIST \
      --device 0

    COUNT=$((COUNT + 1))
  fi
done

if [[ $COUNT -eq 0 ]]; then
  echo "No matching audio + MIDI pairs found in:"
  echo "  AUDIO_DIR=$AUDIO_DIR"
  echo "  MIDI_DIR=$MIDI_DIR"
fi