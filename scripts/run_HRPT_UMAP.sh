#!/bin/bash

# Auto-detect project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="${PROJECT_ROOT}"

# CHECKPOINT_PATH="${WORKSPACE}/checkpoints/main_note_technique/note_level_tech_no_transcription_features_fold_2_0915_3CV_with_transcription_features_all/660_iterations.pth"
TRANSCRIPTOR_CHECKPOINT_PATH="${WORKSPACE}/checkpoints/transcriptor_model.pth"



python ${WORKSPACE}/piano_transcription/pytorch/visualize_umap_note_technique_3fold.py \
  --rwc_notes_root ${WORKSPACE}/hdf5s/rwc/ \
  --checkpoints \
    ${WORKSPACE}/checkpoints/note_tech_model_fold_0.pth \
    ${WORKSPACE}/checkpoints/note_tech_model_fold_1.pth \
    ${WORKSPACE}/checkpoints/note_tech_model_fold_2.pth \
  --fold_ids 0 1 2 \
  --split test \
  --transcriptor_checkpoint $TRANSCRIPTOR_CHECKPOINT_PATH \
  --use_trans_features \
  --plot_confusion \
  --ignore_confusion_class no_technique \
  --n_neighbors 10 --min_dist 0.05 \
  --out_dir ${WORKSPACE}/umap

