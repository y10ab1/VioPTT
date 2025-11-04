#!/bin/bash

# Auto-detect project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="${PROJECT_ROOT}"

# MUSC dataset directory. Users need to download MUSC dataset into this folder.
DATASET_DIR="${WORKSPACE}/data/MUSC_dataset"

# Pack audio files to hdf5 format for training
python3 "${WORKSPACE}/piano_transcription/utils/features.py" pack_musc_dataset_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

