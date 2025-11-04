#!/bin/bash

# Auto-detect project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="${PROJECT_ROOT}"

# MOSA dataset directory. Users need to download MOSA dataset into this folder.
DATASET_DIR="${WORKSPACE}/data/MOSA_dataset"

# Pack audio files to hdf5 format for training
python3 "${WORKSPACE}/piano_transcription/utils/features.py" pack_mosa_dataset_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

