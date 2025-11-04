
# MAESTRO dataset directory. Users need to download MAESTRO dataset into this folder.
DATASET_DIR="/mnt/gestalt/home/tkwang/ViolinMamba/violin-transcription/dataset"

# Modify to your workspace
WORKSPACE="/mnt/gestalt/home/tkwang/ViolinMamba/violin-transcription/dataset"

# Pack audio files to hdf5 format for training
python3 piano_transcription/utils/features.py pack_musc_dataset_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE
