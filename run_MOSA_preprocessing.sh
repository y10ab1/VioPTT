
# MAESTRO dataset directory. Users need to download MAESTRO dataset into this folder.
DATASET_DIR="/mnt/gestalt/home/tkwang/MOSA_dataset"

# Modify to your workspace
WORKSPACE="/mnt/gestalt/home/tkwang/MOSA_ver2"

# Pack audio files to hdf5 format for training
python3 piano_transcription/utils/features.py pack_mosa_dataset_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE
