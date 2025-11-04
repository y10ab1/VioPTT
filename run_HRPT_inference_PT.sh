# CHECKPOINT_PATH="/ssd2/tk/ViolinMambaData/checkpoints/main_contrast/mix_aug_ns2_b8_cosine_alltech_lr5e-4_pretrained=True_technique=1.0/Regress_onset_offset_frame_velocity_technique_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=8/20000_iterations.pth"
# CHECKPOINT_PATH="/ssd2/tk/ViolinMambaData/checkpoints/main_contrast/feature_mix_aug_ns2_b4_cosine_alltech_lr5e-4_pretrained=True_technique=1.0/Regress_onset_offset_frame_velocity_technique_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=4/20000_iterations.pth"
# CHECKPOINT_PATH="/home/yuehpo/coding/violin-mamba/checkpoints/main_contrast/w_aug_mosapt_train_from_scratch_w_technique_mixed_dataset/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=5/10000_iterations.pth"
# CHECKPOINT_PATH="/home/yuehpo/coding/violin-mamba/checkpoints/main_contrast/w_aug_w_technique_annotation_mixed_dataset_new_ssv_local_technique_feature_10frames_volume_normalized/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=5/18000_iterations.pth"
CHECKPOINT_PATH="/home/yuehpo/coding/violin-mamba/checkpoints/main_contrast/w_aug_w_technique_annotation_mixed_dataset_mosapt_ssv_local_technique_feature_10frames_volume_normalized_per_class_acc_0906/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=5/15000_iterations.pth"

# Audio file path (or directory!!!!) that are going to be transcribed
# AUDIO_PATH="/home/tkwang/violin-mamba/test_samples/Augustin Hadelich plays Bach Giga from Partita No. 2 Live (2021).mp3"
AUDIO_PATH="/home/yuehpo/coding/violin-mamba/Arthur_Grumiaux_Bach-Sonatas_and_Partitas_for_violin_2_05.mp3"
# AUDIO_PATH="/home/yuehpo/coding/violin-mamba/Augustin Hadelich plays Bach Giga from Partita No. 2 Live (2021).mp3"

# Transcribed MIDI output directory
OUTPUT_DIR="/home/yuehpo/coding/violin-mamba/output/test_results_w_technique_annotation_mixed_dataset_new_ssv_local_technique_feature_10frames_volume_normalized/"

# Model type
MODEL_TYPE="Regress_onset_offset_frame_velocity_CRNN"

# Post processor type
POST_PROCESSOR_TYPE="regression"  # Don't need to change

cd piano_transcription

# Function to process a single audio file
process_audio_file() {
    local audio_file="$1"
    
    python3 pytorch/inference_pt.py \
        --model_type="$MODEL_TYPE" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --post_processor_type "$POST_PROCESSOR_TYPE" \
        --audio_path "$audio_file" \
        --device 0
    
    echo "Completed: $audio_file"
    echo "----------------------------------------"
}

# Check if AUDIO_PATH is a directory or a file
if [ -d "$AUDIO_PATH" ]; then
    echo "Processing directory: $AUDIO_PATH"
    echo "Looking for .wav and .mp3 files..."
    
    # Find all .wav and .mp3 files in the directory (case insensitive)
    audio_files=$(find "$AUDIO_PATH" -type f \( -iname "*.wav" -o -iname "*.mp3" \))
    
    if [ -z "$audio_files" ]; then
        echo "No .wav or .mp3 files found in directory: $AUDIO_PATH"
        exit 1
    fi
    
    # Count total files
    total_files=$(echo "$audio_files" | wc -l)
    echo "Found $total_files audio files (.wav/.mp3) to process"
    echo "----------------------------------------"
    
    # Process each audio file
    current_file=0
    while IFS= read -r audio_file; do
        current_file=$((current_file + 1))
        echo "[$current_file/$total_files] Processing: $audio_file"
        process_audio_file "$audio_file"
    done <<< "$audio_files"
    
    echo "All files processed successfully!"
    
elif [ -f "$AUDIO_PATH" ]; then
    echo "Processing single file: $AUDIO_PATH"
    process_audio_file "$AUDIO_PATH"
    
else
    echo "Error: $AUDIO_PATH is neither a valid file nor directory"
    exit 1
fi
