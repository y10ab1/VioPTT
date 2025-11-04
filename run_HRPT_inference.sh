
# Inference High Resolution Piano Transcription (HRPT)

# Select inference model path
# Model checkpoint path
WORKSPACE="/home/yuehpo/coding/VioPTT"

CHECKPOINT_PATH="${WORKSPACE}/checkpoints/transcriptor_model.pth"

# Audio file path or directory that are going to be transcribed
AUDIO_DIR="/home/yuehpo/data/violin_transcription"

# Transcribed MIDI output directory
OUTPUT_DIR="${WORKSPACE}/output/violin_transcription"

# Model type
MODEL_TYPE="Regress_onset_offset_frame_velocity_CRNN" # Don't need to change

# Post processor type
POST_PROCESSOR_TYPE="regression"  # Don't need to change





cd piano_transcription

# Function to process a single audio file
process_audio_file() {
    local audio_file="$1"
    
    python3 pytorch/inference.py \
        --model_type='Regress_onset_offset_frame_velocity_CRNN' \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --post_processor_type "$POST_PROCESSOR_TYPE" \
        --audio_path "$audio_file" \
        --device 0
    
    echo "Completed: $audio_file"
    echo "----------------------------------------"
}

# Check if AUDIO_DIR is a directory or a file
if [ -d "$AUDIO_DIR" ]; then
    echo "Processing directory: $AUDIO_DIR"
    echo "Looking for .wav and .mp3 files..."
    
    # Find all .flac, .wav and .mp3 files in the directory (case insensitive)
    audio_files=$(find "$AUDIO_DIR" -type f \( -iname "*.flac" -o -iname "*.wav" -o -iname "*.mp3" \))
    
    if [ -z "$audio_files" ]; then
        echo "No .wav or .mp3 files found in directory: $AUDIO_DIR"
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
    
elif [ -f "$AUDIO_DIR" ]; then
    echo "Processing single file: $AUDIO_DIR"
    process_audio_file "$AUDIO_DIR"
    
else
    echo "Error: $AUDIO_DIR is neither a valid file nor directory"
    exit 1
fi
