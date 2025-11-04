# VioPTT: Violin Technique-Aware Transcription from Synthetic Data Augmentation

### Installation
```bash
pip install -r requirements.txt
```

### How to Use

**Note**: Before running the scripts, you need to modify the paths in the scripts according to your setup.

```bash
# 1. Audio to MIDI
# Convert audio files to MIDI format
# Edit AUDIO_DIR, OUTPUT_DIR, and CHECKPOINT_PATH in the script before running
bash scripts/run_HRPT_inference.sh

# 2. Recognize playing techniques for each note
# Recognize playing techniques for each note based on audio and MIDI files
# Edit AUDIO_DIR, MIDI_DIR, OUTPUT_DIR, and checkpoint paths in the script before running
bash scripts/run_HRPT_inference_note_tech.sh
```

### Description
- **scripts/run_HRPT_inference.sh**: Convert audio files to MIDI format
  - Supports .wav, .mp3, .flac formats
  - Can process single file or entire directory
  - **Configuration required**: Edit the following variables at the top of the script:
    - `AUDIO_DIR`: Path to audio file or directory
    - `OUTPUT_DIR`: Output directory for MIDI files
    - `CHECKPOINT_PATH`: Path to transcription model checkpoint
  
- **scripts/run_HRPT_inference_note_tech.sh**: Recognize playing techniques for each note
  - Requires audio directory and corresponding MIDI directory
  - Outputs playing techniques for each note to CSV file
  - **Configuration required**: Edit the following variables at the top of the script:
    - `AUDIO_DIR`: Directory containing audio files
    - `MIDI_DIR`: Directory containing corresponding MIDI files
    - `OUTPUT_DIR`: Output directory for technique CSV files
    - `NOTE_MODEL_CHECKPOINT`: Path to note technique model
    - `TRANSCRIPTOR_CHECKPOINT`: Path to transcriptor model
