# VioPTT: Violin Technique-Aware Transcription from Synthetic Data Augmentation

### Installation
```bash
pip install -r requirements.txt
```

### How to Use

**Note**: Before running the scripts, you need to modify the paths in the scripts according to your setup.

#### Option A: Two-step pipeline
Run transcription and technique classification separately.

```bash
# Step 1. Audio to MIDI
# Edit AUDIO_DIR, OUTPUT_DIR, and CHECKPOINT_PATH in the script before running
bash scripts/run_HRPT_inference.sh

# Step 2. Recognize playing techniques for each note
# Requires audio directory and corresponding MIDI directory from Step 1
# Edit AUDIO_DIR, MIDI_DIR, OUTPUT_DIR, and checkpoint paths in the script before running
bash scripts/run_HRPT_inference_note_tech.sh
```

#### Option B: End-to-end
Transcription and technique classification in a single step — no pre-existing MIDI required.

```bash
# Audio → MIDI + per-note technique labels (CSV) in one pass
# Edit AUDIO_DIR, OUTPUT_DIR, and checkpoint paths in the script before running
bash scripts/run_infer_technique.sh
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

- **scripts/run_infer_technique.sh**: End-to-end transcription + technique classification
  - Takes raw audio as input; no pre-existing MIDI needed
  - Outputs per-note technique labels to CSV and a technique-annotated MIDI file
  - Uses the full model (all transcription features, no ablation)
  - **Configuration required**: Edit the following variables at the top of the script:
    - `AUDIO_DIR`: Directory containing audio files (.wav, .mp3, .flac)
    - `OUTPUT_DIR`: Output directory for CSV and MIDI files
    - `NOTE_MODEL_CHECKPOINT`: Path to note technique model (`checkpoints/note_tech_model.pth`)
    - `TRANSCRIPTOR_CHECKPOINT`: Path to transcription model checkpoint
