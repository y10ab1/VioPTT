# VioPTT: Violin Technique-Aware Transcription from Synthetic Data Augmentation

### How to Use

```bash
# 1. Audio to MIDI
# Convert audio files to MIDI format
bash run_HRPT_inference.sh

# 2. Recognize playing techniques for each note
# Recognize playing techniques for each note based on audio and MIDI files
bash run_HRPT_inference_PT_note.sh /path/to/audio_dir /path/to/midi_dir
```

### Description
- **run_HRPT_inference.sh**: Convert audio files to MIDI format
  - Supports .wav, .mp3, .flac formats
  - Can process single file or entire directory
  
- **run_HRPT_inference_PT_note.sh**: Recognize playing techniques for each note
  - Requires audio directory and corresponding MIDI directory
  - Outputs playing techniques for each note to CSV file
## Dataset
### MOSA Dataset Directory Structure

- ev/
  - ba1/
    - ev01/
      - t1/
        - annotation/
          - annotations/
            - ba1_ev01_t1_align_notetime.csv
            - ba1_ev01_t1_note.csv
        - ba1_ev01_t1_audio.wav
      - t2/
  - ev02/
  - ev03/
  - ev04/
  - ev05/
  - ba3/
  - ba4/
  - be4/
  - be8/
  - de1/
  - de2/
  - el1/
  - me4/
  - mo4/
  - mo5/
- yv/
  - ba1/
  - ba3/
  - ba4/
