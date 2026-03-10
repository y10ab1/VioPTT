import os
import json
import glob
import argparse
import pandas as pd
import numpy as np
import h5py
import soundfile as sf
import librosa
from tqdm import tqdm

# Default Mappings
DEFAULT_MAPPING = {
    "tonalTechnique": {
        "": 0,
        "none": 0,
        "pizzicato": 1,
        "harmonics": 2,
        "openstring": 3
    },
    "articulation": {
        "": 0,
        "none": 0,
        "release": 1,
        "staccato": 2,
        "spiccato": 3
    }
}

def load_mapping(mapping_path):
    if mapping_path and os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            return json.load(f)
    return DEFAULT_MAPPING

def get_mapped_value(value, mapping):
    if pd.isna(value):
        value = ""
    value = str(value).strip().lower()
    return mapping.get(value, 0)

def process_song(song_folder, output_dir, mapping, target_sr=None):
    # Check status.json
    status_path = os.path.join(song_folder, "status.json")
    if not os.path.exists(status_path):
        return None
    
    try:
        with open(status_path, 'r') as f:
            status = json.load(f)
            if not status.get("completed", False):
                return None
    except Exception:
        return None

    # Find required files
    wav_files = glob.glob(os.path.join(song_folder, "*_cut.wav"))
    csv_path = os.path.join(song_folder, "annotation_revised.csv")

    if not wav_files or not os.path.exists(csv_path):
        return None
    
    wav_path = wav_files[0] # Use the first one found
    
    # Read CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return None

    # Read Audio
    try:
        audio, sr = sf.read(wav_path, dtype='int16')
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1).astype(np.int16)

        # Resample to target_sr if specified and different from native sr
        if target_sr is not None and target_sr != sr:
            audio_float = audio.astype(np.float32) / 32767.0
            audio_float = librosa.resample(audio_float, orig_sr=sr, target_sr=target_sr)
            audio = (audio_float * 32767.0).clip(-32768, 32767).astype(np.int16)
            sr = target_sr
    except Exception as e:
        print(f"Error reading audio {wav_path}: {e}")
        return None

    # Generate Events
    events = []
    
    # Statistics for this song
    stats = {
        "notes": len(df),
        "tonalTechnique": {},
        "articulation": {},
        "legato": 0
    }

    for _, row in df.iterrows():
        pitch = int(row['pitch'])
        start = float(row['start'])
        end = float(row['end'])
        
        # Features
        tech_val = get_mapped_value(row.get('tonalTechnique', ''), mapping['tonalTechnique'])
        art_val = get_mapped_value(row.get('articulation', ''), mapping['articulation'])
        
        leg_val = row.get('legato', 0)
        if pd.isna(leg_val):
            leg_val = 0
        leg_val = int(leg_val)

        # Update stats
        tech_val_raw = row.get('tonalTechnique')
        tech_key = 'none' if pd.isna(tech_val_raw) else str(tech_val_raw).strip().lower() or 'none'
        stats["tonalTechnique"][tech_key] = stats["tonalTechnique"].get(tech_key, 0) + 1
        
        art_val_raw = row.get('articulation')
        art_key = 'none' if pd.isna(art_val_raw) else str(art_val_raw).strip().lower() or 'none'
        stats["articulation"][art_key] = stats["articulation"].get(art_key, 0) + 1

        if leg_val == 1:
            stats["legato"] += 1

        # Create Note On/Off events
        # Note On
        events.append({
            "time": start,
            "type": "note_on",
            "pitch": pitch,
            "velocity": 80,
            "tech": tech_val,
            "art": art_val,
            "leg": leg_val
        })
        # Note Off
        events.append({
            "time": end,
            "type": "note_off",
            "pitch": pitch,
            "velocity": 0,
            "tech": tech_val,
            "art": art_val,
            "leg": leg_val
        })

    # Sort events by time
    events.sort(key=lambda x: x['time'])

    # Prepare arrays for H5
    midi_event_strings = []
    midi_event_times = []
    tech_array = []
    art_array = []
    leg_array = []

    for ev in events:
        # Format: note_on channel=0 note=62 velocity=80 time=2.72...
        evt_str = f"{ev['type']} channel=0 note={ev['pitch']} velocity={ev['velocity']} time={ev['time']}"
        midi_event_strings.append(evt_str.encode('ascii')) # H5 requires bytes for fixed-length strings usually, or use special dtype
        midi_event_times.append(ev['time'])
        tech_array.append(ev['tech'])
        art_array.append(ev['art'])
        leg_array.append(ev['leg'])

    # Create H5
    song_title = os.path.basename(song_folder)
    output_path = os.path.join(output_dir, f"{song_title}.h5")
    
    # Deterministic split based on song name hash
    hash_val = int(hash(song_title)) % 100
    if hash_val < 80:
        split = 'train'
    elif hash_val < 90:
        split = 'validation'
    else:
        split = 'test'

    try:
        with h5py.File(output_path, 'w') as hf:
            # midi_event
            # Use a fixed length string dtype or variable. The previous file used |S100.
            dt = h5py.special_dtype(vlen=str) 
            # Actually previous file was |S100. Let's try to match or just use numpy S100.
            # np.array(..., dtype='S100')
            hf.create_dataset('midi_event', data=np.array(midi_event_strings, dtype='S100'))
            
            hf.create_dataset('midi_event_time', data=np.array(midi_event_times, dtype=np.float32))
            hf.create_dataset('waveform', data=audio)
            
            # New datasets
            hf.create_dataset('tonalTechnique', data=np.array(tech_array, dtype=np.int32))
            hf.create_dataset('articulation', data=np.array(art_array, dtype=np.int32))
            hf.create_dataset('legato', data=np.array(leg_array, dtype=np.int32))
            
            # Attributes
            hf.attrs['duration'] = len(audio) / sr
            hf.attrs['sample_rate'] = sr
            hf.attrs['split'] = split.encode('ascii')

    except Exception as e:
        print(f"Error writing H5 {output_path}: {e}")
        return None

    return stats

def main():
    parser = argparse.ArgumentParser(description="Convert raw dataset to H5 for Viotech")
    parser.add_argument("--data_dir", default="/mnt/hdd/audio-midi-marker/data", help="Root directory of raw data")
    parser.add_argument("--output_dir", default="/mnt/hdd/viotech", help="Output directory for H5 files")
    parser.add_argument("--mapping_file", default="mapping_config.json", help="Path to JSON mapping file")
    parser.add_argument("--target_sr", type=int, default=16000, help="Target sample rate for resampling (default: 16000)")
    
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    mapping = load_mapping(args.mapping_file)
    
    # Traverse folders
    song_folders = [f.path for f in os.scandir(args.data_dir) if f.is_dir()]
    
    total_songs = 0
    total_notes = 0
    total_tech_counts = {}
    total_art_counts = {}
    total_legato = 0

    print(f"Scanning {len(song_folders)} folders in {args.data_dir}...")
    print(f"Target sample rate: {args.target_sr}")

    for folder in tqdm(song_folders):
        stats = process_song(folder, args.output_dir, mapping, target_sr=args.target_sr)
        if stats:
            total_songs += 1
            total_notes += stats['notes']
            total_legato += stats['legato']
            
            for k, v in stats['tonalTechnique'].items():
                total_tech_counts[k] = total_tech_counts.get(k, 0) + v
            
            for k, v in stats['articulation'].items():
                total_art_counts[k] = total_art_counts.get(k, 0) + v

    print("\n=== Summary ===")
    print(f"Number of songs processed: {total_songs}")
    print(f"Total number of notes: {total_notes}")
    print("\nTonal Technique Counts:")
    for k, v in total_tech_counts.items():
        print(f"  {k}: {v}")
    
    print("\nArticulation Counts:")
    for k, v in total_art_counts.items():
        print(f"  {k}: {v}")
        
    print(f"\nLegato Notes: {total_legato}")

if __name__ == "__main__":
    main()
