import os
import json
import argparse
import pandas as pd
import numpy as np
import h5py
import soundfile as sf
import librosa
from tqdm import tqdm

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

TECHNIQUE_LABEL = {
    "normal": {"tonalTechnique": "none", "articulation": "none"},
    "flageolet": {"tonalTechnique": "harmonics", "articulation": "none"},
    "pizzicato": {"tonalTechnique": "pizzicato", "articulation": "none"},
    "spiccato": {"tonalTechnique": "none", "articulation": "spiccato"},
}

LEGATO_THRESHOLD_SEC = 0.05


def load_mapping(mapping_path):
    if mapping_path and os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            return json.load(f)
    return DEFAULT_MAPPING


def compute_legato(df, threshold):
    """Mark legato=1 for notes whose onset is close to the previous note's offset (bowchange)."""
    legato = np.zeros(len(df), dtype=np.int32)
    onsets = df['onset'].values
    offsets = df['offset'].values
    for i in range(1, len(df)):
        gap = onsets[i] - offsets[i - 1]
        if abs(gap) <= threshold:
            legato[i] = 1
    return legato


def process_file(wav_path, csv_path, technique, output_dir, mapping, target_sr=None,
                  legato_threshold=LEGATO_THRESHOLD_SEC):
    basename = os.path.splitext(os.path.basename(wav_path))[0]

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return None

    required_cols = {'onset', 'offset', 'midi_number'}
    if not required_cols.issubset(df.columns):
        print(f"Missing columns in {csv_path}: {required_cols - set(df.columns)}")
        return None

    try:
        audio, sr = sf.read(wav_path, dtype='int16')
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1).astype(np.int16)

        if target_sr is not None and target_sr != sr:
            audio_float = audio.astype(np.float32) / 32767.0
            audio_float = librosa.resample(audio_float, orig_sr=sr, target_sr=target_sr)
            audio = (audio_float * 32767.0).clip(-32768, 32767).astype(np.int16)
            sr = target_sr
    except Exception as e:
        print(f"Error reading audio {wav_path}: {e}")
        return None

    tech_label = TECHNIQUE_LABEL[technique]
    tech_val = mapping['tonalTechnique'].get(tech_label['tonalTechnique'], 0)
    art_val = mapping['articulation'].get(tech_label['articulation'], 0)

    df = df.sort_values('onset').reset_index(drop=True)
    legato_arr = compute_legato(df, threshold=legato_threshold)

    events = []
    stats = {
        "notes": len(df),
        "tonalTechnique": {tech_label['tonalTechnique']: len(df)},
        "articulation": {tech_label['articulation']: len(df)},
        "legato": int(legato_arr.sum())
    }

    for idx, row in df.iterrows():
        pitch = int(row['midi_number'])
        start = float(row['onset'])
        end = float(row['offset'])
        leg_val = int(legato_arr[idx])

        events.append({
            "time": start,
            "type": "note_on",
            "pitch": pitch,
            "velocity": 80,
            "tech": tech_val,
            "art": art_val,
            "leg": leg_val
        })
        events.append({
            "time": end,
            "type": "note_off",
            "pitch": pitch,
            "velocity": 0,
            "tech": tech_val,
            "art": art_val,
            "leg": leg_val
        })

    events.sort(key=lambda x: x['time'])

    midi_event_strings = []
    midi_event_times = []
    tech_array = []
    art_array = []
    leg_array = []

    for ev in events:
        evt_str = f"{ev['type']} channel=0 note={ev['pitch']} velocity={ev['velocity']} time={ev['time']}"
        midi_event_strings.append(evt_str.encode('ascii'))
        midi_event_times.append(ev['time'])
        tech_array.append(ev['tech'])
        art_array.append(ev['art'])
        leg_array.append(ev['leg'])

    song_title = f"{technique}_{basename}"
    output_path = os.path.join(output_dir, f"{song_title}.h5")

    hash_val = int(hash(song_title)) % 100
    if hash_val < 80:
        split = 'train'
    elif hash_val < 90:
        split = 'validation'
    else:
        split = 'test'

    try:
        with h5py.File(output_path, 'w') as hf:
            hf.create_dataset('midi_event', data=np.array(midi_event_strings, dtype='S100'))
            hf.create_dataset('midi_event_time', data=np.array(midi_event_times, dtype=np.float32))
            hf.create_dataset('waveform', data=audio)
            hf.create_dataset('tonalTechnique', data=np.array(tech_array, dtype=np.int32))
            hf.create_dataset('articulation', data=np.array(art_array, dtype=np.int32))
            hf.create_dataset('legato', data=np.array(leg_array, dtype=np.int32))
            hf.attrs['duration'] = len(audio) / sr
            hf.attrs['sample_rate'] = sr
            hf.attrs['split'] = split.encode('ascii')
    except Exception as e:
        print(f"Error writing H5 {output_path}: {e}")
        return None

    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert MOSA_VPT dataset to H5 for Viotech")
    parser.add_argument("--data_dir", default="/mnt/hdd/MOSA_VPT", help="Root directory of MOSA_VPT")
    parser.add_argument("--output_dir", default="/mnt/hdd/mosavpt_hdf5_only_flageolet", help="Output directory for H5 files")
    parser.add_argument("--mapping_file", default="mapping_config.json", help="Path to JSON mapping file")
    parser.add_argument("--target_sr", type=int, default=16000, help="Target sample rate (default: 16000)")
    parser.add_argument("--legato_threshold", type=float, default=LEGATO_THRESHOLD_SEC,
                        help="Max gap (sec) between consecutive notes to mark as legato/bowchange")
    parser.add_argument("--techniques", nargs='+', default=None,
                        choices=list(TECHNIQUE_LABEL.keys()),
                        help="Technique folders to process (default: all)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    mapping = load_mapping(args.mapping_file)

    wav_dir = os.path.join(args.data_dir, "pt_ssv_4")
    csv_dir = os.path.join(args.data_dir, "csv")
    techniques = args.techniques if args.techniques else list(TECHNIQUE_LABEL.keys())

    total_songs = 0
    total_notes = 0
    total_tech_counts = {}
    total_art_counts = {}
    total_legato = 0

    tasks = []
    for tech in techniques:
        tech_wav_dir = os.path.join(wav_dir, tech)
        if not os.path.isdir(tech_wav_dir):
            print(f"Warning: technique folder not found: {tech_wav_dir}")
            continue
        for wav_name in sorted(os.listdir(tech_wav_dir)):
            if not wav_name.endswith('.wav'):
                continue
            basename = os.path.splitext(wav_name)[0]
            csv_path = os.path.join(csv_dir, f"{basename}.csv")
            if not os.path.exists(csv_path):
                continue
            wav_path = os.path.join(tech_wav_dir, wav_name)
            tasks.append((wav_path, csv_path, tech))

    print(f"Found {len(tasks)} (technique, wav) pairs across {len(techniques)} techniques")
    print(f"Target sample rate: {args.target_sr}")
    print(f"Legato threshold: {args.legato_threshold}s")

    for wav_path, csv_path, tech in tqdm(tasks):
        stats = process_file(wav_path, csv_path, tech, args.output_dir, mapping,
                             target_sr=args.target_sr, legato_threshold=args.legato_threshold)
        if stats:
            total_songs += 1
            total_notes += stats['notes']
            total_legato += stats['legato']

            for k, v in stats['tonalTechnique'].items():
                total_tech_counts[k] = total_tech_counts.get(k, 0) + v

            for k, v in stats['articulation'].items():
                total_art_counts[k] = total_art_counts.get(k, 0) + v

    print("\n=== Summary ===")
    print(f"Number of files processed: {total_songs}")
    print(f"Total number of notes: {total_notes}")
    print("\nTonal Technique Counts:")
    for k, v in sorted(total_tech_counts.items()):
        print(f"  {k}: {v}")
    print("\nArticulation Counts:")
    for k, v in sorted(total_art_counts.items()):
        print(f"  {k}: {v}")
    print(f"\nLegato (bowchange) Notes: {total_legato}")


if __name__ == "__main__":
    main()


""" RUN
# 只處理 flageolet
python prepare_mosavpt.py --techniques flageolet
# 選多個
python prepare_mosavpt.py --techniques flageolet pizzicato
# 全部（預設行為，不加參數）
python prepare_mosavpt.py
"""