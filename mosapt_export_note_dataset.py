import argparse
import os
import sys
from typing import Any

import h5py
import numpy as np
import soundfile as sf

# Project utils
try:
    sys.path.insert(1, os.path.join(os.path.dirname(__file__), 'piano_transcription', 'utils'))
    from utilities import TargetProcessor, traverse_folder  # type: ignore
    import config as _pt_config  # type: ignore
    SAMPLE_RATE = int(getattr(_pt_config, 'sample_rate', 16000))
    BEGIN_NOTE = int(getattr(_pt_config, 'begin_note', 21))
    CLASSES_NUM = int(getattr(_pt_config, 'classes_num', 88))
    FPS = int(getattr(_pt_config, 'frames_per_second', 100))
except Exception:
    SAMPLE_RATE = 16000
    BEGIN_NOTE = 21
    CLASSES_NUM = 88
    FPS = 100


def _decode_if_bytes(x: Any) -> Any:
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode()
        except Exception:
            return x
    return x


def _to_float_audio(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.int16:
        y = (x.astype(np.float32) / 32768.0)
    elif x.dtype == np.int32:
        y = (x.astype(np.float32) / 2147483648.0)
    else:
        y = x.astype(np.float32)
    m = float(np.max(np.abs(y))) if y.size > 0 else 1.0
    if m > 0:
        y = y / (m + 1e-8)
    return y


def _infer_tech_from_filename(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    parts = base.split('_')
    tech = parts[-1].lower() if parts else 'no_technique'
    if tech not in ['flageolet', 'normal', 'pizzicato', 'spiccato']:
        tech = 'no_technique'
    return tech


def main():
    parser = argparse.ArgumentParser(description='Export per-note WAVs from MOSAPT HDF5s with labels in filename')
    parser.add_argument('--hdf5s_dir', type=str, required=True, help='Directory containing MOSAPT H5 files')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for note WAVs')
    parser.add_argument('--per_file_subdir', action='store_true', help='Put each file\'s notes into subfolders')
    parser.add_argument('--limit', type=int, default=0, help='Export at most N notes (0=all)')
    args = parser.parse_args()

    hdf5s_dir = os.path.expanduser(args.hdf5s_dir)
    if not os.path.isdir(hdf5s_dir):
        raise NotADirectoryError(f'Not a directory: {hdf5s_dir}')

    os.makedirs(args.out_dir, exist_ok=True)

    names, paths = traverse_folder(hdf5s_dir)
    paths = sorted(paths)

    exported = 0
    for p in paths:
        try:
            with h5py.File(p, 'r') as hf:
                if ('midi_event' not in hf) or ('midi_event_time' not in hf) or ('waveform' not in hf):
                    continue

                # Technique from attr if available, else from filename suffix
                tech = _decode_if_bytes(hf.attrs.get('pt_name', ''))
                if tech is None or tech == '':
                    tech = _infer_tech_from_filename(p)
                tech = str(tech).strip().lower()
                if tech == '':
                    tech = 'no_technique'

                midi_events = [_decode_if_bytes(e) for e in hf['midi_event'][:]]
                midi_times = hf['midi_event_time'][:]
                duration_sec = float(hf.attrs['duration']) if 'duration' in hf.attrs else (len(hf['waveform']) / SAMPLE_RATE)

                tp = TargetProcessor(duration_sec, FPS, BEGIN_NOTE, CLASSES_NUM)
                _, note_events, _ = tp.process(0.0, midi_times, midi_events, extend_pedal=True)

                base_key = os.path.splitext(os.path.basename(p))[0]
                out_dir = args.out_dir
                if args.per_file_subdir:
                    out_dir = os.path.join(args.out_dir, base_key)
                    os.makedirs(out_dir, exist_ok=True)

                for i, ev in enumerate(note_events):
                    onset = float(ev['onset_time']); offset = float(ev['offset_time'])
                    if offset <= onset:
                        continue
                    s0 = int(onset * SAMPLE_RATE)
                    s1 = int(offset * SAMPLE_RATE)
                    seg = _to_float_audio(hf['waveform'][s0:s1])
                    fname = f"{base_key}_note_{i:03d}_tech-{tech}_st-{onset:.3f}_dur-{(offset-onset):.3f}.wav"
                    out_path = os.path.join(out_dir, fname)
                    sf.write(out_path, seg, SAMPLE_RATE)
                    exported += 1
                    if args.limit and exported >= args.limit:
                        break
        except Exception:
            continue

        if args.limit and exported >= args.limit:
            break

    print(f"Exported {exported} note WAVs to: {args.out_dir}")


if __name__ == '__main__':
    main()


