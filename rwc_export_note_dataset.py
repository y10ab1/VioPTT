import argparse
import os
import sys
import re
from typing import Any

import h5py
import numpy as np
import soundfile as sf

# Use project sample rate
try:
    sys.path.insert(1, os.path.join(os.path.dirname(__file__), 'piano_transcription', 'utils'))
    import config as _pt_config  # type: ignore
    SAMPLE_RATE = int(getattr(_pt_config, 'sample_rate', 44100))
except Exception:
    SAMPLE_RATE = 44100

SAMPLE_RATE = 44100

def _decode_if_bytes(x: Any) -> Any:
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode()
        except Exception:
            return x
    return x


def _to_float_audio(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.int16:
        return (x.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    if x.dtype == np.int32:
        return (x.astype(np.float32) / 2147483648.0).clip(-1.0, 1.0)
    x = x.astype(np.float32)
    mx = float(np.max(np.abs(x))) if x.size > 0 else 1.0
    if mx > 1.0:
        x = x / (mx + 1e-8)
    return x


def main():
    parser = argparse.ArgumentParser(description='Export per-note WAVs from RWC H5 with labels in filename')
    parser.add_argument('--h5_path', type=str, default='~/data/rwc_new.h5', help='Path to RWC HDF5 file')
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to write note-level WAVs')
    parser.add_argument('--per_file_subdir', action='store_true', help='Group notes by file_key subdirectories')
    parser.add_argument('--limit', type=int, default=0, help='Export at most N notes (0=all)')
    args = parser.parse_args()

    h5_path = os.path.expanduser(args.h5_path)
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f'H5 file not found: {h5_path}')

    os.makedirs(args.out_dir, exist_ok=True)

    count = 0
    with h5py.File(h5_path, 'r') as f:
        file_keys = [k for k in f.keys() if k.startswith('file_')]
        for file_key in file_keys:
            g = f[file_key]
            tech = _decode_if_bytes(g.attrs.get('pt_name', '')) or _decode_if_bytes(g.attrs.get('technique', ''))
            tech = str(tech).strip().lower() if tech is not None else 'no_technique'
            if tech == '':
                tech = 'no_technique'

            if 'notes' not in g:
                continue
            notes_g = g['notes']

            for note_key in notes_g.keys():
                if not note_key.startswith('note_'):
                    continue
                ng = notes_g[note_key]

                if 'waveform' not in ng:
                    continue
                wav = _to_float_audio(ng['waveform'][:])
                start_time = float(_decode_if_bytes(ng.attrs.get('start_time', 0.0)))
                duration = float(_decode_if_bytes(ng.attrs.get('duration', 0.0)))

                if args.per_file_subdir:
                    out_dir = os.path.join(args.out_dir, file_key)
                    os.makedirs(out_dir, exist_ok=True)
                else:
                    out_dir = args.out_dir

                # Filename encodes labels
                fname = f"{file_key}_{note_key}_tech-{tech}_st-{start_time:.3f}_dur-{duration:.3f}.wav"
                out_path = os.path.join(out_dir, fname)
                sf.write(out_path, wav, SAMPLE_RATE)

                count += 1
                if args.limit and count >= args.limit:
                    break
            if args.limit and count >= args.limit:
                break

    print(f"Exported {count} note WAVs to: {args.out_dir}")


if __name__ == '__main__':
    main()


