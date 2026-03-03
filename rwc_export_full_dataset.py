import argparse
import os
import sys
import h5py
import numpy as np
import soundfile as sf
from typing import Any

# Use project utilities for consistency if needed, but keeping it standalone for simplicity
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
    
    # If already float, ensure range
    x = x.astype(np.float32)
    mx = float(np.max(np.abs(x))) if x.size > 0 else 1.0
    if mx > 1.0:
        x = x / (mx + 1e-8)
    return x

def main():
    parser = argparse.ArgumentParser(description='Export full-track WAVs from RWC H5')
    parser.add_argument('--h5_path', type=str, default='~/data/rwc_new.h5', help='Path to RWC HDF5 file')
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to write full WAVs')
    args = parser.parse_args()

    h5_path = os.path.expanduser(args.h5_path)
    if not os.path.exists(h5_path):
        print(f"Error: H5 file not found at {h5_path}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    with h5py.File(h5_path, 'r') as f:
        file_keys = sorted([k for k in f.keys() if k.startswith('file_')])
        print(f"Found {len(file_keys)} tracks in H5. Starting export...")

        for file_key in file_keys:
            g = f[file_key]
            if 'full_waveform' not in g:
                print(f"Skipping {file_key}: 'full_waveform' dataset not found.")
                continue

            # Get sample rate and filename from attributes
            sr = g.attrs.get('sample_rate', 44100)
            original_filename = _decode_if_bytes(g.attrs.get('filename', ''))
            
            if original_filename:
                # Use the original filename without extension
                base_name = os.path.splitext(os.path.basename(original_filename))[0]
            else:
                base_name = file_key

            # Extract and convert audio
            print(f"Processing {file_key} ({base_name})...", end='', flush=True)
            wav = _to_float_audio(g['full_waveform'][:])
            
            out_path = os.path.join(args.out_dir, f"{base_name}.wav")
            
            # Ensure unique filename if multiple tracks have the same base_name
            if os.path.exists(out_path):
                out_path = os.path.join(args.out_dir, f"{base_name}_{file_key}.wav")
                
            sf.write(out_path, wav, int(sr))
            print(f" Done -> {os.path.basename(out_path)}")

    print(f"\nSuccessfully exported tracks to: {args.out_dir}")

if __name__ == '__main__':
    main()

