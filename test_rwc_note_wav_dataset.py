import os
import sys
import argparse
import soundfile as sf

# Add utils to path
UTILS_DIR = os.path.join(os.path.dirname(__file__), 'piano_transcription', 'utils')
sys.path.insert(1, UTILS_DIR)

from data_generator import RWCTechNoteWavDataset  # type: ignore
import config  # type: ignore


def main():
    parser = argparse.ArgumentParser(description='Test RWCTechNoteWavDataset')
    parser.add_argument('--root', type=str, required=True, help='Root folder of exported note WAVs')
    parser.add_argument('--num', type=int, default=5, help='Number of samples to preview')
    parser.add_argument('--out', type=str, required=True, help='Output folder to save preview wavs')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    ds = RWCTechNoteWavDataset(args.root, frames_per_second=config.frames_per_second)
    n = min(args.num, len(ds))
    print(f"Dataset size: {len(ds)}. Previewing {n} samples.")

    for i in range(n):
        item = ds[i]
        wav = item['waveform']
        tech = item['technique'][0]  # (C,)
        cls_idx = int(tech.argmax())
        cls_name = ds.technique_classes[cls_idx] if hasattr(ds, 'technique_classes') else str(cls_idx)
        src_path = ds.paths[i]
        print(f"#{i}: path={src_path}")
        print(f"    waveform.shape={wav.shape}, sr={config.sample_rate}")
        print(f"    technique one-hot={tech.tolist()} -> {cls_name}")

        # Save preview wav
        base = os.path.basename(src_path)
        out_path = os.path.join(args.out, f"preview_{i}_{base}")
        sf.write(out_path, wav, config.sample_rate)
        print(f"    saved: {out_path}")


if __name__ == '__main__':
    main()


