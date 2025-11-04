import os
import sys
import argparse
import soundfile as sf

# Add utils to path
UTILS_DIR = os.path.join(os.path.dirname(__file__), 'piano_transcription', 'utils')
sys.path.insert(1, UTILS_DIR)

from data_generator import MOSAPTNoteDataset  # type: ignore
from utilities import traverse_folder  # type: ignore
import config  # type: ignore


def main():
    parser = argparse.ArgumentParser(description='Test MOSAPTNoteDataset (online note slicing)')
    parser.add_argument('--root', type=str, required=True, help='Directory containing MOSAPT HDF5 files')
    parser.add_argument('--num', type=int, default=5, help='Number of samples to preview')
    parser.add_argument('--out', type=str, required=True, help='Output folder to save preview wavs')
    parser.add_argument('--fixed_seconds', type=float, default=2.0, help='Fixed length (seconds) for each sliced note (centered). If <=0 use exact note duration')
    parser.add_argument('--fps', type=int, default=config.frames_per_second, help='Frames per second for alignment')
    parser.add_argument('--mini_data', action='store_true', default=True, help='Scan fewer files/notes for quick test')
    parser.add_argument('--max_files', type=int, default=1, help='Limit number of H5 files to scan when mini_data is set')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    fixed_seconds = None if (args.fixed_seconds is not None and args.fixed_seconds <= 0) else args.fixed_seconds

    # If mini_data: restrict to first N files to avoid long scans
    allowed_file_keys = None
    if args.mini_data:
        names, paths = traverse_folder(args.root)
        # Sort for determinism
        paths = sorted(paths)[: max(1, args.max_files)]
        allowed_file_keys = [os.path.splitext(os.path.basename(p))[0] for p in paths]

    ds = MOSAPTNoteDataset(
        hdf5s_dir=args.root,
        frames_per_second=args.fps,
        fixed_seconds=fixed_seconds,
        augmentor=None,
        split='train',
        allowed_file_keys=allowed_file_keys,
        mini_data=args.mini_data,
    )

    if len(ds) == 0:
        print(f"Dataset size: 0. No notes found under {args.root}.")
        return

    n = min(args.num, len(ds))
    print(f"Dataset size: {len(ds)}. Previewing {n} samples.")

    for i in range(n):
        item = ds[i]
        wav = item['waveform']
        tech = item['technique'][0]  # (C,)
        cls_idx = int(tech.argmax())
        cls_name = ds.technique_classes[cls_idx] if hasattr(ds, 'technique_classes') else str(cls_idx)

        # Pull source info from internal index
        try:
            src_h5, onset_s, offset_s, tech_name = ds.items[i]
        except Exception:
            src_h5, onset_s, offset_s, tech_name = ('<unknown>', 0.0, 0.0, cls_name)

        print(f"#{i}: src={src_h5}")
        print(f"    onset={onset_s:.3f}s offset={offset_s:.3f}s dur={max(0.0, offset_s - onset_s):.3f}s fixed_seconds={fixed_seconds}")
        print(f"    waveform.shape={wav.shape}, sr={config.sample_rate}")
        print(f"    technique one-hot={tech.tolist()} -> {cls_name}")

        # Save preview wav
        base = os.path.splitext(os.path.basename(src_h5))[0]
        out_path = os.path.join(args.out, f"preview_{i}_{base}_on-{onset_s:.3f}_off-{offset_s:.3f}_tech-{cls_name}.wav")
        sf.write(out_path, wav, config.sample_rate)
        print(f"    saved: {out_path}")


if __name__ == '__main__':
    main()



