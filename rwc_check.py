import argparse
import os
import sys
import re
from typing import Any, Dict, Union

import h5py
import numpy as np
import soundfile as sf

# Try to use project config sample rate; fallback to 16000
try:
    sys.path.insert(1, os.path.join(os.path.dirname(__file__), 'piano_transcription', 'utils'))
    import config as _pt_config  # type: ignore
    SAMPLE_RATE = int(getattr(_pt_config, 'sample_rate', 16000))
except Exception:
    SAMPLE_RATE = 16000


def _decode_if_bytes(x: Any) -> Any:
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode()
        except Exception:
            return x
    return x


def _fmt_attr_value(v: Any) -> str:
    v = _decode_if_bytes(v)
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__}(len={len(v)})"
    if hasattr(v, 'shape') and hasattr(v, 'dtype'):
        try:
            return f"ndarray(shape={v.shape}, dtype={v.dtype})"
        except Exception:
            return f"ndarray(?)"
    return str(v)


def _safe_filename(name: str) -> str:
    name = name.strip().replace(' ', '_')
    return re.sub(r'[^A-Za-z0-9_.\-]+', '', name)


def _to_float_audio(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.int16:
        return (x.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    if x.dtype == np.int32:
        # Assume 24-bit stored in 32-bit container
        return (x.astype(np.float32) / 2147483648.0).clip(-1.0, 1.0)
    x = x.astype(np.float32)
    # Normalize if outside [-1, 1]
    max_abs = float(np.max(np.abs(x))) if x.size > 0 else 1.0
    if max_abs > 1.0:
        x = x / (max_abs + 1e-8)
    return x


def print_group_structure(
    obj: Union[h5py.File, h5py.Group],
    name: str = '/',
    indent: int = 0,
    max_children: int = 20,
    depth: int = 0,
    max_depth: int = 4,
) -> None:
    prefix = '  ' * indent
    if isinstance(obj, h5py.Dataset):
        print(f"{prefix}- {name} [Dataset] shape={obj.shape} dtype={obj.dtype}")
        return

    # Group or File
    node_type = 'File' if isinstance(obj, h5py.File) else 'Group'
    print(f"{prefix}- {name} [{node_type}]")

    # Print attributes
    if len(obj.attrs) > 0:
        for k, v in obj.attrs.items():
            print(f"{prefix}    @attr {k}: {_fmt_attr_value(v)}")

    if depth >= max_depth:
        child_count = len(obj.keys()) if hasattr(obj, 'keys') else 0
        if child_count:
            print(f"{prefix}    ... (max_depth reached; {child_count} children not shown)")
        return

    # Print children
    try:
        keys = list(obj.keys())
    except Exception:
        keys = []
    shown = 0
    for key in keys:
        if shown >= max_children:
            print(f"{prefix}    ... (+{len(keys) - shown} more)")
            break
        child = obj[key]
        if isinstance(child, h5py.Dataset):
            print(f"{prefix}  - {key} [Dataset] shape={child.shape} dtype={child.dtype}")
        else:
            print_group_structure(child, name=key, indent=indent + 1, max_children=max_children, depth=depth + 1, max_depth=max_depth)
        shown += 1


def summarize_rwc(h5_path: str, max_files: int = 5, max_notes: int = 5) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        'num_files': 0,
        'num_notes_total': 0,
        'files': [],
    }

    with h5py.File(h5_path, 'r') as f:
        file_keys = [k for k in f.keys() if k.startswith('file_')]
        summary['num_files'] = len(file_keys)

        for i, file_key in enumerate(file_keys):
            if i >= max_files:
                break
            g = f[file_key]
            file_info: Dict[str, Any] = {'file_key': file_key, 'attrs': {}, 'datasets': {}, 'num_notes': 0, 'notes_preview': []}
            # Attributes
            for ak, av in g.attrs.items():
                file_info['attrs'][ak] = _decode_if_bytes(av)

            # Datasets directly under file group (if any)
            for dk, dv in g.items():
                if isinstance(dv, h5py.Dataset):
                    file_info['datasets'][dk] = {'shape': dv.shape, 'dtype': str(dv.dtype)}

            # Notes group
            if 'notes' in g and isinstance(g['notes'], h5py.Group):
                notes_g = g['notes']
                note_keys = [nk for nk in notes_g.keys() if nk.startswith('note_')]
                file_info['num_notes'] = len(note_keys)
                summary['num_notes_total'] += len(note_keys)
                for j, nk in enumerate(note_keys[:max_notes]):
                    ng = notes_g[nk]
                    note_entry: Dict[str, Any] = {'note_key': nk, 'datasets': {}, 'attrs': {}}
                    for n_ak, n_av in ng.attrs.items():
                        note_entry['attrs'][n_ak] = _decode_if_bytes(n_av)
                    for n_dk, n_dv in ng.items():
                        if isinstance(n_dv, h5py.Dataset):
                            note_entry['datasets'][n_dk] = {'shape': n_dv.shape, 'dtype': str(n_dv.dtype)}
                    file_info['notes_preview'].append(note_entry)

            summary['files'].append(file_info)

    return summary


def save_audio_previews(h5_path: str, out_dir: str, max_files: int = 5, max_notes: int = 5) -> None:
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    with h5py.File(h5_path, 'r') as f:
        file_keys = [k for k in f.keys() if k.startswith('file_')]
        for i, file_key in enumerate(file_keys[:max_files]):
            g = f[file_key]
            # Attributes for naming
            filename = _decode_if_bytes(g.attrs.get('filename', 'unknown'))
            technique = _decode_if_bytes(g.attrs.get('pt_name', '')) or _decode_if_bytes(g.attrs.get('technique', ''))
            label = _decode_if_bytes(g.attrs.get('pt_label', ''))

            base_name = _safe_filename(str(filename)) or file_key
            tech_tag = _safe_filename(str(technique)) if technique else 'na'
            label_tag = _safe_filename(str(label)) if label != '' else 'na'

            # Full audio dataset
            full_ds = None
            if 'full_waveform' in g and isinstance(g['full_waveform'], h5py.Dataset):
                full_ds = g['full_waveform']
            elif 'waveform' in g and isinstance(g['waveform'], h5py.Dataset):
                full_ds = g['waveform']
            if full_ds is not None:
                wav = _to_float_audio(full_ds[:])
                full_path = os.path.join(out_dir, f"{file_key}_{tech_tag}_{label_tag}_full.wav")
                sf.write(full_path, wav, SAMPLE_RATE)
                saved += 1

            # Note-level clips
            if 'notes' in g and isinstance(g['notes'], h5py.Group):
                notes_g = g['notes']
                note_keys = [nk for nk in notes_g.keys() if nk.startswith('note_')]
                for nk in note_keys[:max_notes]:
                    ng = notes_g[nk]
                    if 'waveform' in ng and isinstance(ng['waveform'], h5py.Dataset):
                        note_wav = _to_float_audio(ng['waveform'][:])
                        start_time = _decode_if_bytes(ng.attrs.get('start_time', 'na'))
                        try:
                            st_str = f"{float(start_time):.3f}"
                        except Exception:
                            st_str = str(start_time)
                        note_path = os.path.join(out_dir, f"{file_key}_{nk}_start_{_safe_filename(st_str)}.wav")
                        sf.write(note_path, note_wav, SAMPLE_RATE)
                        saved += 1

    print(f"Saved {saved} audio preview file(s) to: {out_dir}")


def summarize_techniques(h5_path: str):
    """Return per-file technique and aggregated counts.

    Returns:
      per_file: dict[file_key] -> technique_name (lower)
      counts: dict[technique_name] -> count
    """
    per_file: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    with h5py.File(h5_path, 'r') as f:
        file_keys = [k for k in f.keys() if k.startswith('file_')]
        for file_key in file_keys:
            g = f[file_key]
            tech = _decode_if_bytes(g.attrs.get('pt_name', '')) or _decode_if_bytes(g.attrs.get('technique', ''))
            tech = str(tech).strip().lower() if tech is not None else ''
            if tech == '':
                tech = 'no_technique'
            per_file[file_key] = tech
            counts[tech] = counts.get(tech, 0) + 1
    return per_file, counts


def main():
    parser = argparse.ArgumentParser(description='Inspect RWC HDF5 structure and contents')
    parser.add_argument('--h5_path', type=str, default='~/data/rwc_processed_data.h5', help='Path to RWC HDF5 file')
    parser.add_argument('--max_files', type=int, default=5, help='Max number of files to preview')
    parser.add_argument('--max_notes', type=int, default=5, help='Max number of notes per file to preview')
    parser.add_argument('--full', action='store_true', help='Print full recursive structure')
    parser.add_argument('--save_audio_dir', type=str, default=None, help='Directory to save a few audio previews (full and note-level)')
    parser.add_argument('--list_techniques', action='store_true', help='List technique per file and class counts')
    args = parser.parse_args()

    h5_path = os.path.expanduser(args.h5_path)
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    print(f"RWC H5: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        # High-level structure
        print("\nTop-level keys:")
        for k in list(f.keys())[:50]:
            node = f[k]
            node_type = 'Dataset' if isinstance(node, h5py.Dataset) else 'Group'
            if isinstance(node, h5py.Dataset):
                extra = f" shape={node.shape} dtype={node.dtype}"
            else:
                extra = f" (children={len(node.keys())})"
            print(f"  - {k} [{node_type}]{extra}")

        if args.full:
            print("\nFull structure (truncated by depth):")
            print_group_structure(f, name='/', indent=0, max_children=50, depth=0, max_depth=6)

    # Targeted summary for expected schema
    print("\nSummary preview (files and notes):")
    summary = summarize_rwc(h5_path, max_files=args.max_files, max_notes=args.max_notes)
    print(f"  Num files: {summary['num_files']}")
    print(f"  Total notes (approx): {summary['num_notes_total']}")

    for fi in summary['files']:
        print(f"\n  {fi['file_key']}")
        # File attributes of interest
        filename = fi['attrs'].get('filename')
        technique = fi['attrs'].get('pt_name') or fi['attrs'].get('technique')
        label = fi['attrs'].get('pt_label')
        duration = fi['attrs'].get('duration')
        print(f"    attrs: filename={filename} technique={technique} label={label} duration={duration}")
        for dk, dv in fi['datasets'].items():
            print(f"    dataset: {dk} -> shape={dv['shape']} dtype={dv['dtype']}")
        print(f"    notes: count={fi['num_notes']}")
        for ni in fi['notes_preview']:
            dur = ni['attrs'].get('duration')
            st = ni['attrs'].get('start_time')
            print(f"      {ni['note_key']}: start_time={st} duration={dur}")
            for n_dk, n_dv in ni['datasets'].items():
                print(f"        dataset: {n_dk} -> shape={n_dv['shape']} dtype={n_dv['dtype']}")

    if args.save_audio_dir:
        print(f"\nSaving audio previews to: {args.save_audio_dir}")
        save_audio_previews(h5_path, args.save_audio_dir, max_files=args.max_files, max_notes=args.max_notes)

    if args.list_techniques:
        print("\nTechnique summary:")
        per_file, counts = summarize_techniques(h5_path)
        total_files = len(per_file)
        for tech in sorted(counts.keys()):
            print(f"  {tech}: {counts[tech]} files ({counts[tech]/max(total_files,1):.1%})")
        print("\nPer-file technique (first N):")
        shown = 0
        for fk in sorted(per_file.keys()):
            print(f"  {fk}: {per_file[fk]}")
            shown += 1
            if shown >= args.max_files:
                if total_files > shown:
                    print(f"  ... (+{total_files - shown} more)")
                break


if __name__ == '__main__':
    main()


