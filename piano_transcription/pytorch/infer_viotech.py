"""
VioTech Inference: Violin Transcription + Frame-Level Technique Prediction

Input:
  - audio_dir: directory containing audio files (.wav/.flac/.mp3)
  - mapping_config: JSON file mapping technique names to indices
  - checkpoint_path: model checkpoint (Regress_onset_offset_frame_velocity_CRNN with technique head)

Output (per audio file):
  - MIDI with technique-separated tracks
  - CSV with per-note technique labels
  - Visualization PNG: piano roll + technique prediction rolls (one row per technique class)
"""

import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'utils'))

import argparse
import csv
import glob
import json
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

from inference_pt import PianoTranscription
from utilities import (load_audio, write_events_to_midi_with_technique,
                       write_events_to_midi, get_filename)
import config


# ---------------------------------------------------------------------------
# Mapping config helpers
# ---------------------------------------------------------------------------

def load_mapping_config(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def build_idx_to_name(mapping_config: dict) -> Dict[str, Dict[int, str]]:
    """Invert mapping_config so we get {group: {idx: name}} with non-empty names preferred."""
    result: Dict[str, Dict[int, str]] = {}
    for group, name_to_idx in mapping_config.items():
        idx_to_name: Dict[int, str] = {}
        for name, idx in name_to_idx.items():
            if name in ('', None):
                continue
            if idx not in idx_to_name or name != 'none':
                idx_to_name[idx] = name
        result[group] = dict(sorted(idx_to_name.items()))
    return result


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Technique assignment
# ---------------------------------------------------------------------------

def assign_techniques_to_notes(
    note_events: List[dict],
    output_dict: dict,
    fps: int,
    idx_to_name: Dict[str, Dict[int, str]],
) -> List[dict]:
    """Assign per-note technique labels using frame-level model outputs."""

    tonal_raw = output_dict.get('tonal_technique_output')
    artic_raw = output_dict.get('articulation_output')
    legato_raw = output_dict.get('legato_output')

    tonal_names = idx_to_name.get('tonalTechnique', {})
    artic_names = idx_to_name.get('articulation', {})

    for ev in note_events:
        on_f = int(ev['onset_time'] * fps)
        off_f = max(on_f + 1, int(ev['offset_time'] * fps))

        if tonal_raw is not None:
            seg = _softmax(tonal_raw[on_f:off_f], axis=-1)
            avg = seg.mean(axis=0) if len(seg) > 0 else np.zeros(tonal_raw.shape[-1])
            idx = int(np.argmax(avg))
            ev['tonal_technique'] = idx
            ev['tonal_technique_name'] = tonal_names.get(idx, f'tonal_{idx}')
            ev['tonal_confidence'] = float(avg[idx])

        if artic_raw is not None:
            seg = _softmax(artic_raw[on_f:off_f], axis=-1)
            avg = seg.mean(axis=0) if len(seg) > 0 else np.zeros(artic_raw.shape[-1])
            idx = int(np.argmax(avg))
            ev['articulation'] = idx
            ev['articulation_name'] = artic_names.get(idx, f'artic_{idx}')
            ev['artic_confidence'] = float(avg[idx])

        if legato_raw is not None:
            seg = legato_raw[on_f:off_f]
            ev['legato'] = float(seg.mean()) if len(seg) > 0 else 0.0

        # technique key used by write_events_to_midi_with_technique
        ev['technique'] = ev.get('tonal_technique', -1)

    return note_events


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_technique_csv(
    csv_path: str,
    note_events: List[dict],
) -> None:
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'note_index', 'midi_note', 'onset_time', 'offset_time',
            'duration', 'velocity',
            'tonal_technique', 'tonal_confidence',
            'articulation', 'artic_confidence',
            'legato',
        ])
        for i, ev in enumerate(note_events):
            on = ev['onset_time']
            off = ev['offset_time']
            w.writerow([
                i, ev['midi_note'],
                f'{on:.6f}', f'{off:.6f}', f'{off - on:.6f}',
                ev.get('velocity', 0),
                ev.get('tonal_technique_name', 'unknown'),
                f"{ev.get('tonal_confidence', 0):.4f}",
                ev.get('articulation_name', 'unknown'),
                f"{ev.get('artic_confidence', 0):.4f}",
                f"{ev.get('legato', 0):.4f}",
            ])
    print(f'  CSV saved: {csv_path}')


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

_TONAL_COLORS = {
    0: '#78909C',   # none  - grey
    1: '#E91E63',   # pizzicato - pink
    2: '#2196F3',   # harmonics - blue
    3: '#FF9800',   # openstring - orange
    -1: '#9E9E9E',  # unknown
}

_ARTIC_COLORS = {
    0: '#78909C',
    1: '#8BC34A',   # release - green
    2: '#FF5722',   # staccato - red-orange
    3: '#9C27B0',   # spiccato - purple
    -1: '#9E9E9E',
}


def _plot_piano_roll(
    ax: plt.Axes,
    note_events: List[dict],
    frame_output: np.ndarray,
    fps: int,
    duration_s: float,
):
    """Piano roll: frame activation background + coloured note rectangles."""
    total_frames = frame_output.shape[0]
    t_end = total_frames / fps

    ax.imshow(
        frame_output.T, aspect='auto', origin='lower', cmap='Greys',
        extent=[0, t_end, config.begin_note, config.begin_note + config.classes_num],
        alpha=0.25, vmin=0, vmax=1, interpolation='nearest',
    )

    for ev in note_events:
        on, off = ev['onset_time'], ev['offset_time']
        pitch = ev['midi_note']
        tech = ev.get('tonal_technique', -1)
        colour = _TONAL_COLORS.get(tech, '#9E9E9E')
        rect = Rectangle(
            (on, pitch - 0.4), max(off - on, 0.01), 0.8,
            linewidth=0.4, edgecolor='#333333', facecolor=colour, alpha=0.9,
        )
        ax.add_patch(rect)

    if note_events:
        pitches = [e['midi_note'] for e in note_events]
        lo = max(config.begin_note, min(pitches) - 3)
        hi = min(config.begin_note + config.classes_num, max(pitches) + 3)
        ax.set_ylim(lo, hi)

    ax.set_xlim(0, duration_s)
    ax.set_ylabel('MIDI Note')


def _plot_technique_roll(
    ax: plt.Axes,
    probs: np.ndarray,
    labels: List[str],
    fps: int,
    title: str,
):
    """Heatmap where each row is one technique class, x-axis is time."""
    if probs.ndim == 1:
        probs = probs[:, None]
    n_frames, n_cls = probs.shape
    t_end = n_frames / fps

    cmap = LinearSegmentedColormap.from_list('tech', ['#FAFAFA', '#1A237E'])
    im = ax.imshow(
        probs.T, aspect='auto', origin='lower', cmap=cmap,
        extent=[0, t_end, -0.5, n_cls - 0.5],
        vmin=0, vmax=1, interpolation='nearest',
    )

    ax.set_yticks(range(n_cls))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold', loc='left')
    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.008, label='Prob')


def visualize(
    output_dict: dict,
    note_events: List[dict],
    idx_to_name: Dict[str, Dict[int, str]],
    duration_s: float,
    save_path: str,
    title: str = '',
):
    fps = config.frames_per_second

    has_tonal = 'tonal_technique_output' in output_dict
    has_artic = 'articulation_output' in output_dict
    has_legato = 'legato_output' in output_dict

    n_rows = 1 + int(has_tonal) + int(has_artic) + int(has_legato)
    ratios = [4]
    if has_tonal:
        ratios.append(2)
    if has_artic:
        ratios.append(2)
    if has_legato:
        ratios.append(1)

    fig_w = min(60, max(14, duration_s * 0.8))
    fig_h = sum(ratios) * 0.7 + 1.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(n_rows, 1, height_ratios=ratios, hspace=0.35)
    axes = [fig.add_subplot(gs[i]) for i in range(n_rows)]

    # Piano roll
    _plot_piano_roll(axes[0], note_events, output_dict['frame_output'], fps, duration_s)
    axes[0].set_title(f'Piano Roll  —  {title}', fontsize=11, fontweight='bold', loc='left')

    # Build technique legend for piano roll
    tonal_names = idx_to_name.get('tonalTechnique', {})
    legend_handles = []
    for idx, name in tonal_names.items():
        c = _TONAL_COLORS.get(idx, '#9E9E9E')
        legend_handles.append(Rectangle((0, 0), 1, 1, fc=c, ec='#333', label=name))
    if legend_handles:
        axes[0].legend(handles=legend_handles, loc='upper right', fontsize=8,
                       ncol=len(legend_handles), framealpha=0.8)

    row = 1

    if has_tonal:
        tonal_probs = _softmax(output_dict['tonal_technique_output'], axis=-1)
        labels = [tonal_names.get(i, f'tonal_{i}') for i in range(tonal_probs.shape[-1])]
        _plot_technique_roll(axes[row], tonal_probs, labels, fps, 'Tonal Technique')
        row += 1

    if has_artic:
        artic_probs = _softmax(output_dict['articulation_output'], axis=-1)
        artic_map = idx_to_name.get('articulation', {})
        labels = [artic_map.get(i, f'artic_{i}') for i in range(artic_probs.shape[-1])]
        _plot_technique_roll(axes[row], artic_probs, labels, fps, 'Articulation')
        row += 1

    if has_legato:
        legato = output_dict['legato_output']
        if legato.ndim == 2 and legato.shape[-1] == 1:
            legato = legato[:, 0]
        _plot_technique_roll(axes[row], legato[:, None] if legato.ndim == 1 else legato,
                             ['legato'], fps, 'Legato')
        row += 1

    axes[-1].set_xlabel('Time (s)', fontsize=10)

    # Share x-axis limits
    for a in axes:
        a.set_xlim(0, duration_s)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Visualization saved: {save_path}')


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(
    audio_path: str,
    transcriptor: PianoTranscription,
    output_dir: str,
    mapping_config: dict,
    idx_to_name: Dict[str, Dict[int, str]],
) -> Optional[List[dict]]:
    fps = config.frames_per_second
    base = get_filename(audio_path)
    midi_path = os.path.join(output_dir, f'{base}.mid')
    csv_path = os.path.join(output_dir, f'{base}_techniques.csv')
    png_path = os.path.join(output_dir, f'{base}_visualization.png')

    # Load & normalise audio
    try:
        waveform, _ = load_audio(audio_path, sr=config.sample_rate, mono=True)
    except Exception as e:
        print(f'  [Error] load audio: {e}')
        return None
    if waveform.size == 0:
        print(f'  [Skip] empty audio')
        return None
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    duration_s = len(waveform) / config.sample_rate

    # Transcribe (technique outputs are included in output_dict automatically)
    result = transcriptor.transcribe(waveform)
    output_dict = result['output_dict']
    note_events = result['est_note_events']
    pedal_events = result['est_pedal_events']

    if not note_events:
        print(f'  [Skip] no notes detected')
        return None

    # Assign technique labels
    note_events = assign_techniques_to_notes(note_events, output_dict, fps, idx_to_name)

    # MIDI
    write_events_to_midi_with_technique(
        start_time=0, note_events=note_events,
        pedal_events=pedal_events, midi_path=midi_path,
    )
    print(f'  MIDI saved: {midi_path}')

    # CSV
    write_technique_csv(csv_path, note_events)

    # Visualization
    visualize(output_dict, note_events, idx_to_name, duration_s, png_path, title=base)

    return note_events


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='VioTech Inference — violin transcription + technique prediction',
    )
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='Directory of audio files (.wav/.flac/.mp3)')
    parser.add_argument('--mapping_config', type=str, required=True,
                        help='Path to mapping_config.json')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for MIDI / CSV / visualizations')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU index (-1 for CPU)')
    parser.add_argument('--model_type', type=str,
                        default='Regress_onset_offset_frame_velocity_CRNN')

    args = parser.parse_args()

    # Mapping config
    mapping_config = load_mapping_config(args.mapping_config)
    idx_to_name = build_idx_to_name(mapping_config)

    print('Technique classes:')
    for group, names in idx_to_name.items():
        print(f'  {group}: {names}')

    # Device
    if args.device >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model (predict_technique=True via enable_technique)
    print('\nLoading model...')
    transcriptor = PianoTranscription(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint_path,
        device=device,
        segment_samples=int(config.sample_rate * config.segment_seconds),
        post_processor_type='regression',
        enable_technique=True,
    )

    # Collect audio files
    audio_files: List[str] = []
    for pattern in ('*.wav', '*.WAV', '*.flac', '*.FLAC', '*.mp3', '*.MP3'):
        audio_files.extend(glob.glob(os.path.join(args.audio_dir, pattern)))
    audio_files = sorted(set(audio_files))

    if not audio_files:
        print(f'No audio files found in {args.audio_dir}')
        return

    print(f'Found {len(audio_files)} audio file(s)\n')

    t0 = time.time()
    for i, path in enumerate(audio_files):
        print(f'[{i + 1}/{len(audio_files)}] {os.path.basename(path)}')
        process_file(path, transcriptor, args.output_dir, mapping_config, idx_to_name)

    elapsed = time.time() - t0
    print(f'\nDone — {len(audio_files)} files in {elapsed:.1f}s')
    print(f'Results in: {args.output_dir}')


if __name__ == '__main__':
    main()
