import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import argparse
import csv
from typing import List, Optional, Tuple

import numpy as np
import torch

from models_contrast import NoteLevelTechniqueModel, Regress_onset_offset_frame_velocity_CRNN
from pytorch_utils import move_data_to_device
from utilities import load_audio, read_midi, read_musc_midi
import config


DEFAULT_TECHNIQUE_CLASSES = ['flageolet', 'normal', 'pizzicato', 'spiccato', 'no_technique']


def build_transcriptor(device: torch.device, frames_per_second: int, classes_num: int, checkpoint_path: Optional[str]) -> Optional[torch.nn.Module]:
    if checkpoint_path is None or (not os.path.exists(checkpoint_path)):
        return None
    model = Regress_onset_offset_frame_velocity_CRNN(
        frames_per_second=frames_per_second,
        classes_num=classes_num,
        output_features=False,
        predict_technique=False,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt)
    try:
        model.load_state_dict(state, strict=False)
    except TypeError:
        model.load_state_dict(state)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def load_note_model(checkpoint_path: str, device: torch.device, frames_per_second: int, use_trans_features: bool, trans_feat_dim: int) -> NoteLevelTechniqueModel:
    model = NoteLevelTechniqueModel(
        classes_num=5,
        sample_rate=config.sample_rate,
        frames_per_second=frames_per_second,
        trans_feat_dim=trans_feat_dim,
        use_trans_features=use_trans_features,
        output_features=False,
    ).to(device)
    # PyTorch >= 2.6 defaults to weights_only=True which can fail on older checkpoints
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get('note_model', ckpt.get('model', ckpt))
    try:
        model.load_state_dict(state, strict=False)
    except TypeError:
        model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def parse_midi_to_note_events(midi_path: str) -> List[dict]:
    """Parse MIDI to a list of note events with onset/offset in seconds.

    Each note event dict: {'midi_note': int, 'onset_time': float, 'offset_time': float, 'velocity': int}
    """
    try:
        midi_dict = read_midi(midi_path)
    except Exception:
        midi_dict = read_musc_midi(midi_path)

    midi_events = midi_dict['midi_event']
    midi_times = midi_dict['midi_event_time']

    buffer = {}
    note_events: List[dict] = []

    for i in range(len(midi_events)):
        parts = midi_events[i].split(' ')
        t = float(midi_times[i])
        if parts[0] in ['note_on', 'note_off']:
            # E.g. ['note_on', 'channel=0', 'note=64', 'velocity=80', 'time=...']
            try:
                midi_note = int(parts[2].split('=')[1])
                velocity = int(parts[3].split('=')[1])
            except Exception:
                continue
            if parts[0] == 'note_on' and velocity > 0:
                buffer[midi_note] = {'onset_time': t, 'velocity': velocity}
            else:
                if midi_note in buffer:
                    note_events.append({
                        'midi_note': midi_note,
                        'onset_time': float(buffer[midi_note]['onset_time']),
                        'offset_time': float(t),
                        'velocity': int(buffer[midi_note]['velocity']),
                    })
                    del buffer[midi_note]

    # Close any unpaired onsets with a tiny tail
    for midi_note, meta in buffer.items():
        note_events.append({
            'midi_note': midi_note,
            'onset_time': float(meta['onset_time']),
            'offset_time': float(meta['onset_time'] + 0.05),
            'velocity': int(meta['velocity']),
        })

    note_events.sort(key=lambda x: (x['onset_time'], x['offset_time'], x['midi_note']))
    return note_events


def slice_waveform_by_notes(waveform: np.ndarray, sr: int, notes: List[dict], pad_seconds: float = 0.02) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """Slice waveform per note with small padding.

    Returns a list of 1D numpy arrays and their (start_sample, end_sample) tuples.
    """
    n = waveform.shape[0]
    pad = int(round(pad_seconds * sr))
    chunks: List[np.ndarray] = []
    spans: List[Tuple[int, int]] = []
    for ev in notes:
        s = max(0, int(round(ev['onset_time'] * sr)) - pad)
        e = min(n, int(round(ev['offset_time'] * sr)) + pad)
        if e <= s:
            e = min(n, s + int(0.02 * sr))
        chunks.append(waveform[s:e])
        spans.append((s, e))
    return chunks, spans


@torch.no_grad()
def infer_batch(
    model: NoteLevelTechniqueModel,
    device: torch.device,
    wave_chunks: List[np.ndarray],
    transcriptor: Optional[torch.nn.Module],
    use_trans_features: bool,
    trans_features_list: Optional[List[str]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Run batched inference on note waveform chunks.

    Returns (pred_indices, pred_confidences)
    """
    if len(wave_chunks) == 0:
        return np.array([]), np.array([])

    # Pad waves to tensors with ragged handling by per-item forward
    # For efficiency, group by approximate length to form small batches
    lengths = np.array([w.shape[0] for w in wave_chunks])
    order = np.argsort(lengths)
    pred_idx_list: List[int] = []
    pred_conf_list: List[float] = []

    BATCH = 32
    i = 0
    while i < len(order):
        j = min(len(order), i + BATCH)
        idxs = order[i:j].tolist()
        batch = [wave_chunks[k] for k in idxs]

        max_len = max(len(b) for b in batch)
        # Ensure minimum length for STFT reflect padding (n_fft//2 = 1024 in models)
        target_len = max(max_len, 2048)
        x = np.stack([np.pad(b, (0, max(0, target_len - len(b))), mode='constant') for b in batch], axis=0)
        x_t = torch.tensor(x, dtype=torch.float32, device=device)

        trans_feats_t = None
        if use_trans_features and transcriptor is not None:
            out = transcriptor(x_t)
            onset = out.get('reg_onset_output')
            offset = out.get('reg_offset_output')
            frame = out.get('frame_output')
            velocity = out.get('velocity_output')
            if trans_features_list is None:
                tf = torch.cat([onset, offset, frame, velocity], dim=-1)
            else:
                tf = torch.cat([out.get(feat) for feat in trans_features_list], dim=-1)
            trans_feats_t = tf

        logits = model(x_t, trans_features=trans_feats_t)  # (b, C)
        probs = torch.softmax(logits, dim=-1)
        conf, pred = torch.max(probs, dim=-1)
        # Restore original ordering for this block
        for pi, ci in zip(pred.tolist(), conf.tolist()):
            pred_idx_list.append(int(pi))
            pred_conf_list.append(float(ci))

        i = j

    # Reorder back to original note order
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(len(order))
    pred_idx_arr = np.array(pred_idx_list)[inv_order]
    pred_conf_arr = np.array(pred_conf_list)[inv_order]
    return pred_idx_arr, pred_conf_arr


def write_csv(
    csv_path: str,
    notes: List[dict],
    pred_idx: np.ndarray,
    pred_conf: np.ndarray,
    class_names: List[str],
) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['note_index', 'midi_note', 'onset_time', 'offset_time', 'duration', 'velocity', 'technique', 'confidence'])
        for i, ev in enumerate(notes):
            onset = float(ev['onset_time'])
            offset = float(ev['offset_time'])
            duration = max(0.0, offset - onset)
            label = class_names[int(pred_idx[i])] if len(class_names) > int(pred_idx[i]) else str(int(pred_idx[i]))
            writer.writerow([
                i,
                int(ev['midi_note']),
                f"{onset:.6f}",
                f"{offset:.6f}",
                f"{duration:.6f}",
                int(ev.get('velocity', 0)),
                label,
                f"{float(pred_conf[i]):.6f}",
            ])


def main():
    parser = argparse.ArgumentParser(description='Infer note-level technique from audio+MIDI and export CSV')
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--midi_path', type=str, required=True)
    parser.add_argument('--note_model_checkpoint', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_trans_features', action='store_true', default=False)
    parser.add_argument('--transcriptor_checkpoint', type=str, default=None)
    parser.add_argument('--trans_features_list', nargs='+', default=None)
    parser.add_argument('--pad_seconds', type=float, default=0.02)

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}") if (args.device is not None and args.device >= 0 and torch.cuda.is_available()) else torch.device('cpu')
    frames_per_second = config.frames_per_second
    sample_rate = config.sample_rate

    # Compute trans feature dim
    trans_feat_dim = 88 * (4 if args.trans_features_list is None else len(args.trans_features_list))

    # Load audio
    waveform, sr = load_audio(args.audio_path, sr=sample_rate, mono=True)
    if sr != sample_rate:
        raise RuntimeError(f"Unexpected sample rate {sr}, expected {sample_rate}")

    # Parse MIDI to note events
    note_events = parse_midi_to_note_events(args.midi_path)
    if len(note_events) == 0:
        raise RuntimeError(f"No note events parsed from MIDI: {args.midi_path}")

    # Slice waveform per note
    chunks, _ = slice_waveform_by_notes(waveform, sample_rate, note_events, pad_seconds=args.pad_seconds)

    # Models
    note_model = load_note_model(args.note_model_checkpoint, device, frames_per_second, args.use_trans_features, trans_feat_dim)
    transcriptor = None
    if args.use_trans_features:
        transcriptor = build_transcriptor(device, frames_per_second, config.classes_num, args.transcriptor_checkpoint)

    # Inference
    pred_idx, pred_conf = infer_batch(
        model=note_model,
        device=device,
        wave_chunks=chunks,
        transcriptor=transcriptor,
        use_trans_features=args.use_trans_features,
        trans_features_list=args.trans_features_list,
    )

    # Map to labels
    class_names = DEFAULT_TECHNIQUE_CLASSES

    # Write CSV
    write_csv(args.output_csv, note_events, pred_idx, pred_conf, class_names)


if __name__ == '__main__':
    main()


