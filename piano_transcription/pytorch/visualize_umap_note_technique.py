import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import argparse
import numpy as np
import torch
import torch.utils.data
import matplotlib.pyplot as plt

from typing import Optional

import matplotlib as mpl

# Enlarge all fonts by 1.5x
_FONT_SCALE = 1.5
_base_font = mpl.rcParams.get('font.size', 10.0)
if isinstance(_base_font, (int, float)):
    mpl.rcParams['font.size'] = _base_font * _FONT_SCALE

from models_contrast import NoteLevelTechniqueModel, Regress_onset_offset_frame_velocity_CRNN
from data_generator import RWCTechNoteWavDataset
from pytorch_utils import move_data_to_device
import config


def load_note_model(checkpoint_path: Optional[str], device: torch.device, frames_per_second: int, trans_feat_dim: int = 88 * 4, use_trans_features: bool = False) -> NoteLevelTechniqueModel:
    model = NoteLevelTechniqueModel(
        classes_num=5,
        sample_rate=config.sample_rate,
        frames_per_second=frames_per_second,
        trans_feat_dim=trans_feat_dim,
        use_trans_features=use_trans_features,
        output_features=True,
    ).to(device)
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get('note_model', ckpt.get('model', ckpt))
        try:
            model.load_state_dict(state, strict=False)
        except TypeError:
            model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

def build_transcriptor(device, frames_per_second, classes_num, checkpoint_path):
    model = Regress_onset_offset_frame_velocity_CRNN(
        frames_per_second=frames_per_second,
        classes_num=classes_num,
        output_features=False,
        predict_technique=False,
    ).to(device)
    if checkpoint_path is None or (not os.path.exists(checkpoint_path)):
        raise FileNotFoundError(f"Transcription checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=False)
    try:
        model.load_state_dict(ckpt['model'], strict=False)
    except TypeError:
        model.load_state_dict(ckpt['model'])
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model

def extract_features(args):
    device = torch.device(f"cuda:{args.device}") if (args.device is not None and args.device >= 0 and torch.cuda.is_available()) else torch.device('cpu')
    frames_per_second = config.frames_per_second

    dataset = RWCTechNoteWavDataset(
        os.path.expanduser(args.rwc_notes_root),
        frames_per_second=frames_per_second,
        split='test' if args.split == 'test' else 'train',
        augmentor=None,
        fold_id=args.fold_id,
        return_logmel=False,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    transcriptor = build_transcriptor(device, frames_per_second, config.classes_num, args.transcriptor_checkpoint) 

    model = load_note_model(args.checkpoint, device, frames_per_second, use_trans_features=args.use_trans_features)

    features = []  # (N, 128)
    labels = []    # (N,)

    with torch.no_grad():
        for batch in loader:
            waveform = move_data_to_device(batch['waveform'], device).float()  # (B, L)
            trans_feats = None
            if args.use_trans_features and transcriptor is not None:
                with torch.no_grad():
                    out = transcriptor(waveform.float())
                    onset = out.get('reg_onset_output')
                    offset = out.get('reg_offset_output')
                    frame = out.get('frame_output')
                    velocity = out.get('velocity_output')
                    if args.trans_features_list is None:
                        tf = torch.cat([onset, offset, frame, velocity], dim=-1)  # (B, Tc, 88*4)
                    else:
                        tf = torch.cat([out.get(feat) for feat in args.trans_features_list], dim=-1)  # (B, Tc, 88*n_features)
                    # time align will be handled after encoding; estimate T via spec hop
                    trans_feats = tf

            technique_oh = batch['technique']  # (B, 1, C)
            logits, emb = model(waveform, trans_feats)  # emb: (B, 128)
            features.append(emb.cpu().numpy())
            labels.append(technique_oh[:, 0, :].argmax(dim=-1).cpu().numpy())

    X = np.concatenate(features, axis=0)
    y = np.concatenate(labels, axis=0)
    return X, y, getattr(dataset, 'technique_classes', ['flageolet', 'normal', 'pizzicato', 'spiccato', 'no_technique'])


def run_umap(X: np.ndarray, y: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1, metric: str = 'euclidean'):
    try:
        import umap
    except Exception:
        import umap.umap_ as umap
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
    Z = reducer.fit_transform(X)
    return Z


def plot_scatter(Z: np.ndarray, y: np.ndarray, class_names, out_png: str):
    plt.figure(figsize=(8, 6), dpi=150)
    num_classes = len(class_names)
    cmap = plt.get_cmap('tab10')
    for c in range(num_classes):
        mask = (y == c)
        if not np.any(mask):
            continue
        plt.scatter(Z[mask, 0], Z[mask, 1], s=10, color=cmap(c), alpha=0.7, label=class_names[c] if class_names[c] != "normal" else "detache")

    plt.legend(loc='best', fontsize=10 * _FONT_SCALE)
    plt.title('UMAP of Note-Level Technique Embeddings')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize UMAP for note-level technique embeddings')
    parser.add_argument('--rwc_notes_root', type=str, default='/home/yuehpo/coding/violin-mamba/rwc_notes')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained note model checkpoint (.pth)')
    parser.add_argument('--split', type=str, choices=['train', 'test'], default='test')
    parser.add_argument('--fold_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--out_png', type=str, default='umap_note_technique.png')
    parser.add_argument('--out_npz', type=str, default='umap_note_technique.npz')
    parser.add_argument('--n_neighbors', type=int, default=30)
    parser.add_argument('--min_dist', type=float, default=0.05)
    parser.add_argument('--metric', type=str, default='euclidean')
    parser.add_argument('--transcriptor_checkpoint', type=str, default=None)
    parser.add_argument('--use_trans_features', action='store_true', default=False)
    parser.add_argument('--trans_features_list', nargs='+', default=['reg_onset_output', 'reg_offset_output', 'frame_output', 'velocity_output'])

    args = parser.parse_args()

    X, y, class_names = extract_features(args)
    Z = run_umap(X, y, n_neighbors=args.n_neighbors, min_dist=args.min_dist, metric=args.metric)

    # Save
    np.savez(args.out_npz, X=X, y=y, Z=Z, class_names=np.array(class_names))
    plot_scatter(Z, y, class_names, args.out_png)
    print(f"Saved UMAP plot to {args.out_png} and data to {args.out_npz}")


if __name__ == '__main__':
    main()




