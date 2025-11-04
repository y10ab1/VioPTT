import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import argparse
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data
import matplotlib.pyplot as plt
import matplotlib as mpl

from pytorch_utils import move_data_to_device
from data_generator import RWCTechNoteWavDataset
from models_contrast import NoteLevelTechniqueModel, Regress_onset_offset_frame_velocity_CRNN
import config


# Enlarge all fonts by 1.5x
_FONT_SCALE = 1.5
_base_font = mpl.rcParams.get('font.size', 10.0)
if isinstance(_base_font, (int, float)):
    mpl.rcParams['font.size'] = _base_font * _FONT_SCALE


def load_note_model(checkpoint_path: Optional[str], device: torch.device, frames_per_second: int,
                    trans_feat_dim: int = 88 * 4, use_trans_features: bool = False) -> NoteLevelTechniqueModel:
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


def run_umap(X: np.ndarray, n_neighbors: int = 30, min_dist: float = 0.05, metric: str = 'euclidean'):
    try:
        import umap
    except Exception:
        import umap.umap_ as umap
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
    Z = reducer.fit_transform(X)
    return Z


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: Optional[List[str]] = None, normalize: bool = True):
    num_classes = len(class_names) if class_names is not None else (int(max(y_true.max(), y_pred.max())) + 1)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm.astype(np.float64) / row_sums
        return cm, cm_norm
    else:
        return cm, None


def per_class_accuracy(cm_counts: np.ndarray) -> np.ndarray:
    row_sums = cm_counts.sum(axis=1)
    correct = np.diag(cm_counts)
    acc = np.divide(correct, row_sums, out=np.zeros_like(correct, dtype=float), where=row_sums > 0)
    return acc


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_png: str, normalize: bool = True):
    labels_to_show = ["détaché" if name == "normal" else name for name in class_names]
    plt.figure(figsize=(8, 6), dpi=150)
    M = cm
    cmap = plt.cm.Blues
    im = plt.imshow(M, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(labels_to_show))
    plt.xticks(tick_marks, labels_to_show, rotation=45, ha='right')
    plt.yticks(tick_marks, labels_to_show)

    fmt = '.2f' if normalize else 'd'
    thresh = M.max() / 2.0 if M.size > 0 else 0.5
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            value = M[i, j]
            plt.text(j, i, format(value, fmt), ha="center", va="center",
                     color="white" if value > thresh else "black", fontsize=9 * _FONT_SCALE)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.title('Technique Confusion Matrix' + (' (normalized)' if normalize else ''))
    plt.title('Technique Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_scatter(Z: np.ndarray, y: np.ndarray, class_names: List[str], out_png: str):
    plt.figure(figsize=(8, 6), dpi=150)
    num_classes = len(class_names)
    cmap = plt.get_cmap('tab10')
    for c in range(num_classes):
        mask = (y == c)
        if not np.any(mask):
            continue
        plt.scatter(Z[mask, 0], Z[mask, 1], s=10, color=cmap(c), alpha=0.7,
                    label=class_names[c] if class_names[c] != "normal" else "détaché")

    plt.legend(loc='best', fontsize=10 * _FONT_SCALE)
    plt.title('UMAP of Note-Level Technique Embeddings')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def extract_with_models(model: NoteLevelTechniqueModel, transcriptor: Optional[torch.nn.Module], dataset: RWCTechNoteWavDataset,
                        device: torch.device, use_trans_features: bool, trans_features_list: Optional[List[str]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    features = []
    labels = []
    preds = []

    with torch.no_grad():
        for batch in loader:
            waveform = move_data_to_device(batch['waveform'], device).float()
            trans_feats = None
            if use_trans_features and transcriptor is not None:
                out = transcriptor(waveform.float())
                onset = out.get('reg_onset_output')
                offset = out.get('reg_offset_output')
                frame = out.get('frame_output')
                velocity = out.get('velocity_output')
                if trans_features_list is None:
                    tf = torch.cat([onset, offset, frame, velocity], dim=-1)
                else:
                    tf = torch.cat([out.get(feat) for feat in trans_features_list], dim=-1)
                trans_feats = tf

            technique_oh = batch['technique']
            logits, emb = model(waveform, trans_feats)
            features.append(emb.cpu().numpy())
            labels.append(technique_oh[:, 0, :].argmax(dim=-1).cpu().numpy())
            if logits.ndim == 3:
                pred = logits[:, 0, :].argmax(dim=-1)
            elif logits.ndim == 2:
                pred = logits.argmax(dim=-1)
            else:
                raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
            preds.append(pred.cpu().numpy())

    X = np.concatenate(features, axis=0)
    y = np.concatenate(labels, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    class_names = getattr(dataset, 'technique_classes', ['flageolet', 'normal', 'pizzicato', 'spiccato', 'no_technique'])
    return X, y, y_pred, class_names


def main():
    parser = argparse.ArgumentParser(description='UMAP + Confusion for 3-fold note technique models')
    parser.add_argument('--rwc_notes_root', type=str, default='/home/yuehpo/coding/violin-mamba/rwc_notes')
    parser.add_argument('--checkpoints', type=str, nargs=3, required=True, help='Three note-model checkpoints for folds 0/1/2')
    parser.add_argument('--fold_ids', type=int, nargs=3, default=[0, 1, 2])
    parser.add_argument('--split', type=str, choices=['train', 'test'], default='test')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default='.')
    parser.add_argument('--n_neighbors', type=int, default=30)
    parser.add_argument('--min_dist', type=float, default=0.05)
    parser.add_argument('--metric', type=str, default='euclidean')
    parser.add_argument('--transcriptor_checkpoint', type=str, default=None)
    parser.add_argument('--use_trans_features', action='store_true', default=False)
    parser.add_argument('--trans_features_list', nargs='+', default=['reg_onset_output', 'reg_offset_output', 'frame_output', 'velocity_output'])
    parser.add_argument('--plot_confusion', action='store_true', default=True)
    parser.add_argument('--ignore_confusion_class', type=str, default='no_technique')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.device}") if (args.device is not None and args.device >= 0 and torch.cuda.is_available()) else torch.device('cpu')
    fps = config.frames_per_second

    transcriptor = None
    if args.use_trans_features:
        transcriptor = build_transcriptor(device, fps, config.classes_num, args.transcriptor_checkpoint)

    npz_paths = []
    # Accumulate for all-fold confusion
    all_y_conf = []
    all_y_pred_conf = []
    all_conf_class_names = None
    for ckpt_path, fold_id in zip(args.checkpoints, args.fold_ids):
        model = load_note_model(ckpt_path, device, fps, use_trans_features=args.use_trans_features)
        dataset = RWCTechNoteWavDataset(
            os.path.expanduser(args.rwc_notes_root),
            frames_per_second=fps,
            split='test' if args.split == 'test' else 'train',
            augmentor=None,
            fold_id=fold_id,
            return_logmel=False,
        )

        X, y, y_pred, class_names = extract_with_models(model, transcriptor, dataset, device,
                                                        use_trans_features=args.use_trans_features,
                                                        trans_features_list=args.trans_features_list)
        Z = run_umap(X, n_neighbors=args.n_neighbors, min_dist=args.min_dist, metric=args.metric)

        # Confusion: optionally ignore a class
        conf_class_names = list(class_names)
        y_conf = y.copy()
        y_pred_conf = y_pred.copy()
        ignore_name = (args.ignore_confusion_class or '').strip()
        if ignore_name and ignore_name in conf_class_names:
            ignore_idx = conf_class_names.index(ignore_name)
            keep_mask = (y_conf != ignore_idx) & (y_pred_conf != ignore_idx)
            y_conf = y_conf[keep_mask]
            y_pred_conf = y_pred_conf[keep_mask]
            remap = np.arange(len(conf_class_names))
            remap[ignore_idx:] -= 1
            y_conf = remap[y_conf]
            y_pred_conf = remap[y_pred_conf]
            conf_class_names = [n for i, n in enumerate(conf_class_names) if i != ignore_idx]

        cm_counts, cm_norm = compute_confusion_matrix(y_conf, y_pred_conf, class_names=conf_class_names, normalize=True)
        acc = per_class_accuracy(cm_counts)

        prefix = f"fold{fold_id}"
        out_npz = os.path.join(args.out_dir, f"umap_note_technique_{prefix}.npz")
        out_png = os.path.join(args.out_dir, f"umap_note_technique_{prefix}.png")
        conf_png = os.path.join(args.out_dir, f"confusion_note_technique_{prefix}.png")

        np.savez(out_npz, X=X, y=y, y_pred=y_pred, Z=Z, class_names=np.array(class_names),
                 cm=cm_counts, cm_norm=cm_norm, confusion_class_names=np.array(conf_class_names),
                 per_class_acc=acc, per_class_support=cm_counts.sum(axis=1),
                 checkpoint=os.path.abspath(ckpt_path), fold_id=fold_id)
        plot_scatter(Z, y, class_names, out_png)
        if args.plot_confusion:
            plot_confusion_matrix(cm_norm if cm_norm is not None else cm_counts, conf_class_names, conf_png, normalize=(cm_norm is not None))
        # Print per-class accuracy
        printable_labels = ["détaché" if n == "normal" else n for n in conf_class_names]
        acc_str = ", ".join([f"{name}: {a*100:.1f}% (n={int(s)})" for name, a, s in zip(printable_labels, acc, cm_counts.sum(axis=1))])
        print(f"Fold {fold_id} per-class acc -> {acc_str}")
        print(f"Saved fold {fold_id}: {out_png}, {conf_png}, {out_npz}")

        # accumulate
        all_y_conf.append(y_conf)
        all_y_pred_conf.append(y_pred_conf)
        if all_conf_class_names is None:
            all_conf_class_names = list(conf_class_names)

    # Combined confusion across all folds (concatenate predictions)
    if len(all_y_conf) > 0 and all_conf_class_names is not None:
        y_all = np.concatenate(all_y_conf, axis=0)
        y_pred_all = np.concatenate(all_y_pred_conf, axis=0)
        cm_all, cm_norm_all = compute_confusion_matrix(y_all, y_pred_all, class_names=all_conf_class_names, normalize=True)
        acc_all = per_class_accuracy(cm_all)
        all_conf_png = os.path.join(args.out_dir, 'confusion_note_technique_all_folds.png')
        all_conf_npz = os.path.join(args.out_dir, 'confusion_note_technique_all_folds.npz')
        if args.plot_confusion:
            plot_confusion_matrix(cm_norm_all if cm_norm_all is not None else cm_all, all_conf_class_names, all_conf_png, normalize=(cm_norm_all is not None))
        np.savez(all_conf_npz, cm=cm_all, cm_norm=cm_norm_all, confusion_class_names=np.array(all_conf_class_names),
                 per_class_acc=acc_all, per_class_support=cm_all.sum(axis=1),
                 checkpoints=np.array([os.path.abspath(p) for p in args.checkpoints]), fold_ids=np.array(args.fold_ids))
        printable_labels = ["détaché" if n == "normal" else n for n in all_conf_class_names]
        acc_str = ", ".join([f"{name}: {a*100:.1f}% (n={int(s)})" for name, a, s in zip(printable_labels, acc_all, cm_all.sum(axis=1))])
        print(f"All-folds per-class acc -> {acc_str}")
        print(f"Saved all-folds confusion: {all_conf_png}, {all_conf_npz}")


if __name__ == '__main__':
    main()


