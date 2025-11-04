import os
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


def load_npz_confusion(npz_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    data = np.load(npz_path, allow_pickle=True)
    cm_norm = data.get('cm_norm')
    cm = data.get('cm')
    class_names = data.get('confusion_class_names', data.get('class_names'))
    if isinstance(class_names, np.ndarray):
        class_names = class_names.tolist()
    if cm_norm is None and cm is None:
        raise ValueError(f"No confusion matrix found in {npz_path}")
    return cm, cm_norm, class_names


def ensure_same_labels(mats: List[np.ndarray], labels_list: List[List[str]]) -> Tuple[List[np.ndarray], List[str]]:
    # Use intersection order from the first file
    base_labels = labels_list[0]
    for labels in labels_list[1:]:
        if labels != base_labels:
            # align by label names intersection
            label_to_idx = {l: i for i, l in enumerate(labels)}
            common = [l for l in base_labels if l in label_to_idx]
            if not common:
                raise ValueError("No common labels across inputs")
            aligned = []
            for M, L in zip(mats, labels_list):
                idx_map = {l: i for i, l in enumerate(L)}
                sel = [idx_map[l] for l in common]
                M_sel = M[np.ix_(sel, sel)]
                aligned.append(M_sel)
            return aligned, common
    return mats, base_labels


def average_confusions(npz_paths: List[str], ignore_class: str = 'no_technique') -> Tuple[np.ndarray, List[str]]:
    matrices = []
    labels_list = []
    for path in npz_paths:
        cm_raw, cm_norm, labels = load_npz_confusion(path)
        # prefer normalized if present
        M = cm_norm if cm_norm is not None else cm_raw.astype(np.float64) / np.maximum(cm_raw.sum(axis=1, keepdims=True), 1)
        labels = ["detache" if n == "normal" else n for n in labels]
        if ignore_class and ignore_class in labels:
            idx = labels.index(ignore_class)
            M = np.delete(np.delete(M, idx, axis=0), idx, axis=1)
            labels = [l for i, l in enumerate(labels) if i != idx]
        matrices.append(M)
        labels_list.append(labels)

    matrices, labels = ensure_same_labels(matrices, labels_list)
    avg = np.mean(np.stack(matrices, axis=0), axis=0)
    return avg, labels


def plot_confusion(M: np.ndarray, labels: List[str], out_png: str, title: str = 'Average Technique Confusion (normalized)'):
    plt.figure(figsize=(8, 6), dpi=150)
    im = plt.imshow(M, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha='right')
    plt.yticks(tick_marks, labels)

    fmt = '.2f'
    thresh = M.max() / 2.0 if M.size > 0 else 0.5
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            value = M[i, j]
            plt.text(j, i, format(value, fmt), ha="center", va="center",
                     color="white" if value > thresh else "black", fontsize=10)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Average confusion matrices across folds and plot')
    parser.add_argument('--inputs', type=str, nargs='+', required=True,
                        help='List of NPZ files or a glob pattern; use shell expansion for globs')
    parser.add_argument('--ignore_confusion_class', type=str, default='no_technique',
                        help='Class to exclude before averaging. Empty to keep all.')
    parser.add_argument('--out_png', type=str, default='avg_confusion_note_technique.png')
    parser.add_argument('--out_npz', type=str, default='avg_confusion_note_technique.npz')
    args = parser.parse_args()

    # Expand any globs (shell may already expand, but be robust)
    expanded = []
    for item in args.inputs:
        if any(ch in item for ch in ['*', '?', '[']):
            expanded.extend(sorted(glob.glob(item)))
        else:
            expanded.append(item)
    npz_paths = [p for p in expanded if os.path.isfile(p)]
    if not npz_paths:
        raise FileNotFoundError('No input NPZ files matched')

    avg_cm, labels = average_confusions(npz_paths, ignore_class=args.ignore_confusion_class.strip())
    plot_confusion(avg_cm, labels, args.out_png)

    np.savez(args.out_npz, avg_cm=avg_cm, labels=np.array(labels), inputs=np.array(npz_paths))
    print(f"Saved averaged confusion plot to {args.out_png} and data to {args.out_npz}")


if __name__ == '__main__':
    main()






