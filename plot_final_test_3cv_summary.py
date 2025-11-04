#!/usr/bin/env python3
"""
Read final_test_0915_3CV_summary.csv and generate bar charts with error bars.

Outputs PNG files for:
- overall (mean ± std)
- balanced (mean ± std)
- per-class: flageolet, normal, pizzicato, spiccato (mean ± std)

Usage:
  python plot_final_test_3cv_summary.py \
    --input /home/yuehpo/coding/violin-mamba/final_test_0915_3CV_summary.csv \
    --out-dir /home/yuehpo/coding/violin-mamba/plots
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.cm as cm


METRICS = [
    ("overall", "Overall Accuracy"),
    ("balanced", "Balanced Accuracy"),
    ("flageolet", "Flageolet Accuracy"),
    ("normal", "Normal Accuracy"),
    ("pizzicato", "Pizzicato Accuracy"),
    ("spiccato", "Spiccato Accuracy"),
]


def read_summary_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def plot_bar_with_error(rows: List[Dict[str, str]], metric_key: str, metric_label: str, out_path: str, sort_by_acc: bool) -> None:
    mean_key = f"{metric_key}_mean"
    std_key = f"{metric_key}_std"

    # Ordering
    if sort_by_acc:
        # Sort by mean descending (best to worst)
        rows_reordered = sorted(
            rows,
            key=lambda r: float(r.get(mean_key) or 0.0),
            reverse=True,
        )
    else:
        # Keep requested fixed order: put no_transcription_features last
        rows_reordered = [r for r in rows if r.get("setting") != "no_transcription_features"]
        rows_reordered += [r for r in rows if r.get("setting") == "no_transcription_features"]

    settings = [r["setting"] for r in rows_reordered]
    means = [float(r[mean_key]) if r.get(mean_key) not in (None, "") else 0.0 for r in rows_reordered]
    stds = [float(r[std_key]) if r.get(std_key) not in (None, "") else 0.0 for r in rows_reordered]

    def short_label(s: str) -> str:
        mapping = {
            "with_transcription_features_all": "ALL",
            "with_transcription_features_exclude_onset": "excl. Onset",
            "with_transcription_features_exclude_offset": "excl. Offset",
            "with_transcription_features_exclude_frame": "excl. Frame",
            "with_transcription_features_exclude_velocity": "excl. Velocity",
            "no_transcription_features": "None",
        }
        return mapping.get(s, s)

    labels = [short_label(s) for s in settings]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(settings))
    # Create a dark-to-light blue gradient for bar colors
    n = len(settings)
    # Use modern colormap API to avoid deprecation warnings
    cmap = matplotlib.colormaps.get_cmap("Blues_r")  # reversed so index 0 is darker
    colors = [cmap(0.2 + 0.6 * (i / (n - 1 if n > 1 else 1))) for i in range(n)]
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="#2b3a67")

    ax.set_title(metric_label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot bar charts with error bars from summary CSV")
    parser.add_argument("--input", required=True, help="Path to summary CSV")
    parser.add_argument("--out-dir", required=True, help="Directory to write PNGs to")
    parser.add_argument("--sort-by-acc", action="store_true", help="Sort bars best-to-worst by mean accuracy of the plotted metric")
    args = parser.parse_args()

    rows = read_summary_csv(args.input)

    for metric_key, metric_label in METRICS:
        out_path = os.path.join(args.out_dir, f"{metric_key}.png")
        plot_bar_with_error(rows, metric_key, metric_label, out_path, args.sort_by_acc)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()


