#!/usr/bin/env python3
"""
Parse a final_test_0915_3CV-style log and output:
1) A CSV summary with 3-fold means and standard deviations per setting
2) A raw per-fold CSV

Input file format example lines:
  root        : INFO     final_test_acc=0.6467
  root        : INFO     final_test_acc_flageolet=0.9000 (27/30)
  ...
  root        : INFO     final_test_acc_balanced=0.7105
  -------------Done for note_level_tech_no_transcription_features_fold_0_0915_3CV fold 0-----------------------------

Rows are identified by the trailing "Done for ... fold N" marker. We aggregate three folds per setting.

Summary CSV columns:
  setting, overall_mean, overall_std, balanced_mean, balanced_std,
  flageolet_mean, flageolet_std, normal_mean, normal_std,
  pizzicato_mean, pizzicato_std, spiccato_mean, spiccato_std

Raw per-fold CSV columns:
  setting, fold, overall, balanced, flageolet, normal, pizzicato, spiccato

Usage:
  python parse_final_test_3cv_to_csv.py \
    --input /home/yuehpo/coding/violin-mamba/final_test_0915_3CV.txt \
    --output /home/yuehpo/coding/violin-mamba/final_test_0915_3CV_summary.csv \
    --raw-output /home/yuehpo/coding/violin-mamba/final_test_0915_3CV_raw.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple


FOLD_DONE_RE = re.compile(r"Done for (.+?) fold (\d+)")


@dataclass
class FoldMetrics:
    setting: str
    fold_index: int
    overall: Optional[float] = None
    balanced: Optional[float] = None
    flageolet: Optional[float] = None
    normal: Optional[float] = None
    pizzicato: Optional[float] = None
    spiccato: Optional[float] = None


def normalize_setting(raw_name: str) -> str:
    """Map raw block name to a normalized setting key.

    The raw name often includes substrings like "_fold_0_0915_3CV_" within it.
    We categorize by the presence of transcription feature markers.
    """
    name = raw_name
    # Uniform handling for classification into known buckets
    if "with_transcription_features_all" in name:
        return "with_transcription_features_all"
    if "with_transcription_features_exclude_onset" in name:
        return "with_transcription_features_exclude_onset"
    if "with_transcription_features_exclude_offset" in name:
        return "with_transcription_features_exclude_offset"
    if "with_transcription_features_exclude_frame" in name:
        return "with_transcription_features_exclude_frame"
    if "with_transcription_features_exclude_velocity" in name:
        return "with_transcription_features_exclude_velocity"

    # Default bucket: no transcription features
    return "no_transcription_features"


def parse_metrics_file(path: str) -> List[FoldMetrics]:
    """Parse the log file and return a list of per-fold metrics."""
    results: List[FoldMetrics] = []
    current: Dict[str, float] = {}

    def read_float_after_equals(line: str) -> Optional[float]:
        try:
            return float(line.split("=")[-1].strip().split()[0])
        except Exception:
            return None

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if "final_test_acc=" in line and "final_test_acc_" not in line:
                value = read_float_after_equals(line)
                if value is not None:
                    current["overall"] = value
                continue

            if "final_test_acc_balanced=" in line:
                value = read_float_after_equals(line)
                if value is not None:
                    current["balanced"] = value
                continue

            # Per-class metrics
            if "final_test_acc_flageolet=" in line:
                value = read_float_after_equals(line)
                if value is not None:
                    current["flageolet"] = value
                continue
            if "final_test_acc_normal=" in line:
                value = read_float_after_equals(line)
                if value is not None:
                    current["normal"] = value
                continue
            if "final_test_acc_pizzicato=" in line:
                value = read_float_after_equals(line)
                if value is not None:
                    current["pizzicato"] = value
                continue
            if "final_test_acc_spiccato=" in line:
                value = read_float_after_equals(line)
                if value is not None:
                    current["spiccato"] = value
                continue

            # Fold boundary: finalize a record
            if "Done for" in line and " fold " in line:
                m = FOLD_DONE_RE.search(line)
                if not m:
                    continue
                raw_name, fold_str = m.group(1), m.group(2)
                setting = normalize_setting(raw_name)
                try:
                    fold_index = int(fold_str)
                except ValueError:
                    continue

                # Only finalize if we collected at least the overall metric
                if "overall" in current:
                    results.append(
                        FoldMetrics(
                            setting=setting,
                            fold_index=fold_index,
                            overall=current.get("overall"),
                            balanced=current.get("balanced"),
                            flageolet=current.get("flageolet"),
                            normal=current.get("normal"),
                            pizzicato=current.get("pizzicato"),
                            spiccato=current.get("spiccato"),
                        )
                    )
                current = {}

    return results


def mean(values: List[Optional[float]]) -> Optional[float]:
    numeric = [v for v in values if isinstance(v, (int, float))]
    if not numeric:
        return None
    return sum(numeric) / float(len(numeric))


def stddev(values: List[Optional[float]], sample: bool = True) -> Optional[float]:
    numeric = [v for v in values if isinstance(v, (int, float))]
    n = len(numeric)
    if n == 0:
        return None
    if sample and n < 2:
        return  None
    m = sum(numeric) / float(n)
    var = sum((x - m) ** 2 for x in numeric) / float(n - 1 if sample else n)
    return var ** 0.5


def aggregate_means(per_fold: List[FoldMetrics]) -> List[Dict[str, Optional[float]]]:
    grouped: Dict[str, List[FoldMetrics]] = defaultdict(list)
    for item in per_fold:
        grouped[item.setting].append(item)

    aggregates: List[Dict[str, Optional[float]]] = []
    for setting, rows in grouped.items():
        aggregates.append(
            {
                "setting": setting,
                "overall_mean": mean([r.overall for r in rows]),
                "overall_std": stddev([r.overall for r in rows]),
                "balanced_mean": mean([r.balanced for r in rows]),
                "balanced_std": stddev([r.balanced for r in rows]),
                "flageolet_mean": mean([r.flageolet for r in rows]),
                "flageolet_std": stddev([r.flageolet for r in rows]),
                "normal_mean": mean([r.normal for r in rows]),
                "normal_std": stddev([r.normal for r in rows]),
                "pizzicato_mean": mean([r.pizzicato for r in rows]),
                "pizzicato_std": stddev([r.pizzicato for r in rows]),
                "spiccato_mean": mean([r.spiccato for r in rows]),
                "spiccato_std": stddev([r.spiccato for r in rows]),
            }
        )

    # Stable ordering by setting name
    aggregates.sort(key=lambda d: d["setting"])  # type: ignore[index]
    return aggregates


def write_summary_csv(output_path: str, aggregates: List[Dict[str, Optional[float]]]) -> None:
    fieldnames = [
        "setting",
        "overall_mean", "overall_std",
        "balanced_mean", "balanced_std",
        "flageolet_mean", "flageolet_std",
        "normal_mean", "normal_std",
        "pizzicato_mean", "pizzicato_std",
        "spiccato_mean", "spiccato_std",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregates:
            # Format floats to 4 decimal places if present
            formatted = {}
            for k, v in row.items():
                if isinstance(v, float):
                    formatted[k] = f"{v:.4f}"
                else:
                    formatted[k] = v
            writer.writerow(formatted)


def write_raw_csv(output_path: str, per_fold: List[FoldMetrics]) -> None:
    fieldnames = [
        "setting",
        "fold",
        "overall",
        "balanced",
        "flageolet",
        "normal",
        "pizzicato",
        "spiccato",
    ]
    # Sort for stability
    sorted_rows = sorted(per_fold, key=lambda r: (r.setting, r.fold_index))
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted_rows:
            def fmt(x: Optional[float]) -> Optional[str]:
                return f"{x:.4f}" if isinstance(x, float) else None
            writer.writerow({
                "setting": r.setting,
                "fold": r.fold_index,
                "overall": fmt(r.overall),
                "balanced": fmt(r.balanced),
                "flageolet": fmt(r.flageolet),
                "normal": fmt(r.normal),
                "pizzicato": fmt(r.pizzicato),
                "spiccato": fmt(r.spiccato),
            })


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse 3-fold CV log and output summary and raw CSVs")
    parser.add_argument("--input", required=True, help="Path to final_test_... log file")
    parser.add_argument("--output", required=True, help="Path to write summary CSV")
    parser.add_argument("--raw-output", required=False, help="Path to write raw per-fold CSV")
    args = parser.parse_args()

    per_fold = parse_metrics_file(args.input)
    if not per_fold:
        raise SystemExit("No fold metrics parsed. Please check the input file format.")

    aggregates = aggregate_means(per_fold)
    write_summary_csv(args.output, aggregates)

    if args.raw_output:
        write_raw_csv(args.raw_output, per_fold)
        print(
            f"Wrote summary CSV with {len(aggregates)} settings to: {args.output}\n"
            f"Wrote raw per-fold CSV with {len(per_fold)} rows to: {args.raw_output}"
        )
    else:
        print(f"Wrote summary CSV with {len(aggregates)} settings to: {args.output}")


if __name__ == "__main__":
    main()


