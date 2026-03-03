import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import argparse
import time
import csv
import glob
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch

from inference import PianoTranscription
from infer_note_technique_from_midi import (
    load_note_model, slice_waveform_by_notes, infer_batch, 
    write_csv, DEFAULT_TECHNIQUE_CLASSES
)
from utilities import load_audio, write_events_to_midi_with_technique, get_filename
import config


def get_rwc_technique(filename: str) -> Optional[str]:
    """從 RWC 檔名縮寫推斷技巧標籤"""
    mapping = {
        'FLM': 'flageolet',
        'NOM': 'normal',
        'PZP': 'pizzicato',
        'SPF': 'spiccato'
    }
    for abbr, full_name in mapping.items():
        if abbr in filename.upper():
            return full_name
    return None


def process_file(
    audio_path: str,
    transcriptor_obj: PianoTranscription,
    note_model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    class_names: List[str]
) -> Optional[np.ndarray]:
    """處理單一檔案並回傳預測索引陣列"""
    sample_rate = config.sample_rate
    frames_per_second = config.frames_per_second
    
    base_name = get_filename(audio_path)
    output_csv = os.path.join(args.output_dir, f"{base_name}_techniques.csv")
    output_midi = os.path.join(args.output_dir, f"{base_name}_transcribed_with_techniques.mid")

    # 1. Load Audio
    try:
        waveform, sr = load_audio(audio_path, sr=sample_rate, mono=True)
        # Normalize full waveform before transcription
        if waveform.size > 0:
            waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    except Exception as e:
        print(f"  [Error] Failed to load {audio_path}: {e}")
        return None
    
    # 2. Transcription
    transcribed_dict = transcriptor_obj.transcribe(waveform, midi_path=None)
    note_events = transcribed_dict['est_note_events']

    if not note_events:
        print(f"  [Skipped] No notes detected in {base_name}")
        return None

    # 3. Technique Classification
    # Slice waveform per note
    chunks, _ = slice_waveform_by_notes(waveform, sample_rate, note_events, pad_seconds=args.pad_seconds)
    
    # Normalize each note chunk to [-1, 1]
    chunks = [c / (np.max(np.abs(c)) + 1e-8) if len(c) > 0 else c for c in chunks]

    transcriptor_model = transcriptor_obj.model if args.use_trans_features else None

    # Inference
    pred_idx, pred_conf = infer_batch(
        model=note_model,
        device=device,
        wave_chunks=chunks,
        transcriptor=transcriptor_model,
        use_trans_features=args.use_trans_features,
        trans_features_list=args.trans_features_list,
    )

    # 4. Save results
    for i, event in enumerate(note_events):
        event['technique'] = int(pred_idx[i])
        event['confidence'] = float(pred_conf[i])
    
    write_csv(output_csv, note_events, pred_idx, pred_conf, class_names)
    write_events_to_midi_with_technique(
        start_time=0, 
        note_events=note_events, 
        pedal_events=transcribed_dict['est_pedal_events'], 
        midi_path=output_midi
    )
    
    return pred_idx


def main():
    parser = argparse.ArgumentParser(description='End-to-end transcription and technique classification with Evaluation')
    parser.add_argument('--audio_path', type=str, required=True, help='Path to audio file or directory')
    parser.add_argument('--transcriptor_checkpoint', type=str, required=True)
    parser.add_argument('--note_model_checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_trans_features', action='store_true', default=False)
    parser.add_argument('--trans_features_list', nargs='+', default=None)
    parser.add_argument('--pad_seconds', type=float, default=0.02)
    parser.add_argument('--model_type', type=str, default='Regress_onset_offset_frame_velocity_CRNN')
    parser.add_argument('--gt_technique', type=str, default=None, help='Force a GT technique for all files, or "auto" for RWC mapping')

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}") if (args.device is not None and args.device >= 0 and torch.cuda.is_available()) else torch.device('cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    class_names = DEFAULT_TECHNIQUE_CLASSES
    trans_feat_dim = 88 * (4 if args.trans_features_list is None else len(args.trans_features_list))

    # 一次性載入模型
    print("Loading models...")
    transcriptor_obj = PianoTranscription(model_type=args.model_type, checkpoint_path=args.transcriptor_checkpoint, device=device)
    note_model = load_note_model(args.note_model_checkpoint, device, config.frames_per_second, args.use_trans_features, trans_feat_dim)

    # 收集檔案
    if os.path.isdir(args.audio_path):
        audio_files = sorted(glob.glob(os.path.join(args.audio_path, "*.[wW][aA][vV]")) + 
                             glob.glob(os.path.join(args.audio_path, "*.[fF][lL][aA][cC]")) +
                             glob.glob(os.path.join(args.audio_path, "*.[mM][pP]3")))
    else:
        audio_files = [args.audio_path]

    print(f"Total files to process: {len(audio_files)}")
    
    # 統計儲存: {class_idx: {'correct': 0, 'total': 0}}
    metrics = {i: {'correct': 0, 'total': 0} for i in range(len(class_names))}

    start_time_all = time.time()
    for i, path in enumerate(audio_files):
        fname = os.path.basename(path)
        print(f"[{i+1}/{len(audio_files)}] Processing: {fname}")
        
        # 決定此檔案的 GT
        gt_name = None
        if args.gt_technique == 'auto':
            gt_name = get_rwc_technique(fname)
        elif args.gt_technique:
            gt_name = args.gt_technique
            
        gt_idx = -1
        if gt_name:
            for idx, name in enumerate(class_names):
                if name.lower() == gt_name.lower():
                    gt_idx = idx
                    break

        pred_idx = process_file(path, transcriptor_obj, note_model, device, args, class_names)
        
        if pred_idx is not None and gt_idx != -1:
            correct = np.sum(pred_idx == gt_idx)
            total = len(pred_idx)
            metrics[gt_idx]['correct'] += correct
            metrics[gt_idx]['total'] += total
            print(f"    -> GT: {class_names[gt_idx]} | Accuracy: {correct/total:.2%} ({correct}/{total})")

    # 最終總結
    if any(m['total'] > 0 for m in metrics.values()):
        print("\n" + "="*60)
        print("FINAL EVALUATION SUMMARY")
        print("="*60)
        print(f"{'Technique':15} | {'Accuracy (Recall)':20} | {'Note Count':10}")
        print("-" * 60)
        
        all_accs = []
        total_correct = 0
        total_notes = 0
        
        for i, name in enumerate(class_names):
            # 'no_technique' 通常不納入技巧分類的平均計算
            if name == 'no_technique': continue
            
            correct = metrics[i]['correct']
            total = metrics[i]['total']
            
            if total > 0:
                acc = correct / total
                all_accs.append(acc)
                print(f"{name:15} | {acc:18.2%} | {total:10}")
            else:
                print(f"{name:15} | {'N/A':>18} | {total:10}")
                
            total_correct += correct
            total_notes += total

        print("-" * 60)
        overall_acc = total_correct / total_notes if total_notes > 0 else 0
        macro_acc = np.mean(all_accs) if all_accs else 0
        
        print(f"{'OVERALL ACC':15} | {overall_acc:18.2%} | {total_notes:10}")
        print(f"{'MACRO ACC':15} | {macro_acc:18.2%} | (Avg of classes)")
        print("="*60 + "\n")

    print(f"Total time elapsed: {time.time() - start_time_all:.2f}s")


if __name__ == '__main__':
    main()
