import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'piano_transcription/utils'))
sys.path.insert(1, os.path.join(sys.path[0], 'piano_transcription/pytorch'))
import numpy as np
import argparse
import librosa
import mir_eval
import torch
import time
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from sklearn import metrics

from utilities import (create_folder, get_filename, note_to_freq, TargetProcessor, 
    RegressionPostProcessor, OnsetsFramesPostProcessor, read_midi_bach10, load_audio)
import config as config
from inference import PianoTranscription
import pandas as pd
from pathlib import Path

from data_utils import get_song_folders_bach10, get_song_folders_bvd, get_song_folders_urmp, get_ref_note_events, get_ref_note_events_bach10, get_ref_note_events_bvd, get_ref_note_events_urmp

class Evaluator(object):
    def __init__(self, dataset, output_dir, checkpoint_path, model_type='Note_pedal', 
                 device=None, post_processor_type='regression'):
        """Evaluate piano transcription on Bach10 dataset.
        
        Args:
            dataset: str, 'bach10', 'urmp', 'bvd'
            checkpoint_path: str, path to model checkpoint
            model_type: str, model type for transcription
            device: int, 0~4, None for cpu
            post_processor_type: str, 'regression' or 'onsets_frames'
        """
        self.dataset = dataset
        self.sample_rate = config.sample_rate
        self.frames_per_second = config.frames_per_second
        self.classes_num = config.classes_num
        self.begin_note = config.begin_note
        
        # Evaluation parameters matching the specified criteria
        self.onset_tolerance = 0.05  # 50ms
        self.offset_ratio = 0.2      # 20% of note duration
        self.offset_min_tolerance = 0.05
        
        # For pitch tolerance of 50 cents, we use frequency ratio
        # 50 cents = 50/1200 octaves = frequency_ratio^(1/12) where ratio = 2^(50/1200)
        self.pitch_tolerance = 2**(50/1200)  # ~1.029 frequency ratio for 50 cents
        
        # Initialize transcriptor
        device = torch.device(f"cuda:{device}") if torch.cuda.is_available() and device is not None else torch.device('cpu')
        print(f"Using device: {device}")
        segment_samples = int(config.segment_seconds * self.sample_rate)
        
        self.transcriptor = PianoTranscription(
            model_type=model_type,
            device=device,
            checkpoint_path=checkpoint_path,
            segment_samples=segment_samples,
            post_processor_type=post_processor_type
        )
        
        self.post_processor_type = post_processor_type
        
        # Get all song folders
        if self.dataset == "bach10":
            self.song_folders = get_song_folders_bach10()
            print(f"Found {len(self.song_folders)} songs in Bach10 dataset")
        elif self.dataset == "urmp":
            self.song_folders = get_song_folders_urmp()
            print(f"Found {len(self.song_folders)} songs in URMP dataset")
        elif self.dataset == "bvd":
            self.song_folders = get_song_folders_bvd()
        else:
            raise ValueError(f"Invalid dataset: {self.dataset}")
        
        self.output_dir = output_dir

        

    
    def evaluate_single_song(self, song_info):
        """Evaluate a single song and return metrics.
        
        Args:
            song_info: dict with 'audio_path', 'midi_path', 'song_name'
            
        Returns:
            dict with evaluation metrics
        """
        audio_path = song_info['audio_path']
        midi_path = song_info['midi_path']
        song_name = song_info['song_name']

        
        
        print(f"Processing: {song_name}")
        print(f"audio_path: {audio_path}")
        print(f"midi_path: {midi_path}")

        if song_info['align_path'] is not None:
            align_path = song_info['align_path']
            print(f"align_path: {align_path}")
        
        # try:
            # Load audio
        audio, _ = load_audio(audio_path, sr=self.sample_rate, mono=True)

        # save audio to wav for checking, using librosa
        # import soundfile as sf
        # sf.write("./check_audio.wav", audio, self.sample_rate)

        '''
        # Load MIDI and get ground truth events
        midi_dict = read_midi_bach10(midi_path)
        
        # Process MIDI to get ground truth note events
        target_processor = TargetProcessor(
            segment_seconds=len(audio) / self.sample_rate,
            frames_per_second=self.frames_per_second,
            begin_note=self.begin_note,
            classes_num=self.classes_num
        )
        
        # Get ground truths
        (target_dict, note_events, pedal_events) = target_processor.process(
            start_time=0,
            midi_events_time=midi_dict['midi_event_time'],
            midi_events=midi_dict['midi_event'],
            extend_pedal=True
        )
        
        # Extract ground truth note information
        ref_on_off_pairs = np.array([[event['onset_time'], event['offset_time']] for event in note_events])
        ref_midi_notes = np.array([event['midi_note'] for event in note_events])
        ref_velocity = np.array([event['velocity'] for event in note_events])
        '''

        # The last return value is velocity, but we don't use it in the evaluation
        if self.dataset == "bach10": # Bach10
            ref_on_off_pairs, ref_midi_notes, _ = get_ref_note_events_bach10(midi_path)
        elif self.dataset == "bvd": # Bach-Violin-Dataset
            ref_on_off_pairs, ref_midi_notes, _ = get_ref_note_events_bvd(midi_path, align_path)
        elif self.dataset == "urmp":
            ref_on_off_pairs, ref_midi_notes, _ = get_ref_note_events_urmp(midi_path)
        else:
            raise ValueError(f"Invalid dataset: {self.dataset}")

        
        # Transcribe audio
        print(f"Transcribing...")
        output_midi_path = self.output_dir / f"{song_name}.mid"
        transcribed_dict = self.transcriptor.transcribe(audio, midi_path=output_midi_path)
        output_dict = transcribed_dict['output_dict']
        est_note_events = transcribed_dict['est_note_events']

        # sort it by onset_time
        est_note_events.sort(key=lambda x: x['onset_time'])
        # print(f"est_note_events: {transcribed_dict['est_note_events']}")

        # Check the form of ext_note_events
        #  {'onset_time': 9.481501, 'offset_time': 10.03, 'midi_note': 76, 'velocity': 80}

        if len(est_note_events) == 0:
            print(f"Warning: No notes detected for {song_name}")
            return {
                'song_name': song_name,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'num_ref_notes': len(ref_on_off_pairs),
                'num_est_notes': 0
            }
        
        
        # Extract estimated note information from est_note_events (list of dicts)
        est_on_offs = np.array([[event['onset_time'], event['offset_time']] for event in est_note_events])
        est_midi_notes = np.array([event['midi_note'] for event in est_note_events])
        est_vels = np.array([event['velocity'] for event in est_note_events]) * config.velocity_scale

        # print(f"ref_on_off_pairs[:10]: {ref_on_off_pairs[:10]}")
        # print(f"est_on_offs[:10]: {est_on_offs[:10]}")

        #  Still Debugging:Make sure all estimated and reference interval durations must be strictly positive, if not, remove it
        # and also remove the corresponding est_note_events
        # Remove reference notes with non-positive duration
        ref_durations = ref_on_off_pairs[:, 1] - ref_on_off_pairs[:, 0]
        valid_ref_mask = ref_durations > 0
        if not np.all(valid_ref_mask):
            print(f"Removing {np.sum(~valid_ref_mask)} reference notes with non-positive duration")
        ref_on_off_pairs = ref_on_off_pairs[valid_ref_mask]
        ref_midi_notes = ref_midi_notes[valid_ref_mask]
        # If you have ref_velocity, also filter it
        # ref_velocity = ref_velocity[valid_ref_mask]

        # Remove estimated notes with non-positive duration
        est_durations = est_on_offs[:, 1] - est_on_offs[:, 0]
        valid_est_mask = est_durations > 0
        if not np.all(valid_est_mask):
            print(f"Removing {np.sum(~valid_est_mask)} estimated notes with non-positive duration")
        est_on_offs = est_on_offs[valid_est_mask]
        est_midi_notes = est_midi_notes[valid_est_mask]
        est_vels = est_vels[valid_est_mask]
        # Also filter est_note_events to keep only valid ones
        est_note_events = [event for i, event in enumerate(est_note_events) if valid_est_mask[i]]
        

        # Convert MIDI notes to frequencies for evaluation
        # print(f"ref_midi_notes[:10]: {ref_midi_notes[:10]}")
        # print(f"est_midi_notes[:10]: {est_midi_notes[:10]}")
        ref_freqs = note_to_freq(ref_midi_notes) if self.dataset != "urmp" else ref_midi_notes # URMP's ref_midi_notes is already in frequency format
        est_freqs = note_to_freq(est_midi_notes)

        print(f"ref_freqs[:10]: {ref_freqs[:10]}")
        print(f"est_freqs[:10]: {est_freqs[:10]}")
        
        # Calculate metrics using mir_eval with specified tolerances
        if len(ref_on_off_pairs) > 0 and len(est_on_offs) > 0:
            precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals=ref_on_off_pairs, 
                ref_pitches=ref_freqs,
                # ref_velocities=ref_velocity,
                est_intervals=est_on_offs,
                est_pitches=est_freqs,
                # est_velocities=est_vels,
                onset_tolerance=self.onset_tolerance,
                offset_ratio=self.offset_ratio,
                offset_min_tolerance=self.offset_min_tolerance,
                # velocity_tolerance=0.1  # velocity tolerance (not specified in requirements)
            )

            _, _, f1_wo_offset, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals=ref_on_off_pairs,
                ref_pitches=ref_freqs,
                est_intervals=est_on_offs,
                est_pitches=est_freqs,
                onset_tolerance=self.onset_tolerance,
                offset_ratio=None,  # No offset ratio
                offset_min_tolerance=None
            )
        else:
            print(f"Warning: No notes detected for {song_name}")
            precision, recall, f1 = 0.0, 0.0, 0.0
        
        result = {
            'song_name': song_name,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f1_wo_offset': f1_wo_offset,
            'num_ref_notes': len(ref_on_off_pairs),
            'num_est_notes': len(est_on_offs)
        }
        
        print(f"{song_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, F1_wo_offset={f1_wo_offset:.3f} "
                f"(Ref: {len(ref_on_off_pairs)}, Est: {len(est_on_offs)})")
        
        return result
        
        # except Exception as e:
        #     print(f"Error processing {song_name}: {str(e)}")
        #     return {
        #         'song_name': song_name,
        #         'precision': 0.0,
        #         'recall': 0.0,
        #         'f1': 0.0,
        #         'num_ref_notes': 0,
        #         'num_est_notes': 0,
        #         'error': str(e)
        #     }
    
    def evaluate_all(self, parallel=True):
        """Evaluate all songs in the Bach10 dataset.
        
        Args:
            parallel: bool, whether to use parallel processing
            
        Returns:
            dict with overall metrics and per-song results
        """
        print(f"Evaluating {len(self.song_folders)} songs...")
        
        if parallel and len(self.song_folders) > 1:
            # Use parallel processing
            with ProcessPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(self.evaluate_single_song, self.song_folders))
        else:
            # Sequential processing
            results = [self.evaluate_single_song(song_info) for song_info in self.song_folders]
        
        # Calculate overall statistics
        valid_results = [r for r in results if 'error' not in r]
        
        if len(valid_results) == 0:
            print("No valid results obtained!")
            return None
        
        overall_precision = np.mean([r['precision'] for r in valid_results])
        overall_recall = np.mean([r['recall'] for r in valid_results])
        overall_f1 = np.mean([r['f1'] for r in valid_results])
        overall_f1_wo_offset = np.mean([r['f1_wo_offset'] for r in valid_results])
        
        total_ref_notes = sum([r['num_ref_notes'] for r in valid_results])
        total_est_notes = sum([r['num_est_notes'] for r in valid_results])
        
        print("\n" + "="*50)
        print("OVERALL RESULTS:")
        print("="*50)
        print(f"Number of songs evaluated: {len(valid_results)}")
        print(f"Total reference notes: {total_ref_notes}")
        print(f"Total estimated notes: {total_est_notes}")
        print(f"Average Precision: {overall_precision:.4f}")
        print(f"Average Recall: {overall_recall:.4f}")
        print(f"Average F1-score: {overall_f1:.4f}")
        print(f"Average F1-score without offset: {overall_f1_wo_offset:.4f}")
        print("="*50)
        
        # Show per-song results
        print("\nPER-SONG RESULTS:")
        print("-"*70)
        print(f"{'Song Name':<25} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'F1-score without offset':<10} {'Ref/Est':<10}")
        print("-"*70)
        for result in valid_results:
            ref_est = f"{result['num_ref_notes']}/{result['num_est_notes']}"
            print(f"{result['song_name']:<25} {result['precision']:<10.3f} "
                  f"{result['recall']:<10.3f} {result['f1']:<10.3f} {result['f1_wo_offset']:<10.3f} {ref_est:<10}")
        
        # Show failed songs
        failed_results = [r for r in results if 'error' in r]
        if failed_results:
            print("\nFAILED SONGS:")
            for result in failed_results:
                print(f"- {result['song_name']}: {result['error']}")
        
        return {
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'overall_f1': overall_f1,
            'overall_f1_wo_offset': overall_f1_wo_offset,
            'total_ref_notes': total_ref_notes,
            'total_est_notes': total_est_notes,
            'num_songs': len(valid_results),
            'per_song_results': valid_results,
            'failed_results': failed_results
        }



# Example usage of f1_wo_offset
# precision, recall, f1 = f1_wo_offset(ref_on_off_pairs, ref_freqs, est_on_offs, est_freqs, self.onset_tolerance, self.offset_min_tolerance)


def main():
    parser = argparse.ArgumentParser(description='Evaluate piano transcription on BVD dataset')
    parser.add_argument('--dataset', type=str, required=True,
                       help='dataset to evaluate', choices=['bach10', 'urmp'])
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='Note_pedal',
                       help='Model type for transcription')
    parser.add_argument('--device', type=str, default=None, help='Device id to use')
    parser.add_argument('--post_processor_type', type=str, default='regression',
                       choices=['regression', 'onsets_frames'],
                       help='Post processor type')
    parser.add_argument('--parallel', action='store_true', default=False,
                       help='Use parallel processing')
    parser.add_argument('--model_tag', type=str, default='',
                       help='Model tag')
    parser.add_argument('--output_dir', type=str, default='./evaluations',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint {args.checkpoint_path} does not exist!")
        return
    
    # Create a subdirectory for the dataset under the evaluations folder
    dataset_output_dir = Path(args.output_dir) / args.dataset / args.model_tag
    if not dataset_output_dir.exists():
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

    # Update output_dir to include dataset subdirectory
    args.output_dir = dataset_output_dir

    # Create evaluator
    evaluator = Evaluator(
        dataset=args.dataset,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        model_type=args.model_type,
        device=args.device,
        post_processor_type=args.post_processor_type,
    )
    
    # Run evaluation
    start_time = time.time()
    results = evaluator.evaluate_all(parallel=args.parallel)
    elapsed_time = time.time() - start_time
    
    if results:
        print(f"\nEvaluation completed in {elapsed_time:.2f} seconds")
        
        # Add checkpoint path to results
        results['checkpoint_path'] = args.checkpoint_path

        # Save results to file
        import json
        
        with open(args.output_dir / 'eval.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {args.output_dir / 'eval.json'}")


if __name__ == '__main__':
    main() 