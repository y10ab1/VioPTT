import numpy as np
import argparse
import csv
import os
import time
import logging
import h5py
import librosa
import logging
from pathlib import Path

from utilities import (create_folder, float32_to_int16, create_logging, 
    get_filename, read_metadata, read_midi, read_maps_midi, read_musc_midi)
import config


def pack_maestro_dataset_to_hdf5(args):
    """Load & resample MAESTRO audio files, then write to hdf5 files.

    Args:
      dataset_dir: str, directory of dataset
      workspace: str, directory of your workspace
    """

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace

    sample_rate = config.sample_rate

    # Paths
    csv_path = os.path.join(dataset_dir, 'maestro-v2.0.0.csv')
    waveform_hdf5s_dir = os.path.join(workspace, 'hdf5s', 'maestro')

    logs_dir = os.path.join(workspace, 'logs', get_filename(__file__))
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    # Read meta dict
    meta_dict = read_metadata(csv_path)

    audios_num = len(meta_dict['canonical_composer'])
    logging.info('Total audios number: {}'.format(audios_num))

    feature_time = time.time()

    # Load & resample each audio file to a hdf5 file
    for n in range(audios_num):
        logging.info('{} {}'.format(n, meta_dict['midi_filename'][n]))

        # Read midi
        midi_path = os.path.join(dataset_dir, meta_dict['midi_filename'][n])
        midi_dict = read_midi(midi_path)

        # Load audio
        audio_path = os.path.join(dataset_dir, meta_dict['audio_filename'][n])
        (audio, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

        packed_hdf5_path = os.path.join(waveform_hdf5s_dir, '{}.h5'.format(
            os.path.splitext(meta_dict['audio_filename'][n])[0]))

        create_folder(os.path.dirname(packed_hdf5_path))

        with h5py.File(packed_hdf5_path, 'w') as hf:
            hf.attrs.create('canonical_composer', data=meta_dict['canonical_composer'][n].encode(), dtype='S100')
            hf.attrs.create('canonical_title', data=meta_dict['canonical_title'][n].encode(), dtype='S100')
            hf.attrs.create('split', data=meta_dict['split'][n].encode(), dtype='S20')
            hf.attrs.create('year', data=meta_dict['year'][n].encode(), dtype='S10')
            hf.attrs.create('midi_filename', data=meta_dict['midi_filename'][n].encode(), dtype='S100')
            hf.attrs.create('audio_filename', data=meta_dict['audio_filename'][n].encode(), dtype='S100')
            hf.attrs.create('duration', data=meta_dict['duration'][n], dtype=np.float32)

            hf.create_dataset(name='midi_event', data=[e.encode() for e in midi_dict['midi_event']], dtype='S100')
            hf.create_dataset(name='midi_event_time', data=midi_dict['midi_event_time'], dtype=np.float32)
            hf.create_dataset(name='waveform', data=float32_to_int16(audio), dtype=np.int16)
        
    logging.info('Write hdf5 to {}'.format(packed_hdf5_path))
    logging.info('Time: {:.3f} s'.format(time.time() - feature_time))


def pack_maps_dataset_to_hdf5(args):
    """MAPS is a piano dataset only used for evaluating our piano transcription
    system (optional). Ref:

    [1] Emiya, Valentin. "MAPS Database A piano database for multipitch 
    estimation and automatic transcription of music. 2016

    Load & resample MAPS audio files, then write to hdf5 files.

    Args:
      dataset_dir: str, directory of dataset
      workspace: str, directory of your workspace
    """

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace

    sample_rate = config.sample_rate
    pianos = ['ENSTDkCl', 'ENSTDkAm']

    # Paths
    waveform_hdf5s_dir = os.path.join(workspace, 'hdf5s', 'maps')

    logs_dir = os.path.join(workspace, 'logs', get_filename(__file__))
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    feature_time = time.time()
    count = 0

    # Load & resample each audio file to a hdf5 file
    for piano in pianos:
        sub_dir = os.path.join(dataset_dir, piano, 'MUS')

        audio_names = [os.path.splitext(name)[0] for name in os.listdir(sub_dir) 
            if os.path.splitext(name)[-1] == '.mid']
        
        for audio_name in audio_names:
            print('{} {}'.format(count, audio_name))
            audio_path = '{}.wav'.format(os.path.join(sub_dir, audio_name))
            midi_path = '{}.mid'.format(os.path.join(sub_dir, audio_name))

            # Load audio with original sample rate first, then resample if needed
            (audio, original_sr) = librosa.core.load(audio_path, sr=None, mono=True)
            
            # Resample to target sample rate if different
            if original_sr != sample_rate:
                audio = librosa.core.resample(audio, orig_sr=original_sr, target_sr=sample_rate)
                logging.info(f'Resampled {audio_name} from {original_sr}Hz to {sample_rate}Hz')
            
            midi_dict = read_maps_midi(midi_path)
            
            packed_hdf5_path = os.path.join(waveform_hdf5s_dir, '{}.h5'.format(audio_name))
            create_folder(os.path.dirname(packed_hdf5_path))

            with h5py.File(packed_hdf5_path, 'w') as hf:
                hf.attrs.create('split', data='test'.encode(), dtype='S20')
                hf.attrs.create('midi_filename', data='{}.mid'.format(audio_name).encode(), dtype='S100')
                hf.attrs.create('audio_filename', data='{}.wav'.format(audio_name).encode(), dtype='S100')
                hf.create_dataset(name='midi_event', data=[e.encode() for e in midi_dict['midi_event']], dtype='S100')
                hf.create_dataset(name='midi_event_time', data=midi_dict['midi_event_time'], dtype=np.float32)
                hf.create_dataset(name='waveform', data=float32_to_int16(audio), dtype=np.int16)
            
            count += 1

    logging.info('Write hdf5 to {}'.format(packed_hdf5_path))
    logging.info('Time: {:.3f} s'.format(time.time() - feature_time))


def pack_mosa_dataset_to_hdf5(args):
    """Load & resample MOSA audio files, then write to hdf5 files.

    Args:
      dataset_dir: str, directory of MOSA dataset
      workspace: str, directory of your workspace
    """

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace

    sample_rate = config.sample_rate

    # Paths
    waveform_hdf5s_dir = os.path.join(workspace, 'hdf5s', 'mosa')

    logs_dir = os.path.join(workspace, 'logs', get_filename(__file__))
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    feature_time = time.time()
    count = 0

    # Process both 'ev' and 'yv' directories
    for split_dir in ['ev', 'yv']:
        split_path = os.path.join(dataset_dir, split_dir)
        if not os.path.exists(split_path):
            continue
        
        # Iterate through songs (ba1, ba3, ba4, etc.)
        for song_name in os.listdir(split_path):
            song_path = os.path.join(split_path, song_name)
            if not os.path.isdir(song_path):
                continue
                
            # Determine split based on song name: ba3 is test, others are train
            split_name = 'test' if song_name == 'ba3' else 'train'
                
            # Iterate through events (ev01, ev02, etc.)
            for event in os.listdir(song_path):
                event_path = os.path.join(song_path, event)
                if not os.path.isdir(event_path):
                    continue
                    
                # Iterate through takes (t1, t2, etc.)
                for take in os.listdir(event_path):
                    take_path = os.path.join(event_path, take)
                    if not os.path.isdir(take_path):
                        continue
                        
                    # Check if audio file exists
                    audio_filename = f"{song_name}_{event}_{take}_audio.wav"
                    audio_path = os.path.join(take_path, audio_filename)
                    
                    # Check if annotation files exist
                    # note_filename = f"{song_name}_{event}_{take}_note.csv"
                    align_filename = f"{song_name}_{event}_{take}_align_notetime.csv"
                    annotations_dir = os.path.join(take_path, 'annotation', 'annotations')
                    # note_path = os.path.join(annotations_dir, note_filename)
                    align_path = os.path.join(annotations_dir, align_filename)
                    
                    
                    logging.info('{} song: {} event: {} take: {} (split: {})'.format(count, song_name, event, take, split_name))

                    # Read annotations
                    try:
                        annotation_dict = read_mosa_annotations(align_path if os.path.exists(align_path) else None)
                    except Exception as e:
                        logging.warning(f"Failed to read annotations for {song_name}_{event}_{take}: {e}")
                        continue

                    # Load audio with original sample rate first, then resample if needed
                    try:
                        (audio, original_sr) = librosa.core.load(audio_path, sr=None, mono=True)
                        
                        # Resample to target sample rate if different from 44.1kHz
                        if original_sr != sample_rate:
                            audio = librosa.core.resample(audio, orig_sr=original_sr, target_sr=sample_rate)
                            logging.info(f'Resampled {audio_filename} from {original_sr}Hz to {sample_rate}Hz')
                        
                        duration = len(audio) / sample_rate
                    except Exception as e:
                        logging.warning(f"Failed to load audio {audio_path}: {e}")
                        continue

                    # Create output filename
                    output_filename = f"{song_name}_{event}_{take}"
                    packed_hdf5_path = os.path.join(waveform_hdf5s_dir, '{}.h5'.format(output_filename))

                    create_folder(os.path.dirname(packed_hdf5_path))

                    with h5py.File(packed_hdf5_path, 'w') as hf:
                        # Store minimal metadata
                        hf.attrs.create('split', data=split_name.encode(), dtype='S20')
                        hf.attrs.create('audio_filename', data=audio_filename.encode(), dtype='S200')
                        hf.attrs.create('duration', data=duration, dtype=np.float32)
                        if os.path.exists(align_path):
                            hf.attrs.create('align_filename', data=align_filename.encode(), dtype='S200')

                        # Store MIDI-like events and audio

                        def float32_to_int16(waveform_float32: np.ndarray) -> np.ndarray:
                            """
                            Convert a float32 waveform in range [-1.0, 1.0] to int16 [-32768, 32767]
                            """
                            # Step 1: Clip to avoid overflow
                            waveform_clipped = np.clip(waveform_float32, -1.0, 1.0)

                            # Step 2: Scale to int16 range and convert
                            waveform_int16 = (waveform_clipped * 32767).astype(np.int16)

                            return waveform_int16
                        hf.create_dataset(name='midi_event', data=[e.encode() for e in annotation_dict['midi_event']], dtype='S100')
                        hf.create_dataset(name='midi_event_time', data=annotation_dict['midi_event_time'], dtype=np.float32)
                        hf.create_dataset(name='waveform', data=float32_to_int16(audio), dtype=np.int16)
                    
                    logging.info('Write hdf5 to {}'.format(packed_hdf5_path))
                    count += 1

    logging.info('Total processed files: {}'.format(count))
    logging.info('Time: {:.3f} s'.format(time.time() - feature_time))


def read_mosa_annotations(align_csv_path):
    """Parse MOSA annotation files.

    Args:
      align_csv_path: str, path to alignment CSV file

    Returns:
      annotation_dict: dict, e.g. {
        'midi_event': ['note_on channel=0 note=60 velocity=80 time=1.0', ...],
        'midi_event_time': [1.0, 1.5, 2.0, ...]}
    """
    
    midi_events = []
    midi_events_time = []
    
    # Read note CSV file
    with open(align_csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        # Expected MOSA format: note_id, onset, offset, score_position, score_position_offset, note, midi_number, note_duration, bar_id, bar_position
        for row in reader:
            if len(row) >= 7:  # At least need onset, offset, and midi_number
                try:
                    # Parse the row data
                    # note_id = int(row[0]) if row[0] else 0
                    onset_time = float(row[1])
                    offset_time = float(row[2])
                    score_position = float(row[3]) if row[3] else 0.0
                    score_position_offset = float(row[4]) if row[4] else 0.0
                    note_name = row[5] if len(row) > 5 else ''
                    midi_number = int(row[6]) if len(row) > 6 and row[6] else 60
                    note_duration = float(row[7]) if len(row) > 7 and row[7] else (offset_time - onset_time)
                    bar_id = int(row[8]) if len(row) > 8 and row[8] else -1
                    bar_position = float(row[9]) if len(row) > 9 and row[9] else 0.0
                    
                    # Use a default velocity or derive from other information
                    # For now, use a moderate velocity
                    velocity = 80
                    
                    # Validate the data
                    if onset_time >= 0 and offset_time > onset_time and 21 <= midi_number <= 108:
                        # Create note_on event
                        note_on_event = f"note_on channel=0 note={midi_number} velocity={velocity} time={onset_time}"
                        midi_events.append(note_on_event)
                        midi_events_time.append(onset_time)
                        
                        # Create note_off event
                        note_off_event = f"note_off channel=0 note={midi_number} velocity=0 time={offset_time}"
                        midi_events.append(note_off_event)
                        midi_events_time.append(offset_time)
                        
                except (ValueError, IndexError) as e:
                    # Skip invalid rows
                    print(f"Warning: Skipping invalid row in {align_csv_path}: {row}, Error: {e}")
                    continue
    
    # Sort events by time
    if midi_events_time:
        sorted_indices = np.argsort(midi_events_time)
        midi_events = [midi_events[i] for i in sorted_indices]
        midi_events_time = [midi_events_time[i] for i in sorted_indices]
    
    annotation_dict = {
        'midi_event': np.array(midi_events),
        'midi_event_time': np.array(midi_events_time)
    }
    
    return annotation_dict

def pack_musc_dataset_to_hdf5(args):
    """Load & resample MUSC audio files, then write to hdf5 files.

    Args:
      dataset_dir: str, directory of MUSC dataset
      workspace: str, directory of your workspace
    """
    import random

    # Arguments & parameters
    dataset_dir = Path(args.dataset_dir)
    workspace = Path(args.workspace)

    sample_rate = config.sample_rate

    # Paths
    waveform_hdf5s_dir = workspace / 'hdf5s' / 'musc' 

    logs_dir = workspace / 'logs' / get_filename(__file__)
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    feature_time = time.time()
    count = 0

    # Find all paired audio and MIDI files
    paired_files = []
    total_midi_data = 0
    total_match_data = 0
    
    # Check paired dataset status: find all .mid files and ensure corresponding .wav exists
    songs_dict = {}
    for midi_path in dataset_dir.glob('**/*.mid'):
        audio_path = midi_path.with_name(midi_path.stem + '_audio.wav')
        total_midi_data += 1
        if audio_path.exists():
            base_name = midi_path.stem  # filename without extension
            # Extract song name in the same loop
            filename_parts = base_name.split('_')
            if len(filename_parts) >= 2:
                song_name = filename_parts[1]  # Second component is the song name
                file_info = {
                    'audio_path': str(audio_path),
                    'midi_path': str(midi_path),
                    'audio_filename': audio_path.name,
                    'midi_filename': midi_path.name,
                    'base_name': base_name
                }
                if song_name not in songs_dict:
                    songs_dict[song_name] = []
                songs_dict[song_name].append(file_info)
                total_match_data += 1
            else:
                raise ValueError(f"Cannot determine song name for {base_name}")
        else:
            print(f"Audio file not found for {midi_path}")

    # Flatten all file_info dicts into paired_files for later processing
    paired_files = [file_info for files in songs_dict.values() for file_info in files]

    logging.info(f'Found {len(paired_files)} paired audio/MIDI files out of {total_midi_data} MIDI files')
    # Get unique song names and split them 5% test / 95% train
    song_names = list(songs_dict.keys())
    random.seed(415)  # For reproducible splits
    random.shuffle(song_names)
    
    test_size = max(1, int(len(song_names) * 0.05))  # At least 1 song for test
    test_songs = set(song_names[:test_size])
    train_songs = set(song_names[test_size:])
    
    logging.info(f'Total songs: {len(song_names)}')
    logging.info(f'Test songs: {len(test_songs)} - {sorted(test_songs)}')
    logging.info(f'Train songs: {len(train_songs)} - {sorted(train_songs)}')
    
    # Process all paired files
    for file_info in paired_files:
        filename_parts = file_info['base_name'].split('_')
        if len(filename_parts) >= 2:
            song_name = filename_parts[1]
            split_name = 'test' if song_name in test_songs else 'train'
        else:
            logging.warning(f"Cannot determine song name for {file_info['base_name']}, assigning to train")
            split_name = 'train'
        
        logging.info('{} processing: {} (split: {})'.format(count, file_info['base_name'], split_name))

        # Read annotations from MIDI file
        # try:
        annotation_dict = read_musc_midi(file_info['midi_path'])
        # except Exception as e:
        #     logging.warning(f"Failed to read annotations for {file_info['base_name']}: {e}")
        #     continue

        # Load audio with original sample rate first, then resample if needed
        try:
            (audio, original_sr) = librosa.core.load(file_info['audio_path'], sr=None, mono=True)
            
            # Resample to target sample rate if different
            if original_sr != sample_rate:
                audio = librosa.core.resample(audio, orig_sr=original_sr, target_sr=sample_rate)
                logging.info(f'Resampled {file_info["audio_filename"]} from {original_sr}Hz to {sample_rate}Hz')
            
            duration = len(audio) / sample_rate
        except Exception as e:
            logging.warning(f"Failed to load audio {file_info['audio_path']}: {e}")
            continue

        # Create output filename
        output_filename = file_info['base_name']
        packed_hdf5_path = os.path.join(waveform_hdf5s_dir, '{}.h5'.format(output_filename))

        create_folder(os.path.dirname(packed_hdf5_path))

        with h5py.File(packed_hdf5_path, 'w') as hf:
            # Store metadata
            hf.attrs.create('split', data=split_name.encode(), dtype='S20')
            hf.attrs.create('audio_filename', data=file_info['audio_filename'].encode(), dtype='S200')
            hf.attrs.create('midi_filename', data=file_info['midi_filename'].encode(), dtype='S200')
            hf.attrs.create('duration', data=duration, dtype=np.float32)
            
            # Extract additional metadata from filename if available
            if len(filename_parts) >= 3:
                hf.attrs.create('composer', data=filename_parts[0].encode(), dtype='S100')
                hf.attrs.create('song_name', data=filename_parts[1].encode(), dtype='S100')
                hf.attrs.create('performer', data=filename_parts[2].encode(), dtype='S100')

            # Store MIDI events and audio
            hf.create_dataset(name='midi_event', data=[e.encode() for e in annotation_dict['midi_event']], dtype='S100')
            hf.create_dataset(name='midi_event_time', data=annotation_dict['midi_event_time'], dtype=np.float32)
            def float32_to_int16(waveform_float32: np.ndarray) -> np.ndarray:
                """
                Convert a float32 waveform in range [-1.0, 1.0] to int16 [-32768, 32767]
                """
                # Step 1: Clip to avoid overflow
                waveform_clipped = np.clip(waveform_float32, -1.0, 1.0)

                # Step 2: Scale to int16 range and convert
                waveform_int16 = (waveform_clipped * 32767).astype(np.int16)

                return waveform_int16
            hf.create_dataset(name='waveform', data=float32_to_int16(audio), dtype=np.int16)
        
        logging.info('Write hdf5 to {}'.format(packed_hdf5_path))
        count += 1

    logging.info('Total processed files: {}'.format(count))
    logging.info('Time: {:.3f} s'.format(time.time() - feature_time))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    parser_pack_maestro = subparsers.add_parser('pack_maestro_dataset_to_hdf5')
    parser_pack_maestro.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_pack_maestro.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')

    parser_pack_maps = subparsers.add_parser('pack_maps_dataset_to_hdf5')
    parser_pack_maps.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_pack_maps.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')

    parser_pack_mosa = subparsers.add_parser('pack_mosa_dataset_to_hdf5')
    parser_pack_mosa.add_argument('--dataset_dir', type=str, required=True, help='Directory of MOSA dataset.')
    parser_pack_mosa.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')

    parser_pack_musc = subparsers.add_parser('pack_musc_dataset_to_hdf5')
    parser_pack_musc.add_argument('--dataset_dir', type=str, required=True, help='Directory of MUSC dataset.')
    parser_pack_musc.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')

    # Parse arguments
    args = parser.parse_args()
    
    if args.mode == 'pack_maestro_dataset_to_hdf5':
        pack_maestro_dataset_to_hdf5(args)
        
    elif args.mode == 'pack_maps_dataset_to_hdf5':
        pack_maps_dataset_to_hdf5(args)

    elif args.mode == 'pack_mosa_dataset_to_hdf5':
        pack_mosa_dataset_to_hdf5(args)

    elif args.mode == 'pack_musc_dataset_to_hdf5':
        pack_musc_dataset_to_hdf5(args)

    else:
        raise Exception('Incorrect arguments!')