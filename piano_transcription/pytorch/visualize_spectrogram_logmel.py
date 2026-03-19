#!/usr/bin/env python3
"""
Visualize spectrogram and logmel of WAV files using the same parameters
as in models_contrast.py (Regress_onset_offset_frame_velocity_CRNN).
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import soundfile as sf

# Add parent for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))

from torchlibrosa.stft import Spectrogram, LogmelFilterBank


def get_extractors(frames_per_second=100):
    """Build Spectrogram and LogmelFilterBank with same params as models_contrast.py"""
    sample_rate = 16000
    window_size = 2048
    hop_size = sample_rate // frames_per_second
    mel_bins = 229
    fmin = 30
    fmax = sample_rate // 2

    window = 'hann'
    center = True
    pad_mode = 'reflect'
    ref = 1.0
    amin = 1e-10
    top_db = None

    spectrogram_extractor = Spectrogram(
        n_fft=window_size,
        hop_length=hop_size,
        win_length=window_size,
        window=window,
        center=center,
        pad_mode=pad_mode,
        freeze_parameters=True,
    )

    logmel_extractor = LogmelFilterBank(
        sr=sample_rate,
        n_fft=window_size,
        n_mels=mel_bins,
        fmin=fmin,
        fmax=fmax,
        ref=ref,
        amin=amin,
        top_db=top_db,
        freeze_parameters=True,
    )

    return spectrogram_extractor, logmel_extractor


def load_and_resample(wav_path, target_sr=16000):
    """Load WAV and resample to target sample rate if needed."""
    waveform, sr = sf.read(wav_path)
    if len(waveform.shape) > 1:
        waveform = waveform.mean(axis=1)

    if sr != target_sr:
        import librosa
        waveform = librosa.resample(waveform.astype(np.float32), orig_sr=sr, target_sr=target_sr)

    return waveform.astype(np.float32)


def visualize_wav(
    wav_paths,
    output_dir=None,
    frames_per_second=100,
    show=True,
    dpi=300,
    figsize=(14.0, 9.0),
):
    """Visualize spectrogram and logmel for each WAV file."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spectrogram_extractor, logmel_extractor = get_extractors(frames_per_second)
    spectrogram_extractor = spectrogram_extractor.to(device)
    logmel_extractor = logmel_extractor.to(device)

    for wav_path in wav_paths:
        if not os.path.isfile(wav_path):
            print(f"Skip (not found): {wav_path}")
            continue

        name = os.path.splitext(os.path.basename(wav_path))[0]
        waveform = load_and_resample(wav_path)
        waveform_t = torch.from_numpy(waveform).unsqueeze(0).to(device)

        with torch.no_grad():
            spec = spectrogram_extractor(waveform_t)
            logmel = logmel_extractor(spec)

        # torchlibrosa outputs (time_steps, freq_or_mel_bins)
        spec_np = spec[0, 0].cpu().numpy()
        logmel_np = logmel[0, 0].cpu().numpy()

        # For display, use log scale for spectrogram (magnitude)
        spec_db = 10 * np.log10(spec_np + 1e-10)
        logmel_db = logmel_np  # already in log scale

        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Spectrogram
        im0 = axes[0].imshow(
            spec_db.T,
            aspect='auto',
            origin='lower',
            cmap='magma',
            extent=[0, spec_np.shape[0] / frames_per_second, 0, 8000],
            interpolation='none',
        )
        axes[0].set_ylabel('Frequency (Hz)')
        axes[0].set_title(f'{name} - Spectrogram (STFT magnitude, dB)')
        plt.colorbar(im0, ax=axes[0], label='dB')

        # Logmel
        im1 = axes[1].imshow(
            logmel_db.T,
            aspect='auto',
            origin='lower',
            cmap='viridis',
            extent=[0, logmel_np.shape[0] / frames_per_second, 0, 229],
            interpolation='none',
        )
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Mel bins')
        axes[1].set_title(f'{name} - Logmel')
        plt.colorbar(im1, ax=axes[1], label='Log mel')

        plt.tight_layout()

        if output_dir:
            out_path = os.path.join(output_dir, f'{name}_spectrogram_logmel.png')
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(out_path, dpi=dpi, bbox_inches='tight', format='png')
            print(f"Saved: {out_path}")

        if show:
            plt.show()
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize spectrogram and logmel of WAV files')
    parser.add_argument('wav_files', nargs='+', help='WAV file path(s)')
    parser.add_argument('--output-dir', '-o', default=None, help='Directory to save figures')
    parser.add_argument('--fps', type=int, default=100, help='Frames per second (default: 100)')
    parser.add_argument('--no-show', action='store_true', help='Do not display, only save to file')
    parser.add_argument('--dpi', type=int, default=300, help='Output image DPI (default: 300)')
    parser.add_argument('--figsize', nargs=2, type=float, default=[14.0, 9.0],
                        metavar=('WIDTH', 'HEIGHT'),
                        help='Figure size in inches, e.g. --figsize 18 12')
    args = parser.parse_args()

    visualize_wav(
        args.wav_files,
        output_dir=args.output_dir,
        frames_per_second=args.fps,
        show=not args.no_show,
        dpi=args.dpi,
        figsize=(args.figsize[0], args.figsize[1]),
    )


if __name__ == '__main__':
    main()
