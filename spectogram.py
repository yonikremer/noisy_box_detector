"""
This script generates a snapshot of a spectrum with randomly generated FSK signals and plots its spectrogram.
"""
from datetime import timedelta
import random

import matplotlib.pyplot as plt
from scipy.signal import ShortTimeFFT, get_window
import numpy as np

from random_snapshot_generation import create_random_snapshot


def create_spectrogram(snapshot: np.ndarray, sample_rate: int):
    if not np.iscomplexobj(snapshot):
        raise ValueError("Input snapshot must be a numpy array of complex values.")

    window_size = 2 ** 14  # Define the window size
    window = get_window('hann', window_size)  # Hann window with size 2^14
    hop_size = window_size // 2  # 50% overlap

    stft = ShortTimeFFT(win=window, hop=hop_size, fs=sample_rate, fft_mode='centered')  # Initialize ShortTimeFFT
    spectrogram = stft.spectrogram(snapshot)  # Compute the spectrogram
    frequencies = stft.f  # Get frequency axis
    times = stft.t(len(snapshot))  # Get time axis

    plot_spectrogram(frequencies, spectrogram, times)


def plot_spectrogram(frequencies, spectrogram, times) -> None:
    """
    Plots the spectrogram using a faster rendering method.

    Parameters:
    - frequencies: Array of frequency values.
    - spectrogram: 2D array of spectrogram values (frequency x time).
    - times: Array of time values.
    """
    plt.figure(figsize=(10, 6))
    extent = [times[0], times[-1], frequencies[0], frequencies[-1]]
    plt.imshow(np.log10(spectrogram), aspect='auto', origin='lower', extent=extent, cmap='Grays')  # Convert to dB scale
    plt.colorbar(label='Power (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.show()


def main():
    random.seed(0)
    sample_rate: int = 1_000_000  # Sampling rate in Hz
    snapshot_bandwidth: int = sample_rate
    snapshot_duration = timedelta(seconds=2)  # Total snapshot duration
    num_signals: int = 10
    snapshot = create_random_snapshot(snapshot_bandwidth, sample_rate, snapshot_duration, num_signals)
    create_spectrogram(snapshot=snapshot, sample_rate=sample_rate)


if __name__ == "__main__":
    main()
