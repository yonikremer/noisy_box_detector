"""
This script generates a snapshot of a spectrum with randomly generated FSK signals and plots its spectrogram.
"""

import matplotlib.pyplot as plt
from scipy.signal import ShortTimeFFT, get_window
import numpy as np

# Constants for spectrogram generation
WINDOW_SIZE = 2**14  # Define the window size (16384 samples)
HOP_SIZE = WINDOW_SIZE // 2  # 50% overlap (8192 samples)
WINDOW_TYPE = "hann"  # Hann window type
FFT_MODE = "centered"  # FFT mode for ShortTimeFFT

def calculate_time_bins(signal_length: int) -> int:
    """
    Calculate the expected number of time bins in a spectrogram.
    
    The formula accounts for:
    - Window size and hop size
    - Need to cover the entire signal
    - Each hop moves by HOP_SIZE samples
    - Need to include both start and end points
    - Need to account for the initial window size
    
    Args:
        signal_length: Length of the input signal in samples
        
    Returns:
        Number of expected time bins in the spectrogram
    """
    return int(np.ceil((signal_length + WINDOW_SIZE - HOP_SIZE) / HOP_SIZE))

def create_spectrogram(snapshot: np.ndarray, sample_rate: int):
    if not isinstance(snapshot, np.ndarray):
        raise TypeError("Input snapshot must be a numpy array.")
    if not np.iscomplexobj(snapshot):
        raise ValueError("Input snapshot must be a numpy array of complex values.")

    window = get_window(WINDOW_TYPE, WINDOW_SIZE)  # Hann window with size 2^14

    stft = ShortTimeFFT(
        win=window, hop=HOP_SIZE, fs=sample_rate, fft_mode=FFT_MODE
    )  # Initialize ShortTimeFFT
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
    plt.imshow(
        np.log10(spectrogram),
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="Grays",
    )  # Convert to dB scale
    plt.colorbar(label="Power (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram")
    plt.tight_layout()
    plt.show()
