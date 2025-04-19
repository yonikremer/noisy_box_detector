"""Signal visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift


def plot_signal(signal: np.ndarray, sample_rate: int, title: str = "Signal") -> None:
    """
    Plot a complex signal in both time and frequency domains.

    Args:
        signal: Complex signal to plot
        sample_rate: Sampling rate in Hz
        title: Title for the plot
    """
    # Time domain plot
    time = np.arange(len(signal)) / sample_rate

    plt.figure(figsize=(15, 10))
    # Maximize the window
    plt.get_current_fig_manager().window.state("zoomed")

    # Plot real and imaginary parts
    plt.subplot(3, 1, 1)
    plt.plot(time, np.real(signal), label="Real")
    plt.plot(time, np.imag(signal), label="Imaginary")
    plt.title(f"{title} - Time Domain")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    # Plot magnitude
    plt.subplot(3, 1, 2)
    plt.plot(time, np.abs(signal))
    plt.title(f"{title} - Magnitude")
    plt.xlabel("Time (s)")
    plt.ylabel("Magnitude")
    plt.grid(True)

    # Frequency domain plot
    plt.subplot(3, 1, 3)
    freq = np.arange(-sample_rate / 2, sample_rate / 2, sample_rate / len(signal))
    spectrum = fftshift(fft(signal))
    plt.plot(freq, np.abs(spectrum))
    plt.title(f"{title} - Frequency Domain")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
