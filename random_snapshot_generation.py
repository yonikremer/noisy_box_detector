"""
Signal Generation and Visualization Module

This module provides functionality to generate various types of modulated signals
(FSK, PSK, QAM, ASK) and visualize them in both time and frequency domains.
"""
from dataclasses import dataclass
import random
from datetime import timedelta
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
from scipy.fft import fft, fftshift


@dataclass
class SignalParameters:
    """Parameters for signal generation."""
    sample_rate: int  # Hz
    snapshot_duration: timedelta
    carrier_frequency: float  # Hz
    
    def __post_init__(self):
        """Pre-compute common values used in signal generation."""
        self.samples_in_snapshot = int(self.sample_rate * self.snapshot_duration.total_seconds())
        self.time_signal = np.linspace(0, self.snapshot_duration.total_seconds(), 
                             self.samples_in_snapshot, 
                             endpoint=False)
        self.snapshot_duration_ms = self.snapshot_duration.total_seconds() * 1000
        # pre-compute 2*pi*time_signal for FSK
        self.two_pi_time_signal = 2 * np.pi * self.time_signal
        # Pre-compute carrier phase for PSK/QAM
        self.carrier_phase = self.two_pi_time_signal * self.carrier_frequency

    def generate_signal_timing(self,
                             mean_duration_ms: float = 100) -> Tuple[int, int]:
        """
        Generate start and end sample indices for a signal within a snapshot.
        
        Args:
            mean_duration_ms: Mean duration of the signal in milliseconds
        
        Returns:
            Tuple of start and end sample indices
        """
        signal_duration_ms = np.random.exponential(mean_duration_ms)
        max_start_time_ms = self.snapshot_duration_ms
        start_time_ms = random.uniform(0, max_start_time_ms)
        
        # Convert times to sample indices
        start_sample = int(start_time_ms * self.sample_rate / 1000)
        end_sample = int(min(start_time_ms + signal_duration_ms, self.snapshot_duration_ms) * self.sample_rate / 1000)
        
        return start_sample, end_sample


class SignalGenerator:
    """Base class for signal generation utilities."""
    
    @staticmethod
    def select_number_of_states() -> int:
        """
        Select the number of states for signal modulation.
        
        Returns:
            int: Number of states (2, 4, 8, 16, 32, or 64)
        """
        return random.choice([2, 4, 8, 16, 32, 64])
    
    @staticmethod
    def select_qam_number_of_states() -> int:
        """
        Select the number of states for QAM modulation.
        QAM constellations must be perfect squares (4, 16, 64, 256, etc.)
        
        Returns:
            int: Number of QAM states (4, 16, or 64)
        """
        return random.choice([4, 16, 64])  # Common QAM constellations

    @staticmethod
    def apply_fade_window(signal: np.ndarray, start_idx: int, end_idx: int, 
                         sample_rate: int) -> None:
        """
        Apply fade-in and fade-out windows to a signal segment.
        
        Args:
            signal: The signal to modify
            start_idx: Start index of the segment
            end_idx: End index of the segment
            sample_rate: Sampling rate in Hz
        """
        fade_length = min(int(0.01 * sample_rate), (end_idx - start_idx) // 2)
        if fade_length > 0:
            fade_window = hann(2 * fade_length)[:fade_length]
            signal[start_idx:start_idx + fade_length] *= fade_window
            signal[end_idx - fade_length:end_idx] *= fade_window[::-1]
    
    @staticmethod
    def normalize_signal(signal: np.ndarray) -> np.ndarray:
        """
        Normalize a signal to have unit power.
        
        Args:
            signal: The signal to normalize
            
        Returns:
            Normalized signal with unit power
        """
        # Calculate signal power
        power = np.mean(np.abs(signal) ** 2)
        if power > 0:
            return signal / power
        return signal


def generate_random_fsk_signal(params: SignalParameters) -> np.ndarray:
    """
    Generate a complex signal that randomly switches between different frequencies.
    
    Args:
        params: Signal generation parameters
        
    Returns:
        Generated complex signal
    """
    signal_bandwidth = random.gauss(20_000, 5_000)
    start_sample, end_sample = params.generate_signal_timing()
    num_frequencies = SignalGenerator.select_number_of_states()
    signal = np.zeros_like(params.time_signal, dtype=np.complex128)

    spacing = signal_bandwidth / (num_frequencies - 1)
    frequencies = params.carrier_frequency + np.arange(num_frequencies) * spacing
    samples_per_symbol = max(1, int(spacing * params.sample_rate))

    for start_idx in range(start_sample, end_sample, samples_per_symbol):
        end_idx = min(start_idx + samples_per_symbol, end_sample)
        if end_idx <= start_idx:
            continue
        frequency = np.random.choice(frequencies)
        phase = params.two_pi_time_signal[start_idx:end_idx] * frequency
        signal[start_idx:end_idx] = np.exp(1j * phase)
        SignalGenerator.apply_fade_window(signal, start_idx, end_idx, params.sample_rate)

    return SignalGenerator.normalize_signal(signal)


def generate_random_psk_signal(params: SignalParameters) -> np.ndarray:
    """
    Generate a complex signal that randomly switches between different phase states.
    
    Args:
        params: Signal generation parameters
        
    Returns:
        Generated complex signal
    """
    start_sample, end_sample = params.generate_signal_timing()
    num_phase_states = SignalGenerator.select_number_of_states()
    phase_states = np.linspace(0, 2 * np.pi, num_phase_states, endpoint=False)
    signal = np.zeros_like(params.time_signal, dtype=np.complex128)
    
    symbol_duration = max(10, int(params.sample_rate / (params.carrier_frequency / 10)))
    
    for start_idx in range(start_sample, end_sample, symbol_duration):
        end_idx = min(start_idx + symbol_duration, end_sample)
        if end_idx <= start_idx:
            continue
            
        phase = np.random.choice(phase_states)
        signal[start_idx:end_idx] = np.exp(1j * (params.carrier_phase[start_idx:end_idx] + phase))
        
        SignalGenerator.apply_fade_window(signal, start_idx, end_idx, params.sample_rate)
    
    return SignalGenerator.normalize_signal(signal)


def generate_random_qam_signal(params: SignalParameters) -> np.ndarray:
    """
    Generate a complex signal using Quadrature Amplitude Modulation (QAM).
    QAM constellations are arranged in a square grid, so the number of states
    must be a perfect square (4, 16, 64, etc.).
    
    Args:
        params: Signal generation parameters
        
    Returns:
        Generated complex signal
    """
    start_sample, end_sample = params.generate_signal_timing()
    num_states = SignalGenerator.select_qam_number_of_states()
    
    side_length = int(np.sqrt(num_states))  # This will always be an integer
    constellation = np.zeros(num_states, dtype=np.complex128)
    
    # Create square constellation
    for i in range(side_length):
        for j in range(side_length):
            x = (2 * i - (side_length - 1)) / (side_length - 1)
            y = (2 * j - (side_length - 1)) / (side_length - 1)
            constellation[i * side_length + j] = x + 1j * y
    
    constellation = SignalGenerator.normalize_signal(constellation)
    
    signal = np.zeros_like(params.time_signal, dtype=np.complex128)
    symbol_duration = max(10, int(params.sample_rate / (params.carrier_frequency / 10)))
    
    for start_idx in range(start_sample, end_sample, symbol_duration):
        end_idx = min(start_idx + symbol_duration, end_sample)
        if end_idx <= start_idx:
            continue
        
        constellation_point = np.random.choice(constellation)
        signal[start_idx:end_idx] = constellation_point * np.exp(1j * params.carrier_phase[start_idx:end_idx])
        
        SignalGenerator.apply_fade_window(signal, start_idx, end_idx, params.sample_rate)
    
    return SignalGenerator.normalize_signal(signal)


def generate_random_ask_signal(params: SignalParameters) -> np.ndarray:
    """
    Generate a complex signal using Amplitude Shift Keying (ASK).
    
    Args:
        params: Signal generation parameters
        
    Returns:
        Generated complex signal
    """
    start_sample, end_sample = params.generate_signal_timing()
    num_levels = SignalGenerator.select_number_of_states()
    
    # Generate amplitude levels and normalize to have unit average power
    amplitude_levels = np.linspace(0.2, 1.0, num_levels)
    amplitude_levels = SignalGenerator.normalize_signal(amplitude_levels)
    
    signal = np.zeros_like(params.time_signal, dtype=np.complex128)
    symbol_duration = max(10, int(params.sample_rate / (params.carrier_frequency / 10)))
    
    for start_idx in range(start_sample, end_sample, symbol_duration):
        end_idx = min(start_idx + symbol_duration, end_sample)
        if end_idx <= start_idx:
            continue
        
        amplitude = np.random.choice(amplitude_levels)
        signal[start_idx:end_idx] = amplitude * np.exp(1j * params.carrier_phase[start_idx:end_idx])
        
        SignalGenerator.apply_fade_window(signal, start_idx, end_idx, params.sample_rate)
    
    return SignalGenerator.normalize_signal(signal)


def create_random_snapshot(snapshot_bandwidth: int, sample_rate: int, 
                         snapshot_duration: timedelta, num_signals: int) -> np.ndarray:
    """
    Create a snapshot of a signal by combining multiple signals with random frequencies.
    Each signal can be FSK, PSK, QAM, or ASK modulated.
    
    Args:
        snapshot_bandwidth: Bandwidth of the snapshot
        sample_rate: The sample rate of the snapshot
        snapshot_duration: The duration of the snapshot
        num_signals: The number of signals to combine
        
    Returns:
        Combined signal snapshot
    """
    min_frequency = -snapshot_bandwidth / 2
    max_frequency = snapshot_bandwidth / 2
    samples_in_snapshot = int(sample_rate * snapshot_duration.total_seconds())
    snapshot = np.zeros(samples_in_snapshot, dtype=np.complex128)
    modulation_types = [generate_random_fsk_signal, generate_random_psk_signal, generate_random_qam_signal, generate_random_ask_signal]
    
    for _ in range(num_signals):
        params = SignalParameters(
            sample_rate=sample_rate,
            snapshot_duration=snapshot_duration,
            carrier_frequency=random.uniform(min_frequency, max_frequency)
        )
        function_to_call = random.choice(modulation_types)
        signal = function_to_call(params)
        snapshot += signal
        
    # noise = np.random.normal(0, 0.0001, snapshot.shape)
    # snapshot += noise
    return snapshot


def plot_signal(signal: np.ndarray, sample_rate: int, title: str = "Signal") -> None:
    """
    Plot a complex signal in both time and frequency domains.
    
    Args:
        signal (np.ndarray): Complex signal to plot
        sample_rate (int): Sampling rate in Hz
        title (str): Title for the plot
    """
    # Time domain plot
    time = np.arange(len(signal)) / sample_rate
    
    plt.figure(figsize=(15, 10))
    
    # Plot real and imaginary parts
    plt.subplot(3, 1, 1)
    plt.plot(time, np.real(signal), label='Real')
    plt.plot(time, np.imag(signal), label='Imaginary')
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
    freq = np.arange(-sample_rate/2, sample_rate/2, sample_rate/len(signal))
    spectrum = fftshift(fft(signal))
    plt.plot(freq, np.abs(spectrum))
    plt.title(f"{title} - Frequency Domain")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def main():
    """Generate and plot random signals using different modulation schemes."""
    # Parameters for signal generation
    sample_rate = 1_000_000  # 1 MHz
    snapshot_duration = timedelta(milliseconds=100)  # 100 ms
    snapshot_bandwidth = 200_000  # 200 kHz
    
    # Generate time signal
    time_signal = np.linspace(0, snapshot_duration.total_seconds(), 
                             int(sample_rate * snapshot_duration.total_seconds()), 
                             endpoint=False)
    
    # Generate and plot individual signals
    for modulation_type, title in [
        (generate_random_fsk_signal, "FSK Signal"),
        (generate_random_psk_signal, "PSK Signal"),
        (generate_random_qam_signal, "QAM Signal"),
        (generate_random_ask_signal, "ASK Signal")
    ]:
        params = SignalParameters(
            sample_rate=sample_rate,
            snapshot_duration=snapshot_duration,
            carrier_frequency=random.uniform(-snapshot_bandwidth/2, snapshot_bandwidth/2)
        )
        signal = modulation_type(params)
        plot_signal(signal, sample_rate, title)
    
    # Generate and plot combined snapshot
    snapshot = create_random_snapshot(
        snapshot_bandwidth, sample_rate, snapshot_duration, num_signals=5
    )
    plot_signal(snapshot, sample_rate, "Combined Signal Snapshot")


if __name__ == "__main__":
    main()
