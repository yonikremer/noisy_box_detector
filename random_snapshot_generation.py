"""
Signal Generation and Visualization Module

This module provides functionality to generate various types of modulated signals
(FSK, PSK, QAM, ASK) and visualize them in both time and frequency domains.
"""
from dataclasses import dataclass
import random
from datetime import timedelta
from typing import Tuple
import yaml

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
from scipy.fft import fft, fftshift
from scipy.signal import butter, filtfilt


def load_config(config_path: str = "data_configs.yaml") -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@dataclass
class SignalParameters:
    """Parameters for signal generation."""
    sample_rate: int  # Hz
    snapshot_duration: timedelta
    carrier_frequency: float  # Hz
    bandwidth: float  # Hz
    mean_signal_duration_ms: float  # Mean duration for exponential distribution
    
    def __post_init__(self):
        """Pre-compute common values used in signal generation."""
        if self.bandwidth <= 0:
            raise ValueError("Bandwidth must be greater than 0")
        
        self.samples_in_snapshot = int(self.sample_rate * self.snapshot_duration.total_seconds())
        self.time_signal = np.linspace(0, self.snapshot_duration.total_seconds(), 
                             self.samples_in_snapshot, 
                             endpoint=False)
        self.snapshot_duration_ms = self.snapshot_duration.total_seconds() * 1000
        # pre-compute 2*pi*time_signal for FSK
        self.two_pi_time_signal = 2 * np.pi * self.time_signal
        # Pre-compute carrier phase for PSK/QAM
        self.carrier_phase = self.two_pi_time_signal * self.carrier_frequency

    def generate_signal_timing(self) -> Tuple[int, int]:
        """
        Generate start and end sample indices for a signal within a snapshot.
        
        Returns:
            Tuple of start and end sample indices
        """
        signal_duration_ms = np.random.exponential(self.mean_signal_duration_ms)
        max_start_time_ms = self.snapshot_duration_ms
        start_time_ms = random.uniform(0, max_start_time_ms)
        
        # Convert times to sample indices
        start_sample = int(start_time_ms * self.sample_rate / 1000)
        end_sample = int(min(start_time_ms + signal_duration_ms, self.snapshot_duration_ms) * self.sample_rate / 1000)
        
        return start_sample, end_sample
    
    def apply_bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply a lowpass filter to the baseband signal.
        The signal should be centered around zero frequency.
        
        Args:
            signal: The baseband signal to filter
            
        Returns:
            Filtered signal
        """
        # Calculate normalized cutoff frequency
        nyquist = self.sample_rate / 2
        # For baseband signal, use lowpass filter with half bandwidth
        cutoff = (self.bandwidth/2) / nyquist
        
        if cutoff <= 0:
            raise ValueError("Cutoff frequency must be greater than 0")
        if cutoff > 1:
            raise ValueError("Cutoff frequency is greater than Nyquist frequency")
        
        # Design and apply Butterworth filter
        order = 4  # Filter order
        b, a = butter(order, cutoff, btype='lowpass')
        return filtfilt(b, a, signal)


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
    start_sample, end_sample = params.generate_signal_timing()
    num_frequencies = SignalGenerator.select_number_of_states()
    signal = np.zeros_like(params.time_signal, dtype=np.complex128)

    spacing = params.bandwidth / (num_frequencies - 1)
    # Generate frequencies around zero
    frequencies = np.arange(-(num_frequencies-1)/2, (num_frequencies+1)/2) * spacing
    
    # Symbol duration based on bandwidth and number of frequencies
    samples_per_symbol = int(params.sample_rate / params.bandwidth)

    for start_idx in range(start_sample, end_sample, samples_per_symbol):
        end_idx = min(start_idx + samples_per_symbol, end_sample)
        if end_idx <= start_idx:
            continue
        frequency = np.random.choice(frequencies)
        phase = params.two_pi_time_signal[start_idx:end_idx] * frequency
        signal[start_idx:end_idx] = np.exp(1j * phase)
        SignalGenerator.apply_fade_window(signal, start_idx, end_idx, params.sample_rate)

    signal = SignalGenerator.normalize_signal(signal)
    # Apply bandpass filter to baseband signal
    filtered_signal = params.apply_bandpass_filter(signal)
    # Modulate with carrier frequency
    return filtered_signal * np.exp(1j * params.carrier_phase)


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
    
    # Symbol duration based on bandwidth
    samples_per_symbol = int(params.sample_rate / params.bandwidth)
    
    for start_idx in range(start_sample, end_sample, samples_per_symbol):
        end_idx = min(start_idx + samples_per_symbol, end_sample)
        if end_idx <= start_idx:
            continue
            
        phase = np.random.choice(phase_states)
        signal[start_idx:end_idx] = np.exp(1j * phase)
        
        SignalGenerator.apply_fade_window(signal, start_idx, end_idx, params.sample_rate)
    
    signal = SignalGenerator.normalize_signal(signal)
    # Apply bandpass filter to baseband signal
    filtered_signal = params.apply_bandpass_filter(signal)
    # Modulate with carrier frequency
    return filtered_signal * np.exp(1j * params.carrier_phase)


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
    
    # Create square constellation centered around zero
    for i in range(side_length):
        for j in range(side_length):
            x = (2 * i - (side_length - 1)) / (side_length - 1)
            y = (2 * j - (side_length - 1)) / (side_length - 1)
            constellation[i * side_length + j] = x + 1j * y
    
    constellation = SignalGenerator.normalize_signal(constellation)
    
    signal = np.zeros_like(params.time_signal, dtype=np.complex128)
    
    # Symbol duration based on bandwidth and constellation size
    samples_per_symbol = int(params.sample_rate / (params.bandwidth * np.log2(num_states)))
    
    for start_idx in range(start_sample, end_sample, samples_per_symbol):
        end_idx = min(start_idx + samples_per_symbol, end_sample)
        if end_idx <= start_idx:
            continue
        
        constellation_point = np.random.choice(constellation)
        signal[start_idx:end_idx] = constellation_point
        
        SignalGenerator.apply_fade_window(signal, start_idx, end_idx, params.sample_rate)
    
    signal = SignalGenerator.normalize_signal(signal)
    # Apply bandpass filter to baseband signal
    filtered_signal = params.apply_bandpass_filter(signal)
    # Modulate with carrier frequency
    return filtered_signal * np.exp(1j * params.carrier_phase)


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
    
    # Symbol duration based on bandwidth and number of levels
    samples_per_symbol = int(params.sample_rate / (params.bandwidth * np.log2(num_levels)))
    
    for start_idx in range(start_sample, end_sample, samples_per_symbol):
        end_idx = min(start_idx + samples_per_symbol, end_sample)
        if end_idx <= start_idx:
            continue
        
        amplitude = np.random.choice(amplitude_levels)
        signal[start_idx:end_idx] = amplitude
        
        SignalGenerator.apply_fade_window(signal, start_idx, end_idx, params.sample_rate)
    
    signal = SignalGenerator.normalize_signal(signal)
    # Apply bandpass filter to baseband signal
    filtered_signal = params.apply_bandpass_filter(signal)
    # Modulate with carrier frequency
    return filtered_signal * np.exp(1j * params.carrier_phase)


def create_random_snapshot(signal_config: dict) -> np.ndarray:
    """
    Create a snapshot of a signal by combining multiple signals with random frequencies.
    Each signal can be FSK, PSK, QAM, or ASK modulated.
    
    Args:
        signal_config: Dictionary containing signal generation parameters
        
    Returns:
        Combined signal snapshot
    """
    sample_rate = signal_config['sample_rate']
    snapshot_bandwidth = signal_config['snapshot_bandwidth']
    snapshot_duration = timedelta(milliseconds=signal_config['snapshot_duration_ms'])
    num_signals = signal_config['num_signals']
    min_frequency = -snapshot_bandwidth / 2
    max_frequency = snapshot_bandwidth / 2
    samples_in_snapshot = int(sample_rate * snapshot_duration.total_seconds())
    snapshot = np.zeros(samples_in_snapshot, dtype=np.complex128)
    modulation_types = [generate_random_fsk_signal, generate_random_psk_signal, generate_random_qam_signal, generate_random_ask_signal]
    
    for _ in range(num_signals):
        carrier_frequency = random.uniform(min_frequency, max_frequency)
        print(f"Carrier frequency: {carrier_frequency}")
        bandwidth = random.gauss(signal_config['mean_bandwidth'], signal_config['bandwidth_std'])
        while bandwidth <= 0:
            bandwidth = random.gauss(signal_config['mean_bandwidth'], signal_config['bandwidth_std'])
        print(f"Bandwidth: {bandwidth}")
        params = SignalParameters(
            sample_rate=sample_rate,
            snapshot_duration=snapshot_duration,
            carrier_frequency=carrier_frequency,
            bandwidth=bandwidth,
            mean_signal_duration_ms=signal_config['mean_signal_duration_ms']
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
    # Maximize the window
    plt.get_current_fig_manager().window.state('zoomed')
    
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
    # Load configuration
    config = load_config()
    signal_config = config['signal']
    
    # Parameters for signal generation
    sample_rate = signal_config['sample_rate']
    snapshot_duration = timedelta(milliseconds=signal_config['snapshot_duration_ms'])
    snapshot_bandwidth = signal_config['snapshot_bandwidth']
    
    # Generate and plot individual signals
    for modulation_type, title in [
        (generate_random_fsk_signal, "FSK Signal"),
        (generate_random_psk_signal, "PSK Signal"),
        (generate_random_qam_signal, "QAM Signal"),
        (generate_random_ask_signal, "ASK Signal")
    ]:
        print(f"Modulation type: {title}")
        bandwidth = random.gauss(signal_config['mean_bandwidth'], signal_config['bandwidth_std'])
        while bandwidth <= 0:
            bandwidth = random.gauss(signal_config['mean_bandwidth'], signal_config['bandwidth_std'])
            
        carrier_frequency = random.uniform(-snapshot_bandwidth/2, snapshot_bandwidth/2)
        print(f"Carrier frequency: {carrier_frequency}")
        print(f"Bandwidth: {bandwidth}")
        params = SignalParameters(
            sample_rate=sample_rate,
            snapshot_duration=snapshot_duration,
            carrier_frequency=carrier_frequency,
            bandwidth=bandwidth,
            mean_signal_duration_ms=signal_config['mean_signal_duration_ms']
        )
        signal = modulation_type(params)
        plot_signal(signal, sample_rate, title)
    
    # Generate and plot combined snapshot
    snapshot = create_random_snapshot(
        signal_config=signal_config
    )
    plot_signal(snapshot, sample_rate, "Combined Signal Snapshot")


if __name__ == "__main__":
    main()
