"""
This module contains:
- Function to generate a random FSK signal
- Function to generate a random PSK signal
- Function to create a snapshot of a signal by combining multiple FSK signals with different carrier frequencies
"""
import random
from datetime import timedelta

import numpy as np
from scipy.signal.windows import hann


def _select_number_of_states() -> int:
    """
    Select the number of states for signal modulation.
    For FSK: number of frequencies
    For PSK: number of phase states
    
    Returns:
        int: Number of states (2, 4, 8, 16, 32, or 64)
    """
    # Common modulation schemes: 2, 4, 8, 16, 32, 64 states
    # Higher number of states = more bits per symbol but more sensitive to noise
    return random.choice([2, 4, 8, 16, 32, 64])


def _generate_signal_timing(snapshot_duration: timedelta, mean_duration_ms: float = 100) -> tuple[timedelta, timedelta]:
    """
    Generate start and end times for a signal within a snapshot.
    
    Args:
        snapshot_duration (timedelta): Total duration of the snapshot
        mean_duration_ms (float): Mean duration of the signal in milliseconds
        
    Returns:
        tuple[timedelta, timedelta]: Start and end times for the signal
    """
    snapshot_duration_milliseconds = snapshot_duration.total_seconds() * 1000
    
    # Generate signal duration with exponential distribution favoring shorter signals
    signal_duration_ms = np.random.exponential(mean_duration_ms)
    
    # Random start time anywhere in the snapshot
    max_start_time_ms = snapshot_duration_milliseconds
    start_time_milliseconds = random.uniform(0, max_start_time_ms)
    start_time = timedelta(milliseconds=start_time_milliseconds)
    
    # Calculate end time, ensuring it doesn't exceed snapshot duration
    end_time_milliseconds = min(start_time_milliseconds + signal_duration_ms, snapshot_duration_milliseconds)
    end_time = timedelta(milliseconds=end_time_milliseconds)
    
    return start_time, end_time


def _apply_fade_window(signal: np.ndarray, start_idx: int, end_idx: int, sample_rate: int) -> None:
    """
    Apply fade-in and fade-out windows to a signal segment.
    
    Args:
        signal (np.ndarray): The signal to modify
        start_idx (int): Start index of the segment
        end_idx (int): End index of the segment
        sample_rate (int): Sampling rate in Hz
    """
    fade_length = min(int(0.01 * sample_rate), (end_idx - start_idx) // 2)
    if fade_length > 0:
        fade_window = hann(2 * fade_length)[:fade_length]
        signal[start_idx:start_idx + fade_length] *= fade_window
        signal[end_idx - fade_length:end_idx] *= fade_window[::-1]


def generate_random_fsk_signal(sample_rate: int, snapshot_duration: timedelta, min_frequency: float,
                               max_frequency: float) -> np.ndarray:
    """
    Generate a complex signal that randomly switches between four equally spaced frequencies.

    Args:
        sample_rate (int): Sampling rate in Hz.
        snapshot_duration (timedelta): Total duration of the snapshot in seconds.
        min_frequency (float): Minimum frequency in Hz.
        max_frequency (float): Maximum frequency in Hz.

    Returns:
        np.ndarray: The generated complex signal.
    """
    base_frequency = random.uniform(min_frequency, max_frequency)
    signal_bandwidth = random.gauss(20_000, 5_000)
    
    start_time, end_time = _generate_signal_timing(snapshot_duration)
    
    num_samples = int(sample_rate * snapshot_duration.total_seconds())
    num_frequencies = _select_number_of_states()

    time_signal = np.linspace(0, snapshot_duration.total_seconds(), num_samples, endpoint=False)
    signal = np.zeros_like(time_signal, dtype=np.complex128)

    spacing = signal_bandwidth / (num_frequencies - 1)  # Calculate the spacing between frequencies
    frequencies = [base_frequency + i * spacing for i in range(num_frequencies)]  # Define the frequencies
    samples_per_symbol = max(1, int(spacing * sample_rate))  # Ensure at least 1 sample per symbol

    signal_first_sample = int(start_time.total_seconds() * sample_rate)
    signal_last_sample = int(end_time.total_seconds() * sample_rate)

    for start_idx in range(signal_first_sample, signal_last_sample, samples_per_symbol):
        end_idx = min(start_idx + samples_per_symbol, signal_last_sample)
        if end_idx <= start_idx:
            continue

        frequency = np.random.choice(frequencies)
        sine_wave = np.sin(2 * np.pi * frequency * time_signal[start_idx:end_idx])
        cosine_wave = np.cos(2 * np.pi * frequency * time_signal[start_idx:end_idx])
        signal[start_idx:end_idx] = cosine_wave + 1j * sine_wave

        _apply_fade_window(signal, start_idx, end_idx, sample_rate)

    return signal


def generate_random_psk_signal(sample_rate: int, snapshot_duration: timedelta, min_frequency: float,
                             max_frequency: float) -> np.ndarray:
    """
    Generate a complex signal that randomly switches between different phase states.

    Args:
        sample_rate (int): Sampling rate in Hz.
        snapshot_duration (timedelta): Total duration of the snapshot in seconds.
        min_frequency (float): Minimum frequency in Hz.
        max_frequency (float): Maximum frequency in Hz.

    Returns:
        np.ndarray: The generated complex signal.
    """
    # Select carrier frequency
    carrier_frequency = random.uniform(min_frequency, max_frequency)
    
    start_time, end_time = _generate_signal_timing(snapshot_duration)
    
    num_samples = int(sample_rate * snapshot_duration.total_seconds())
    
    # Number of phase states (2 for BPSK, 4 for QPSK, 8 for 8-PSK, etc.)
    num_phase_states = _select_number_of_states()
    phase_states = np.linspace(0, 2 * np.pi, num_phase_states, endpoint=False)
    
    time_signal = np.linspace(0, snapshot_duration.total_seconds(), num_samples, endpoint=False)
    signal = np.zeros_like(time_signal, dtype=np.complex128)
    
    # Calculate symbol duration (ensure at least 10 samples per symbol)
    symbol_duration = max(10, int(sample_rate / (carrier_frequency / 10)))
    
    signal_first_sample = int(start_time.total_seconds() * sample_rate)
    signal_last_sample = int(end_time.total_seconds() * sample_rate)
    
    for start_idx in range(signal_first_sample, signal_last_sample, symbol_duration):
        end_idx = min(start_idx + symbol_duration, signal_last_sample)
        if end_idx <= start_idx:
            continue
            
        # Randomly select a phase state
        phase = np.random.choice(phase_states)
        
        # Generate the complex signal with the selected phase
        t = time_signal[start_idx:end_idx]
        signal[start_idx:end_idx] = np.exp(1j * (2 * np.pi * carrier_frequency * t + phase))
        
        _apply_fade_window(signal, start_idx, end_idx, sample_rate)
    
    return signal


def create_random_snapshot(snapshot_bandwidth: int, sample_rate: int, snapshot_duration: timedelta,
                           num_signals: int) -> np.ndarray:
    """
    Create a snapshot of a signal by combining multiple signals with random frequencies.
    :param snapshot_bandwidth: Bandwidth of the snapshot
    :param sample_rate: The sample rate of the snapshot
    :param snapshot_duration: The duration of the snapshot
    :param num_signals: The number of signals to combine
    :return:
    """
    min_frequency = -snapshot_bandwidth / 2
    max_frequency = snapshot_bandwidth / 2
    samples_in_snapshot = int(sample_rate * snapshot_duration.total_seconds())
    snapshot = np.zeros(samples_in_snapshot, dtype=np.complex128)
    for _ in range(num_signals):
        snapshot += generate_random_fsk_signal(sample_rate, snapshot_duration, min_frequency, max_frequency)
    noise = np.random.normal(0, 0.0001, snapshot.shape)  # Add a small random noise
    snapshot += noise
    return snapshot
