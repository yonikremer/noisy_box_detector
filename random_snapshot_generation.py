"""
This module contains:
- Function to generate a random FSK signal
- Function to create a snapshot of a signal by combining multiple FSK signals with different carrier frequencies
"""
import random
from datetime import timedelta

import numpy as np
from scipy.signal.windows import hann


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
    snapshot_duration_milliseconds = snapshot_duration.total_seconds() * 1000
    
    # Generate signal duration with exponential distribution favoring shorter signals
    # Mean duration of 100ms, but can be longer with decreasing probability
    mean_duration_ms = 100
    signal_duration_ms = np.random.exponential(mean_duration_ms)
    
    # Random start time anywhere in the snapshot
    max_start_time_ms = snapshot_duration_milliseconds
    start_time_milliseconds = random.uniform(0, max_start_time_ms)
    start_time = timedelta(milliseconds=start_time_milliseconds)
    
    # Calculate end time, ensuring it doesn't exceed snapshot duration
    end_time_milliseconds = min(start_time_milliseconds + signal_duration_ms, snapshot_duration_milliseconds)
    end_time = timedelta(milliseconds=end_time_milliseconds)
    
    num_samples = int(sample_rate * snapshot_duration.total_seconds())
    num_frequencies = random.choice([2, 4, 8, 16, 32, 64])

    time_signal = np.linspace(0, snapshot_duration.total_seconds(), num_samples, endpoint=False)
    signal = np.zeros_like(time_signal, dtype=np.complex128)

    spacing = signal_bandwidth / (num_frequencies - 1)  # Calculate the spacing between frequencies
    frequencies = [base_frequency + i * spacing for i in range(num_frequencies)]  # Define the frequencies
    samples_per_symbol = max(1, int(spacing * sample_rate))  # Ensure at least 1 sample per symbol

    signal_first_sample = int(start_time.total_seconds() * sample_rate)
    signal_last_sample = int(end_time.total_seconds() * sample_rate)

    for start_idx in range(signal_first_sample, signal_last_sample, samples_per_symbol):  # Generate the signal
        end_idx = min(start_idx + samples_per_symbol, signal_last_sample)  # Ensure end_idx does not exceed bounds
        if end_idx <= start_idx: continue  # Skip invalid symbols

        frequency = np.random.choice(frequencies)
        sine_wave = np.sin(2 * np.pi * frequency * time_signal[start_idx:end_idx])
        cosine_wave = np.cos(2 * np.pi * frequency * time_signal[start_idx:end_idx])
        signal[start_idx:end_idx] = cosine_wave + 1j * sine_wave

        fade_length = min(int(0.01 * sample_rate), (end_idx - start_idx) // 2)  # Apply fade-in and fade-out
        if fade_length > 0:
            fade_window = hann(2 * fade_length)[:fade_length]  # Adjust fade window size dynamically
            signal[start_idx:start_idx + fade_length] *= fade_window  # Apply fade-in
            signal[end_idx - fade_length:end_idx] *= fade_window[::-1]  # Apply fade-out

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
