"""Base class for signal modulation."""

from abc import ABC, abstractmethod
import random
import numpy as np
from scipy.signal.windows import hann
from ..utils.parameters import SignalParameters


class Modulation(ABC):
    """Base class for signal modulation."""

    def __init__(self, params: SignalParameters):
        """
        Initialize modulation with signal parameters.

        Args:
            params: Signal generation parameters
        """
        self.params = params

    @staticmethod
    def select_number_of_states() -> int:
        """
        Select the number of states for signal modulation.

        Returns:
            int: Number of states (2, 4, 8, 16, 32, or 64)
        """
        return random.choice([2, 4, 8, 16, 32, 64])

    @abstractmethod
    def generate_symbol(self, start_idx: int, end_idx: int) -> np.ndarray:
        """
        Generate a symbol for the given time range.

        Args:
            start_idx: Start index of the symbol
            end_idx: End index of the symbol

        Returns:
            Complex signal for the symbol
        """
        raise NotImplementedError("Subclasses must implement generate_symbol")

    @staticmethod
    def apply_fade_window(
        signal: np.ndarray, start_idx: int, end_idx: int, sample_rate: int
    ) -> None:
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
            signal[start_idx : start_idx + fade_length] *= fade_window
            signal[end_idx - fade_length : end_idx] *= fade_window[::-1]

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

    def generate_signal(self) -> np.ndarray:
        """
        Generate a complete modulated signal.

        Returns:
            Complex modulated signal
        """
        start_sample, end_sample = self.params.generate_signal_timing()
        signal = np.zeros_like(self.params.time_signal, dtype=np.complex128)

        # Symbol duration based on bandwidth
        samples_per_symbol = int(self.params.sample_rate / self.params.bandwidth)

        for start_idx in range(start_sample, end_sample, samples_per_symbol):
            end_idx = min(start_idx + samples_per_symbol, end_sample)
            if end_idx <= start_idx:
                continue

            symbol = self.generate_symbol(start_idx, end_idx)
            signal[start_idx:end_idx] = symbol
            self.apply_fade_window(signal, start_idx, end_idx, self.params.sample_rate)

        signal = self.normalize_signal(signal)
        # Apply bandpass filter to baseband signal
        filtered_signal = self.params.apply_bandpass_filter(signal)
        # Modulate with carrier frequency
        return filtered_signal * np.exp(1j * self.params.carrier_phase)
