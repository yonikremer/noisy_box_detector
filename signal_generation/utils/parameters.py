"""Signal parameters and timing utilities."""

from dataclasses import dataclass
import random
from datetime import timedelta
import numpy as np
from scipy.signal import butter, filtfilt


MILLISECONDS_PER_SECOND = 1000


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

        self.samples_in_snapshot = int(
            self.sample_rate * self.snapshot_duration.total_seconds()
        )
        self.time_signal = np.linspace(
            0,
            self.snapshot_duration.total_seconds(),
            self.samples_in_snapshot,
            endpoint=False,
        )
        self.snapshot_duration_ms = (
            self.snapshot_duration.total_seconds() * MILLISECONDS_PER_SECOND
        )
        # pre-compute 2*pi*time_signal for FSK
        self.two_pi_time_signal = 2 * np.pi * self.time_signal
        # Pre-compute carrier phase for PSK/QAM
        self.carrier_phase = self.two_pi_time_signal * self.carrier_frequency

    def generate_signal_timing(self) -> tuple[int, int]:
        """
        Generate start and end sample indices for a signal within a snapshot.

        Returns:
            Tuple of start and end sample indices
        """
        signal_duration_ms = np.random.exponential(self.mean_signal_duration_ms)
        max_start_time_ms = self.snapshot_duration_ms
        start_time_ms = random.uniform(0, max_start_time_ms)

        # Convert times to sample indices
        start_sample = int(start_time_ms * self.sample_rate / MILLISECONDS_PER_SECOND)
        end_sample = int(
            min(start_time_ms + signal_duration_ms, self.snapshot_duration_ms)
            * self.sample_rate
            / MILLISECONDS_PER_SECOND
        )

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
        cutoff = (self.bandwidth / 2) / nyquist

        if cutoff <= 0:
            raise ValueError("Cutoff frequency must be greater than 0")
        if cutoff > 1:
            raise ValueError("Cutoff frequency is greater than Nyquist frequency")

        # Design and apply Butterworth filter
        order = 4  # Filter order
        b, a = butter(order, cutoff, btype="lowpass")
        return filtfilt(b, a, signal)
