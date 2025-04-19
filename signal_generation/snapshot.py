"""Snapshot generation functionality."""

import random
import numpy as np
from datetime import timedelta
from .utils.parameters import SignalParameters
from .modulation.fsk import FSKModulation
from .modulation.psk import PSKModulation
from .modulation.qam import QAMModulation
from .modulation.ask import ASKModulation


def create_random_snapshot(signal_config: dict) -> np.ndarray:
    """
    Create a snapshot of a signal by combining multiple signals with random frequencies.
    Each signal can be FSK, PSK, QAM, or ASK modulated.

    Args:
        signal_config: Dictionary containing signal generation parameters

    Returns:
        Combined signal snapshot
    """
    sample_rate = signal_config["sample_rate"]
    snapshot_bandwidth = signal_config["snapshot_bandwidth"]
    snapshot_duration = timedelta(milliseconds=signal_config["snapshot_duration_ms"])
    num_signals = signal_config["num_signals"]
    min_frequency = -snapshot_bandwidth / 2
    max_frequency = snapshot_bandwidth / 2
    samples_in_snapshot = int(sample_rate * snapshot_duration.total_seconds())
    snapshot = np.zeros(samples_in_snapshot, dtype=np.complex128)

    modulation_types = [FSKModulation, PSKModulation, QAMModulation, ASKModulation]

    for _ in range(num_signals):
        carrier_frequency = random.uniform(min_frequency, max_frequency)
        bandwidth = random.gauss(
            signal_config["mean_bandwidth"], signal_config["bandwidth_std"]
        )
        while bandwidth <= 0:
            bandwidth = random.gauss(
                signal_config["mean_bandwidth"], signal_config["bandwidth_std"]
            )
        params = SignalParameters(
            sample_rate=sample_rate,
            snapshot_duration=snapshot_duration,
            carrier_frequency=carrier_frequency,
            bandwidth=bandwidth,
            mean_signal_duration_ms=signal_config["mean_signal_duration_ms"],
        )
        modulation_class = random.choice(modulation_types)
        modulation = modulation_class(params)
        signal = modulation.generate_signal()
        snapshot += signal

    return snapshot
