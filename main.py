"""Main script for generating and visualizing signals."""

import random
import numpy as np
from datetime import timedelta

from signal_generation.utils.config import load_config
from signal_generation.utils.parameters import SignalParameters
from signal_generation.utils.visualization import plot_signal
from signal_generation.modulation.fsk import FSKModulation
from signal_generation.modulation.psk import PSKModulation
from signal_generation.modulation.qam import QAMModulation
from signal_generation.modulation.ask import ASKModulation


def create_random_snapshot(signal_config: dict) -> tuple[np.ndarray, int]:
    """
    Create a snapshot of a signal by combining multiple signals with random frequencies.
    Each signal can be FSK, PSK, QAM, or ASK modulated.

    Args:
        signal_config: Dictionary containing signal generation parameters

    Returns:
        Tuple of (combined signal snapshot, sample rate)
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
        print(f"Carrier frequency: {carrier_frequency}")
        bandwidth = random.gauss(
            signal_config["mean_bandwidth"], signal_config["bandwidth_std"]
        )
        while bandwidth <= 0:
            bandwidth = random.gauss(
                signal_config["mean_bandwidth"], signal_config["bandwidth_std"]
            )
        print(f"Bandwidth: {bandwidth}")
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

    return snapshot, sample_rate


def main():
    """Generate and plot random signals using different modulation schemes."""
    # Load configuration
    config = load_config()
    signal_config = config["signal"]

    # Parameters for signal generation
    sample_rate = signal_config["sample_rate"]
    snapshot_duration = timedelta(milliseconds=signal_config["snapshot_duration_ms"])
    snapshot_bandwidth = signal_config["snapshot_bandwidth"]

    # Generate and plot individual signals
    for modulation_class, title in [
        (FSKModulation, "FSK Signal"),
        (PSKModulation, "PSK Signal"),
        (QAMModulation, "QAM Signal"),
        (ASKModulation, "ASK Signal"),
    ]:
        print(f"Modulation type: {title}")
        bandwidth = random.gauss(
            signal_config["mean_bandwidth"], signal_config["bandwidth_std"]
        )
        while bandwidth <= 0:
            bandwidth = random.gauss(
                signal_config["mean_bandwidth"], signal_config["bandwidth_std"]
            )

        carrier_frequency = random.uniform(
            -snapshot_bandwidth / 2, snapshot_bandwidth / 2
        )
        print(f"Carrier frequency: {carrier_frequency}")
        print(f"Bandwidth: {bandwidth}")
        params = SignalParameters(
            sample_rate=sample_rate,
            snapshot_duration=snapshot_duration,
            carrier_frequency=carrier_frequency,
            bandwidth=bandwidth,
            mean_signal_duration_ms=signal_config["mean_signal_duration_ms"],
        )
        modulation = modulation_class(params)
        signal = modulation.generate_signal()
        plot_signal(signal, sample_rate, title)

    # Generate and plot combined snapshot
    snapshot, sample_rate = create_random_snapshot(signal_config)
    plot_signal(snapshot, sample_rate, "Combined Signal Snapshot")


if __name__ == "__main__":
    main()
