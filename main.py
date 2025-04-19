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
from signal_generation.snapshot import create_random_snapshot


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
