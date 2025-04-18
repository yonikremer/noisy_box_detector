"""Tests for snapshot generation functionality."""

import pytest
import numpy as np
from scipy import signal
from datetime import timedelta
from signal_generation.snapshot import create_random_snapshot
import random


@pytest.fixture
def test_config():
    """Create a test configuration dictionary."""
    return {
        "signal": {
            "sample_rate": 1000000,  # 1 MHz
            "snapshot_duration_ms": 100,  # 100 ms
            "snapshot_bandwidth": 200000,  # 200 kHz
            "num_signals": 3,  # 3 signals per snapshot
            "mean_bandwidth": 20000,  # 20 kHz
            "bandwidth_std": 5000,  # 5 kHz
            "mean_signal_duration_ms": 20,  # 20 ms
        }
    }


def test_create_random_snapshot_basic(test_config):
    """Test basic snapshot generation."""
    snapshot = create_random_snapshot(test_config["signal"])

    # Check output type and shape
    assert isinstance(snapshot, np.ndarray)
    assert snapshot.dtype == np.complex128

    # Check snapshot length
    expected_samples = int(
        test_config["signal"]["sample_rate"]
        * test_config["signal"]["snapshot_duration_ms"]
        / 1000
    )
    assert len(snapshot) == expected_samples


def test_create_random_snapshot_power(test_config):
    """Test that snapshot power is reasonable."""
    snapshot = create_random_snapshot(test_config["signal"])

    # Calculate average power
    power = np.mean(np.abs(snapshot) ** 2)

    # Power should be roughly proportional to number of signals
    # Each signal is normalized to unit power, so total power ≈ num_signals
    expected_power = test_config["signal"]["num_signals"]
    assert np.isclose(power, expected_power, rtol=0.5)


def test_create_random_snapshot_bandwidth(test_config):
    """Test that signals respect bandwidth constraints."""
    snapshot = create_random_snapshot(test_config["signal"])

    # Calculate spectrum
    spectrum = np.fft.fftshift(np.fft.fft(snapshot))
    freqs = np.fft.fftshift(
        np.fft.fftfreq(len(snapshot), 1 / test_config["signal"]["sample_rate"])
    )

    # Find power outside bandwidth
    bandwidth = test_config["signal"]["snapshot_bandwidth"]
    mask = np.abs(freqs) > bandwidth / 2
    out_of_band_power = np.mean(np.abs(spectrum[mask]) ** 2)

    # Out-of-band power should be small compared to in-band power
    in_band_power = np.mean(np.abs(spectrum[~mask]) ** 2)
    assert out_of_band_power < 0.1 * in_band_power


def test_create_random_snapshot_modulation_types(test_config):
    """Test that different modulation types are used."""
    # Generate multiple snapshots to increase chance of seeing all modulation types
    num_snapshots = 10
    modulation_types_seen = set()

    for _ in range(num_snapshots):
        snapshot = create_random_snapshot(test_config["signal"])

        # Analyze signal characteristics to detect modulation type
        # This is a simplified approach - in practice you might want more sophisticated detection
        spectrum = np.fft.fftshift(np.fft.fft(snapshot))
        phase = np.angle(snapshot)
        amplitude = np.abs(snapshot)

        # Simple heuristic for modulation type detection
        if np.std(amplitude) > 0.5:
            modulation_types_seen.add("ASK")
        if np.std(phase) > 0.5:
            modulation_types_seen.add("PSK")
        if len(np.unique(np.round(amplitude, 2))) > 2:
            modulation_types_seen.add("QAM")
        if len(np.unique(np.round(phase, 2))) > 2:
            modulation_types_seen.add("FSK")

    # Should see at least 3 different modulation types
    assert len(modulation_types_seen) >= 3


def test_create_random_snapshot_error_handling():
    """Test error handling in snapshot creation."""
    # Test invalid signal config
    with pytest.raises(KeyError):
        create_random_snapshot({})

    # Test invalid number of signals
    with pytest.raises(KeyError):
        create_random_snapshot({"sample_rate": 1e6})

    # Test invalid sample rate
    with pytest.raises(KeyError):
        create_random_snapshot({"num_signals": 1})

    # Test invalid snapshot duration
    with pytest.raises(KeyError):
        create_random_snapshot({"sample_rate": 1e6, "num_signals": 1})

    # Test invalid bandwidth
    with pytest.raises(KeyError):
        create_random_snapshot(
            {"sample_rate": 1e6, "num_signals": 1, "snapshot_duration_ms": 100}
        )


def test_create_random_snapshot_negative_bandwidth(monkeypatch):
    """Test that negative bandwidth is handled correctly."""
    # Mock random.gauss to return negative bandwidth first, then positive
    bandwidth_values = [-1000, 20000]
    bandwidth_iter = iter(bandwidth_values)
    monkeypatch.setattr(random, "gauss", lambda mean, std: next(bandwidth_iter))

    # Create a minimal test config
    test_config = {
        "sample_rate": 1e6,
        "snapshot_duration_ms": 100,
        "snapshot_bandwidth": 100000,
        "num_signals": 1,
        "mean_bandwidth": 20000,
        "bandwidth_std": 5000,
        "mean_signal_duration_ms": 20,
    }

    # Should not raise any errors
    snapshot = create_random_snapshot(test_config)
    assert snapshot is not None
