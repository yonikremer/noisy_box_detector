"""Shared test fixtures."""

import pytest
from datetime import timedelta
from signal_generation.utils.parameters import SignalParameters


@pytest.fixture
def signal_params():
    """Create a basic SignalParameters instance for testing."""
    return SignalParameters(
        sample_rate=1000000,  # 1 MHz
        snapshot_duration=timedelta(milliseconds=100),  # 100 ms
        carrier_frequency=50000,  # 50 kHz
        bandwidth=20000,  # 20 kHz
        mean_signal_duration_ms=20,  # 20 ms
    )
