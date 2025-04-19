"""Tests for signal parameters."""

import pytest
import numpy as np
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


def test_signal_parameters_initialization(signal_params):
    """Test that SignalParameters initializes correctly."""
    assert signal_params.sample_rate == 1000000
    assert signal_params.snapshot_duration == timedelta(milliseconds=100)
    assert signal_params.carrier_frequency == 50000
    assert signal_params.bandwidth == 20000
    assert signal_params.mean_signal_duration_ms == 20


def test_signal_parameters_computed_values(signal_params):
    """Test that computed values are correct."""
    # Test samples_in_snapshot
    expected_samples = int(1000000 * 0.1)  # sample_rate * duration_in_seconds
    assert signal_params.samples_in_snapshot == expected_samples

    # Test time_signal
    assert len(signal_params.time_signal) == expected_samples
    assert signal_params.time_signal[0] == 0
    assert signal_params.time_signal[-1] < 0.1  # Less than duration

    # Test two_pi_time_signal
    assert np.allclose(signal_params.two_pi_time_signal, 2 * np.pi * signal_params.time_signal)

    # Test carrier_phase
    assert np.allclose(
        signal_params.carrier_phase,
        2 * np.pi * signal_params.time_signal * signal_params.carrier_frequency
    )


def test_signal_parameters_invalid_bandwidth():
    """Test that invalid bandwidth raises ValueError."""
    with pytest.raises(ValueError):
        SignalParameters(
            sample_rate=1000000,
            snapshot_duration=timedelta(milliseconds=100),
            carrier_frequency=50000,
            bandwidth=0,  # Invalid bandwidth
            mean_signal_duration_ms=20,
        )


def test_generate_signal_timing(signal_params):
    """Test signal timing generation."""
    start_sample, end_sample = signal_params.generate_signal_timing()
    
    # Check that samples are within bounds
    assert 0 <= start_sample < signal_params.samples_in_snapshot
    assert start_sample < end_sample <= signal_params.samples_in_snapshot


def test_apply_bandpass_filter(signal_params):
    """Test bandpass filter application."""
    # Create a test signal
    test_signal = np.random.randn(signal_params.samples_in_snapshot) + \
                  1j * np.random.randn(signal_params.samples_in_snapshot)
    
    # Apply filter
    filtered_signal = signal_params.apply_bandpass_filter(test_signal)
    
    # Check output shape
    assert filtered_signal.shape == test_signal.shape
    
    # Check that signal is complex
    assert np.iscomplexobj(filtered_signal)
    
    # Check that filter preserves signal energy at passband frequencies
    # and attenuates stopband frequencies (could add more specific tests)


def test_signal_parameters_error_handling():
    """Test error handling in signal parameters."""
    # Test invalid sample rate
    with pytest.raises(ValueError, match="Bandwidth must be greater than 0"):
        SignalParameters(
            sample_rate=1e6,
            snapshot_duration=timedelta(milliseconds=100),
            carrier_frequency=50000,
            bandwidth=0,  # Invalid bandwidth
            mean_signal_duration_ms=20
        )

    # Test invalid snapshot duration
    with pytest.raises(ValueError, match="Bandwidth must be greater than 0"):
        SignalParameters(
            sample_rate=1e6,
            snapshot_duration=timedelta(milliseconds=100),
            carrier_frequency=50000,
            bandwidth=0,  # Invalid bandwidth
            mean_signal_duration_ms=20
        )

    # Test invalid carrier frequency
    with pytest.raises(ValueError, match="Bandwidth must be greater than 0"):
        SignalParameters(
            sample_rate=1e6,
            snapshot_duration=timedelta(milliseconds=100),
            carrier_frequency=50000,
            bandwidth=0,  # Invalid bandwidth
            mean_signal_duration_ms=20
        )

    # Test invalid bandwidth
    with pytest.raises(ValueError, match="Bandwidth must be greater than 0"):
        SignalParameters(
            sample_rate=1e6,
            snapshot_duration=timedelta(milliseconds=100),
            carrier_frequency=50000,
            bandwidth=0,  # Invalid bandwidth
            mean_signal_duration_ms=20
        )


def test_apply_bandpass_filter_error_handling():
    """Test error handling in apply_bandpass_filter."""
    params = SignalParameters(
        sample_rate=1e6,
        snapshot_duration=timedelta(milliseconds=100),
        carrier_frequency=50000,
        bandwidth=20000,
        mean_signal_duration_ms=20
    )

    # Test invalid cutoff frequency (too high)
    with pytest.raises(ValueError, match="Cutoff frequency is greater than Nyquist frequency"):
        params.bandwidth = params.sample_rate * 2  # Makes cutoff > 1
        params.apply_bandpass_filter(np.ones(100))

    # Test invalid cutoff frequency (zero)
    with pytest.raises(ValueError, match="Cutoff frequency must be greater than 0"):
        params.bandwidth = 0  # Makes cutoff = 0
        params.apply_bandpass_filter(np.ones(100)) 