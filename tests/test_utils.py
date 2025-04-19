"""Tests for utility functions."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from signal_generation.utils.visualization import plot_signal
from signal_generation.utils.config import load_config


def test_plot_signal():
    """Test plot_signal function."""
    # Create a test signal
    sample_rate = 1000
    t = np.linspace(0, 1, sample_rate)
    signal = np.exp(2j * np.pi * 10 * t)  # 10 Hz complex sinusoid
    
    # Test that plotting doesn't raise any errors
    plot_signal(signal, sample_rate, "Test Signal")
    plt.close('all')  # Clean up


def test_load_config():
    """Test configuration loading."""
    config = load_config()
    
    # Check that config has required sections
    assert "signal" in config
    
    # Check signal parameters
    signal_config = config["signal"]
    required_params = [
        "sample_rate",
        "snapshot_duration_ms",
        "snapshot_bandwidth",
        "num_signals",
        "mean_bandwidth",
        "bandwidth_std",
        "mean_signal_duration_ms"
    ]
    
    for param in required_params:
        assert param in signal_config
        assert isinstance(signal_config[param], (int, float))
        
    # Check specific parameter constraints
    assert signal_config["sample_rate"] > 0
    assert signal_config["snapshot_duration_ms"] > 0
    assert signal_config["snapshot_bandwidth"] > 0
    assert signal_config["num_signals"] > 0
    assert signal_config["mean_bandwidth"] > 0
    assert signal_config["bandwidth_std"] > 0
    assert signal_config["mean_signal_duration_ms"] > 0 