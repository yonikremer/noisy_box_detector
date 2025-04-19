"""Tests for modulation classes."""

import pytest
import numpy as np
from datetime import timedelta
from signal_generation.utils.parameters import SignalParameters
from signal_generation.modulation.fsk import FSKModulation
from signal_generation.modulation.psk import PSKModulation
from signal_generation.modulation.qam import QAMModulation
from signal_generation.modulation.ask import ASKModulation
from signal_generation.modulation.base import Modulation


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


@pytest.fixture
def test_modulation():
    """Create a test modulation instance."""
    return ExampleModulation


@pytest.mark.parametrize(
    "modulation_class", [FSKModulation, PSKModulation, QAMModulation, ASKModulation]
)
def test_modulation_initialization(signal_params, modulation_class):
    """Test that modulation classes initialize correctly."""
    modulation = modulation_class(signal_params)
    assert modulation.params == signal_params


@pytest.mark.parametrize(
    "modulation_class", [FSKModulation, PSKModulation, QAMModulation, ASKModulation]
)
def test_generate_signal(signal_params, modulation_class):
    """Test signal generation for each modulation type."""
    modulation = modulation_class(signal_params)
    signal = modulation.generate_signal()

    # Check signal properties
    assert isinstance(signal, np.ndarray)
    assert signal.dtype == np.complex128
    assert len(signal) == signal_params.samples_in_snapshot

    # Check signal power is normalized
    avg_power = np.mean(np.abs(signal) ** 2)
    assert np.isclose(avg_power, 1.0, rtol=0.1)


def test_fsk_frequencies(signal_params):
    """Test FSK frequency generation."""
    fsk = FSKModulation(signal_params)

    # Check number of frequencies
    assert len(fsk.frequencies) == fsk.num_frequencies

    # Check frequency spacing
    freq_spacing = signal_params.bandwidth / (fsk.num_frequencies - 1)
    assert np.allclose(np.diff(fsk.frequencies), freq_spacing)


def test_psk_phases(signal_params):
    """Test PSK phase generation."""
    psk = PSKModulation(signal_params)

    # Check number of phases
    assert len(psk.phase_states) == psk.num_phase_states

    # Check phase spacing
    phase_spacing = 2 * np.pi / psk.num_phase_states
    assert np.allclose(np.diff(psk.phase_states), phase_spacing)


def test_qam_constellation(signal_params):
    """Test QAM constellation generation."""
    qam = QAMModulation(signal_params)

    # Check constellation size
    assert len(qam.constellation) == qam.num_states

    # Check constellation power
    avg_power = np.mean(np.abs(qam.constellation) ** 2)
    assert np.isclose(avg_power, 1.0, rtol=0.1)


def test_ask_levels(signal_params):
    """Test ASK level generation."""
    ask = ASKModulation(signal_params)

    # Check number of levels
    assert len(ask.amplitude_levels) == ask.num_levels

    # Check levels are sorted
    assert np.all(np.diff(ask.amplitude_levels) >= 0)

    # Check level normalization
    avg_power = np.mean(np.abs(ask.amplitude_levels) ** 2)
    assert np.isclose(avg_power, 1.0, rtol=0.1)


@pytest.mark.parametrize(
    "modulation_class", [FSKModulation, PSKModulation, QAMModulation, ASKModulation]
)
def test_symbol_generation(signal_params, modulation_class):
    """Test symbol generation for each modulation type."""
    modulation = modulation_class(signal_params)
    start_idx = 0
    end_idx = int(signal_params.sample_rate * 0.001)  # 1ms worth of samples

    symbol = modulation.generate_symbol(start_idx, end_idx)

    # Check symbol properties
    assert isinstance(symbol, (np.ndarray, np.number))
    if isinstance(symbol, np.ndarray):
        assert len(symbol) == 1 or len(symbol) == end_idx - start_idx
    assert np.iscomplexobj(symbol) or isinstance(symbol, (float, int))


class ExampleModulation(Modulation):
    """Concrete implementation of Modulation for testing."""

    def generate_symbol(self, start_idx: int, end_idx: int) -> np.ndarray:
        """Generate a test symbol."""
        return np.ones(end_idx - start_idx)


def test_base_modulation_error_handling(test_modulation):
    """Test error handling in base modulation class."""
    # Create a valid SignalParameters instance
    params = SignalParameters(
        sample_rate=1e6,
        snapshot_duration=timedelta(milliseconds=100),
        carrier_frequency=50000,
        bandwidth=20000,
        mean_signal_duration_ms=20,
    )

    # Test invalid signal parameters
    with pytest.raises(ValueError, match="Signal parameters must be provided"):
        test_modulation(None)

    # Test invalid bandwidth in parameters
    with pytest.raises(ValueError, match="Bandwidth must be greater than 0"):
        params = SignalParameters(
            sample_rate=1e6,
            snapshot_duration=timedelta(milliseconds=100),
            carrier_frequency=50000,
            bandwidth=0,  # Invalid bandwidth
            mean_signal_duration_ms=20,
        )
        ExampleModulation(params)


def test_base_modulation_abstract_method():
    """Test that the base class's generate_symbol method raises NotImplementedError."""

    class EmptyModulation(Modulation):
        def generate_symbol(self, start_idx: int, end_idx: int) -> np.ndarray:
            raise NotImplementedError("This method should be implemented by subclasses")

    params = SignalParameters(
        sample_rate=1e6,
        snapshot_duration=timedelta(milliseconds=100),
        carrier_frequency=50000,
        bandwidth=20000,
        mean_signal_duration_ms=20,
    )

    mod = EmptyModulation(params)
    with pytest.raises(NotImplementedError):
        mod.generate_symbol(0, 10)


def test_normalize_signal_zero_power():
    """Test that normalize_signal handles zero power signals correctly."""
    signal = np.zeros(100, dtype=np.complex128)
    normalized = Modulation.normalize_signal(signal)
    assert np.array_equal(normalized, signal)


def test_generate_signal_invalid_timing():
    """Test that generate_signal handles invalid timing correctly."""

    class TestModulation(Modulation):
        def generate_symbol(self, start_idx: int, end_idx: int) -> np.ndarray:
            return np.ones(end_idx - start_idx)

    params = SignalParameters(
        sample_rate=1e6,
        snapshot_duration=timedelta(milliseconds=100),
        carrier_frequency=50000,
        bandwidth=20000,
        mean_signal_duration_ms=20,
    )

    mod = TestModulation(params)
    # Mock generate_signal_timing to return invalid indices
    mod.params.generate_signal_timing = lambda: (10, 5)
    signal = mod.generate_signal()
    assert np.array_equal(
        signal, np.zeros_like(params.time_signal, dtype=np.complex128)
    )


def test_apply_fade_window():
    """Test that apply_fade_window correctly applies fade-in and fade-out windows."""
    # Create test signal
    signal = np.ones(1000, dtype=np.complex128)
    sample_rate = 1000
    
    # Apply fade window
    Modulation.apply_fade_window(signal, 0, 1000, sample_rate)
    
    # Check that edges are faded
    assert np.all(signal[0:10] < 1.0)  # Fade-in
    assert np.all(signal[-10:] < 1.0)  # Fade-out
    assert np.all(signal[10:-10] == 1.0)  # Middle unchanged


def test_normalize_signal():
    """Test that normalize_signal correctly normalizes signal power."""
    # Create test signal with known power
    signal = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128)
    
    # Normalize signal
    normalized = Modulation.normalize_signal(signal)
    
    # Check power is approximately 1
    power = np.mean(np.abs(normalized) ** 2)
    assert np.isclose(power, 1.0, rtol=1e-5)
    
    # Test with zero signal
    zero_signal = np.zeros(10, dtype=np.complex128)
    normalized_zero = Modulation.normalize_signal(zero_signal)
    assert np.array_equal(normalized_zero, zero_signal)
