import os
import sys
import warnings
import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from spectogram import create_spectrogram, plot_spectrogram, WINDOW_SIZE, HOP_SIZE, WINDOW_TYPE, FFT_MODE


def test_create_spectrogram_invalid_input():
    """Test that create_spectrogram raises ValueError for non-complex input."""
    # Test with real-valued array
    real_signal = np.ones(1000)
    with pytest.raises(ValueError, match="Input snapshot must be a numpy array of complex values"):
        create_spectrogram(real_signal, 1000)
    
    # Test with non-numpy input
    with pytest.raises(TypeError, match="Input snapshot must be a numpy array."):
        create_spectrogram([1+1j, 2+2j], 1000)


@patch('spectogram.ShortTimeFFT')
@patch('spectogram.plot_spectrogram')
def test_create_spectrogram_valid_input(mock_plot, mock_stft, monkeypatch):
    """Test that create_spectrogram works correctly with valid input."""
    # Create a mock complex signal
    complex_signal = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128)
    
    # Mock the ShortTimeFFT instance
    mock_stft_instance = MagicMock()
    mock_stft.return_value = mock_stft_instance
    
    # Mock the spectrogram method
    mock_spectrogram = np.array([[1, 2], [3, 4]])
    mock_stft_instance.spectrogram.return_value = mock_spectrogram
    
    # Mock the frequency and time axes
    mock_stft_instance.f = np.array([0, 1])
    mock_stft_instance.t.return_value = np.array([0, 1])
    
    # Call the function
    create_spectrogram(complex_signal, 1000)
    
    # Verify that ShortTimeFFT was called with correct parameters
    mock_stft.assert_called_once()
    call_args = mock_stft.call_args[1]
    assert call_args['fs'] == 1000
    assert call_args['fft_mode'] == FFT_MODE
    assert call_args['hop'] == HOP_SIZE
    
    # Verify that spectrogram was called with the signal
    mock_stft_instance.spectrogram.assert_called_once_with(complex_signal)
    
    # Verify that plot_spectrogram was called with correct parameters
    # We need to check each argument separately due to NumPy array comparison
    mock_plot.assert_called_once()
    args = mock_plot.call_args[0]
    assert len(args) == 3
    assert np.array_equal(args[0], np.array([0, 1]))  # frequencies
    assert np.array_equal(args[1], mock_spectrogram)  # spectrogram
    assert np.array_equal(args[2], np.array([0, 1]))  # times


@patch('spectogram.plt.figure')
@patch('spectogram.plt.imshow')
@patch('spectogram.plt.colorbar')
@patch('spectogram.plt.xlabel')
@patch('spectogram.plt.ylabel')
@patch('spectogram.plt.title')
@patch('spectogram.plt.tight_layout')
@patch('spectogram.plt.show')
def test_plot_spectrogram(mock_show, mock_tight_layout, mock_title, mock_ylabel, 
                         mock_xlabel, mock_colorbar, mock_imshow, mock_figure):
    """Test that plot_spectrogram correctly sets up the plot."""
    # Create test data
    frequencies = np.array([0, 100, 200])
    spectrogram = np.array([[1, 2], [3, 4], [5, 6]])
    times = np.array([0, 1])
    
    # Call the function
    plot_spectrogram(frequencies, spectrogram, times)
    
    # Verify that figure was created with correct size
    mock_figure.assert_called_once_with(figsize=(10, 6))
    
    # Verify that imshow was called with correct parameters
    mock_imshow.assert_called_once()
    call_args = mock_imshow.call_args[1]
    assert call_args['aspect'] == 'auto'
    assert call_args['origin'] == 'lower'
    assert call_args['cmap'] == 'Grays'
    
    # Check extent separately due to NumPy array comparison
    assert np.array_equal(call_args['extent'], np.array([0, 1, 0, 200]))
    
    # Verify that colorbar was added
    mock_colorbar.assert_called_once()
    
    # Verify that labels were set
    mock_xlabel.assert_called_once_with('Time (s)')
    mock_ylabel.assert_called_once_with('Frequency (Hz)')
    mock_title.assert_called_once_with('Spectrogram')
    
    # Verify that tight_layout and show were called
    mock_tight_layout.assert_called_once()
    mock_show.assert_called_once()


def test_plot_spectrogram_log_scale():
    """Test that plot_spectrogram correctly applies log scale to spectrogram."""
    # Create test data with zeros to test log scaling
    frequencies = np.array([0, 1])
    spectrogram = np.array([[0, 1], [2, 3]])
    times = np.array([0, 1])
    
    # Mock plt.imshow to capture the data passed to it
    with patch('spectogram.plt.figure') as mock_figure, \
         patch('spectogram.plt.imshow') as mock_imshow, \
         patch('spectogram.plt.colorbar') as mock_colorbar, \
         patch('spectogram.plt.xlabel') as mock_xlabel, \
         patch('spectogram.plt.ylabel') as mock_ylabel, \
         patch('spectogram.plt.title') as mock_title, \
         patch('spectogram.plt.tight_layout') as mock_tight_layout, \
         patch('spectogram.plt.show') as mock_show:
        
        # Set up the mock imshow to return a mock object that can be used by colorbar
        mock_imshow.return_value = MagicMock()
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            plot_spectrogram(frequencies, spectrogram, times)
        
        # Get the data passed to imshow
        data_passed = mock_imshow.call_args[0][0]
        
        # Check that log10 was applied
        # catch division by zero warning
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            expected = np.log10(spectrogram)
        # Handle -inf values from log10(0)
        expected[expected == -np.inf] = data_passed[data_passed == -np.inf]
        assert np.array_equal(data_passed, expected)
        
        # Verify that colorbar was called with the correct label
        mock_colorbar.assert_called_once_with(label="Power (dB)")


def test_spectrogram_pixel_ratio():
    """Test that the ratio between spectrogram pixels and signal length is correct."""
    # Create a test signal with known length
    sample_rate = 1000000  # 1 MHz
    duration_ms = 100  # 100 ms
    signal_length = int(sample_rate * duration_ms / 1000)  # 100,000 samples
    
    # Create a simple test signal (complex sinusoid)
    t = np.linspace(0, duration_ms/1000, signal_length)
    freq = 10000  # 10 kHz
    complex_signal = np.exp(2j * np.pi * freq * t)
    
    # Only mock the plotting function to avoid displaying the plot during tests
    with patch('spectogram.plot_spectrogram') as mock_plot:
        create_spectrogram(complex_signal, sample_rate)
        
        # Get the actual spectrogram data from the mock call
        mock_call_args = mock_plot.call_args[0]
        frequencies, spectrogram, times = mock_call_args
        
        # Get the actual dimensions from the spectrogram
        num_freq_bins, num_time_bins = spectrogram.shape
        
        # Calculate expected time bins
        # The formula accounts for the window size and hop size:
        # - We need enough hops to cover the entire signal
        # - Each hop moves by HOP_SIZE samples
        # - We need to include both the start and end points
        # - We need to account for the initial window size
        expected_time_bins = int(np.ceil((signal_length + WINDOW_SIZE - HOP_SIZE) / HOP_SIZE))
        assert num_time_bins == expected_time_bins, \
            f"Expected {expected_time_bins} time bins, got {num_time_bins}"
        
        # Verify that the time axis covers the signal duration
        assert len(times) == num_time_bins, \
            f"Expected {num_time_bins} time points, got {len(times)}"
        # The time axis may extend slightly beyond the signal duration due to the window size
        max_expected_duration = (duration_ms/1000) * (1 + WINDOW_SIZE/signal_length)
        assert times[-1] <= max_expected_duration, \
            f"Time axis exceeds maximum expected duration of {max_expected_duration}s"
        
        # Verify frequency axis
        nyquist = sample_rate / 2
        assert frequencies[0] >= -nyquist and frequencies[-1] <= nyquist, \
            f"Frequencies should be within ±{nyquist} Hz"
        
        # Verify that the spectrogram captures the test signal frequency
        # Find the frequency bin closest to our test frequency
        freq_bin = np.argmin(np.abs(frequencies - freq))
        # Find the maximum power in the spectrogram
        max_freq_bin = np.argmax(np.mean(np.abs(spectrogram), axis=1))
        assert abs(max_freq_bin - freq_bin) <= 1, \
            f"Expected peak frequency around bin {freq_bin}, got {max_freq_bin}" 