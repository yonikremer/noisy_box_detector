"""
Signal Generation Package

This package provides functionality to generate various types of modulated signals
(FSK, PSK, QAM, ASK) and visualize them in both time and frequency domains.
"""

__version__ = "0.1.0"

from .utils.parameters import SignalParameters
from .utils.config import load_config
from .utils.visualization import plot_signal
from .modulation.fsk import FSKModulation
from .modulation.psk import PSKModulation
from .modulation.qam import QAMModulation
from .modulation.ask import ASKModulation

__all__ = [
    "SignalParameters",
    "load_config",
    "plot_signal",
    "FSKModulation",
    "PSKModulation",
    "QAMModulation",
    "ASKModulation",
]
