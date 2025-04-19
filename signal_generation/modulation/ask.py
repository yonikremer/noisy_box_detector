"""Amplitude Shift Keying modulation implementation."""

import numpy as np
from .base import Modulation


class ASKModulation(Modulation):
    """Amplitude Shift Keying modulation."""

    def __init__(self, params):
        """
        Initialize ASK modulation.

        Args:
            params: Signal generation parameters
        """
        super().__init__(params)
        self.num_levels = self.select_number_of_states()
        # Generate amplitude levels and normalize to have unit average power
        self.amplitude_levels = np.linspace(0.2, 1.0, self.num_levels)
        self.amplitude_levels = self.normalize_signal(self.amplitude_levels)

    def generate_symbol(self, start_idx: int, end_idx: int) -> np.ndarray:
        """
        Generate a symbol by selecting a random amplitude level.

        Args:
            start_idx: Start index of the symbol
            end_idx: End index of the symbol

        Returns:
            Complex signal for the symbol
        """
        return np.random.choice(self.amplitude_levels)
