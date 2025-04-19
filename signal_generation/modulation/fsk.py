"""Frequency Shift Keying modulation implementation."""

import numpy as np
from .base import Modulation


class FSKModulation(Modulation):
    """Frequency Shift Keying modulation."""

    def __init__(self, params):
        """
        Initialize FSK modulation.

        Args:
            params: Signal generation parameters
        """
        super().__init__(params)
        self.num_frequencies = self.select_number_of_states()
        spacing = params.bandwidth / (self.num_frequencies - 1)
        # Generate frequencies around zero
        self.frequencies = (
            np.arange(-(self.num_frequencies - 1) / 2, (self.num_frequencies + 1) / 2)
            * spacing
        )

    def generate_symbol(self, start_idx: int, end_idx: int) -> np.ndarray:
        """
        Generate a symbol by selecting a random frequency.

        Args:
            start_idx: Start index of the symbol
            end_idx: End index of the symbol

        Returns:
            Complex signal for the symbol
        """
        frequency = np.random.choice(self.frequencies)
        phase = self.params.two_pi_time_signal[start_idx:end_idx] * frequency
        return np.exp(1j * phase)
