"""Phase Shift Keying modulation implementation."""

import numpy as np
from .base import Modulation


class PSKModulation(Modulation):
    """Phase Shift Keying modulation."""

    def __init__(self, params):
        """
        Initialize PSK modulation.

        Args:
            params: Signal generation parameters
        """
        super().__init__(params)
        self.num_phase_states = self.select_number_of_states()
        self.phase_states = np.linspace(
            0, 2 * np.pi, self.num_phase_states, endpoint=False
        )

    def generate_symbol(self, start_idx: int, end_idx: int) -> np.ndarray:
        """
        Generate a symbol by selecting a random phase.

        Args:
            start_idx: Start index of the symbol
            end_idx: End index of the symbol

        Returns:
            Complex signal for the symbol
        """
        phase = np.random.choice(self.phase_states)
        return np.exp(1j * phase)
