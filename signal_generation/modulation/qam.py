"""Quadrature Amplitude Modulation implementation."""

import random
import numpy as np
from .base import Modulation


class QAMModulation(Modulation):
    """Quadrature Amplitude Modulation."""

    def __init__(self, params):
        """
        Initialize QAM modulation.

        Args:
            params: Signal generation parameters
        """
        super().__init__(params)
        self.num_states = self.select_number_of_states()
        self.constellation = self._create_constellation()

    @staticmethod
    def select_number_of_states() -> int:
        """
        Select the number of states for QAM modulation.
        QAM constellations must be perfect squares (4, 16, 64, 256, etc.)

        Returns:
            int: Number of QAM states (4, 16, or 64)
        """
        return random.choice([4, 16, 64])  # Common QAM constellations

    def _create_constellation(self) -> np.ndarray:
        """
        Create a square constellation of points.

        Returns:
            Array of complex constellation points
        """
        side_length = int(np.sqrt(self.num_states))
        constellation = np.zeros(self.num_states, dtype=np.complex128)

        for i in range(side_length):
            for j in range(side_length):
                x = (2 * i - (side_length - 1)) / (side_length - 1)
                y = (2 * j - (side_length - 1)) / (side_length - 1)
                constellation[i * side_length + j] = x + 1j * y

        return self.normalize_signal(constellation)

    def generate_symbol(self, start_idx: int, end_idx: int) -> np.ndarray:
        """
        Generate a symbol by selecting a random constellation point.

        Args:
            start_idx: Start index of the symbol
            end_idx: End index of the symbol

        Returns:
            Complex signal for the symbol
        """
        return np.random.choice(self.constellation)
