from abc import ABC, abstractmethod
from typing import Dict, Optional

import pandas as pd


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, config: Dict):
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, current_index: int) -> Optional[Dict]:
        """Evaluate bars up to current_index and return a signal dict or None.

        Parameters
        ----------
        df            : Full day's DataFrame (timestamp, open, high, low, close, volume)
        current_index : Index of the bar being evaluated (bars 0..current_index are visible)

        Returns
        -------
        Signal dict on a new entry trigger, None otherwise.
        """
