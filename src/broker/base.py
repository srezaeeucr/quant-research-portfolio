from abc import ABC, abstractmethod
from typing import Dict


class BaseBroker(ABC):
    """Abstract broker interface."""

    @abstractmethod
    def submit_order(self, position: Dict) -> Dict:
        """Place an entry order. Returns an order record."""

    @abstractmethod
    def close_position(self, position: Dict, exit_price: float, reason: str) -> Dict:
        """Close an open position. Returns a full trade record."""

    @abstractmethod
    def get_account_balance(self) -> float:
        """Return current cash / account equity."""
