"""RSI Mean Reversion Strategy.

Bet on oversold-to-recovery reversals.

Logic
-----
1. Compute 14-bar RSI on minute closes
2. Wait for RSI < oversold_threshold (default 30)
3. Trigger BUY when RSI crosses BACK ABOVE oversold_threshold AND
   the current bar is bullish (close > open)

Different from existing strategies because it bets AGAINST the recent move
(mean reversion) rather than WITH it (trend/breakout).
"""
import logging
from datetime import time
from typing import Dict, Optional

import pandas as pd
import pytz

from src.strategy.base import BaseStrategy

logger = logging.getLogger(__name__)
_ET = pytz.timezone("America/New_York")

_MARKET_OPEN  = time(9, 30)
_MARKET_CLOSE = time(15, 50)


def _rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """Standard RSI implementation (Wilder's smoothing)."""
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


class RSIReversionStrategy(BaseStrategy):
    """RSI oversold-recovery reversal."""

    def __init__(self, config: Dict):
        super().__init__(config)
        params = config.get("strategy_params", {}).get("rsi_reversion", {})
        strategy_cfg = config.get("strategy", {})

        self.rsi_period         = int(params.get("rsi_period", 14))
        self.oversold_threshold = float(params.get("oversold_threshold", 30.0))
        self.volume_multiplier  = float(strategy_cfg.get("volume_multiplier", 1.0))
        self.use_regime_filter  = bool(strategy_cfg.get("use_regime_filter", False))
        self.regime_sma_days    = int(strategy_cfg.get("regime_sma_days", 20))

        self._signal_fired = False
        self._was_oversold = False
        self.current_regime_sma: Optional[float] = None

    @property
    def name(self) -> str:
        return "RSI"

    def reset_session(self) -> None:
        self._signal_fired = False
        self._was_oversold = False

    def set_regime_sma(self, sma: Optional[float]) -> None:
        self.current_regime_sma = sma

    def generate_signal(self, df: pd.DataFrame, current_index: int) -> Optional[Dict]:
        if self._signal_fired:
            return None
        if current_index < self.rsi_period + 1:
            return None
        bar = df.iloc[current_index]
        bar_time = bar["timestamp"].astimezone(_ET).time()
        if not (_MARKET_OPEN <= bar_time <= _MARKET_CLOSE):
            return None

        closes = df.iloc[: current_index + 1]["close"].astype(float)
        rsi_series = _rsi(closes, self.rsi_period)
        cur_rsi = float(rsi_series.iloc[-1])
        prev_rsi = float(rsi_series.iloc[-2])

        # Update oversold state
        if cur_rsi < self.oversold_threshold:
            self._was_oversold = True
            return None

        # Trigger: RSI crossed back above oversold from below
        if not self._was_oversold:
            return None
        if not (prev_rsi < self.oversold_threshold and cur_rsi >= self.oversold_threshold):
            return None

        # Bullish bar confirmation
        if float(bar["close"]) <= float(bar["open"]):
            return None

        # Volume confirmation
        N = 20
        if current_index >= N:
            recent_vols = df.iloc[current_index - N : current_index]["volume"].astype(float)
            avg_vol = recent_vols.mean()
            if float(bar["volume"]) < avg_vol * self.volume_multiplier:
                return None

        # Regime filter (optional)
        if self.use_regime_filter and self.current_regime_sma is not None:
            if float(bar["close"]) < self.current_regime_sma:
                return None

        self._signal_fired = True
        self._was_oversold = False
        return {
            "signal":      "BUY",
            "symbol":      None,
            "entry_price": float(bar["close"]),
            "rsi":         cur_rsi,
            "prev_rsi":    prev_rsi,
            "bar_time":    bar["timestamp"],
            "reason":      f"RSI recovered from oversold ({prev_rsi:.1f}→{cur_rsi:.1f})",
            "volume_confirmed": True,
        }
