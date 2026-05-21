"""MACD Crossover Strategy — trend-following.

MACD line   = EMA(12) − EMA(26)
Signal line = EMA(9) of MACD
Entry       : MACD crosses ABOVE signal line AND MACD > 0 AND volume confirmation

Different from EMA Crossover because:
  - Uses three EMAs (12/26/9), not two (9/21)
  - Crossover is between MACD and its own smoothed version, not between price EMAs
  - Empirically captures different turning points
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


class MACDCrossoverStrategy(BaseStrategy):
    """MACD signal-line crossover with volume confirmation."""

    def __init__(self, config: Dict):
        super().__init__(config)
        params = config.get("strategy_params", {}).get("macd_crossover", {})
        strategy_cfg = config.get("strategy", {})
        # Textbook parameters
        self.fast_period   = int(params.get("fast_period",  12))
        self.slow_period   = int(params.get("slow_period",  26))
        self.signal_period = int(params.get("signal_period", 9))
        self.volume_multiplier = float(strategy_cfg.get("volume_multiplier", 1.2))
        self.use_regime_filter = bool(strategy_cfg.get("use_regime_filter", False))
        self.regime_sma_days   = int(strategy_cfg.get("regime_sma_days", 20))
        self.afternoon_entries = bool(config.get("afternoon_entries", True))
        self.entry_window_minutes = int(params.get("entry_window_minutes",
                                                    config.get("entry_window_minutes", 360)))

        self._signal_fired = False
        self.current_regime_sma: Optional[float] = None
        self._prev_macd: Optional[float] = None
        self._prev_signal: Optional[float] = None

    @property
    def name(self) -> str:
        return "MACD"

    def reset_session(self) -> None:
        self._signal_fired = False
        self._prev_macd = None
        self._prev_signal = None

    def set_regime_sma(self, sma: Optional[float]) -> None:
        self.current_regime_sma = sma

    def generate_signal(self, df: pd.DataFrame, current_index: int) -> Optional[Dict]:
        if self._signal_fired:
            return None
        if current_index < self.slow_period + self.signal_period:
            return None
        bar = df.iloc[current_index]
        bar_time = bar["timestamp"].astimezone(_ET).time()
        if not (_MARKET_OPEN <= bar_time <= _MARKET_CLOSE):
            return None

        # Bars up to and including current — use only past data
        closes = df.iloc[: current_index + 1]["close"].astype(float)
        # MACD calculation
        ema_fast = closes.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = closes.ewm(span=self.slow_period, adjust=False).mean()
        macd     = ema_fast - ema_slow
        signal   = macd.ewm(span=self.signal_period, adjust=False).mean()

        cur_macd, cur_sig = float(macd.iloc[-1]), float(signal.iloc[-1])
        if self._prev_macd is None or self._prev_signal is None:
            self._prev_macd, self._prev_signal = cur_macd, cur_sig
            return None

        crossed_up = (self._prev_macd <= self._prev_signal) and (cur_macd > cur_sig)
        self._prev_macd, self._prev_signal = cur_macd, cur_sig

        if not crossed_up:
            return None
        if cur_macd <= 0:  # require above zero line — only trade in bullish regime
            return None

        # Volume confirmation: current bar volume > avg of last N bars × multiplier
        N = 20
        if current_index >= N:
            recent_vols = df.iloc[current_index - N : current_index]["volume"].astype(float)
            avg_vol = recent_vols.mean()
            if float(bar["volume"]) < avg_vol * self.volume_multiplier:
                return None

        # Regime filter
        close = float(bar["close"])
        if self.use_regime_filter and self.current_regime_sma is not None:
            if close < self.current_regime_sma:
                return None

        self._signal_fired = True
        return {
            "signal":      "BUY",
            "symbol":      None,
            "entry_price": close,
            "macd":        cur_macd,
            "macd_signal": cur_sig,
            "bar_time":    bar["timestamp"],
            "reason":      f"MACD crossed signal upward ({cur_macd:.4f}>{cur_sig:.4f})",
            "volume_confirmed": True,
        }
