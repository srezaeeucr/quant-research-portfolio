"""Stochastic Oscillator Crossover — different momentum oscillator than RSI.

Logic
-----
%K = ((close - low_N) / (high_N - low_N)) × 100  over period N (default 14)
%D = SMA(3) of %K

Buy: %K crosses ABOVE %D AND below the oversold threshold (20).
Different smoothing/calculation than RSI; often signals at different times.
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


class StochasticCrossoverStrategy(BaseStrategy):
    """%K crosses %D from below in oversold zone."""

    def __init__(self, config: Dict):
        super().__init__(config)
        params = config.get("strategy_params", {}).get("stochastic_crossover", {})
        strategy_cfg = config.get("strategy", {})
        self.k_period       = int(params.get("k_period", 14))
        self.d_period       = int(params.get("d_period", 3))
        self.oversold_max   = float(params.get("oversold_max", 20.0))
        self.volume_mult    = float(strategy_cfg.get("volume_multiplier", 1.0))
        self.use_regime     = bool(strategy_cfg.get("use_regime_filter", False))

        self._signal_fired = False
        self._prev_k = None
        self._prev_d = None
        self.current_regime_sma: Optional[float] = None

    @property
    def name(self) -> str:
        return "Stochastic"

    def reset_session(self) -> None:
        self._signal_fired = False
        self._prev_k = None
        self._prev_d = None

    def set_regime_sma(self, sma: Optional[float]) -> None:
        self.current_regime_sma = sma

    def generate_signal(self, df: pd.DataFrame, current_index: int) -> Optional[Dict]:
        if self._signal_fired:
            return None
        if current_index < self.k_period + self.d_period:
            return None
        bar = df.iloc[current_index]
        bar_time = bar["timestamp"].astimezone(_ET).time()
        if not (_MARKET_OPEN <= bar_time <= _MARKET_CLOSE):
            return None

        # Compute %K and %D
        sl = df.iloc[current_index - self.k_period + 1 : current_index + 1]
        low_n  = float(sl["low"].astype(float).min())
        high_n = float(sl["high"].astype(float).max())
        if high_n == low_n:
            return None
        close = float(bar["close"])
        k_val = (close - low_n) / (high_n - low_n) * 100

        # Compute %D as SMA of last 3 %K values
        # We need the previous 2 %K values too
        recent_closes = df.iloc[current_index - self.d_period + 1 : current_index + 1]
        k_series = []
        for j in range(self.d_period):
            idx = current_index - (self.d_period - 1 - j)
            if idx < self.k_period - 1:
                return None
            window = df.iloc[idx - self.k_period + 1 : idx + 1]
            wlow  = float(window["low"].astype(float).min())
            whigh = float(window["high"].astype(float).max())
            if whigh == wlow: return None
            k_series.append((float(df.iloc[idx]["close"]) - wlow) / (whigh - wlow) * 100)
        d_val = sum(k_series) / len(k_series)

        if self._prev_k is None or self._prev_d is None:
            self._prev_k, self._prev_d = k_val, d_val
            return None

        crossed_up = (self._prev_k <= self._prev_d) and (k_val > d_val)
        in_oversold = self._prev_k < self.oversold_max  # was in oversold before crossing
        self._prev_k, self._prev_d = k_val, d_val

        if not (crossed_up and in_oversold):
            return None

        # Volume confirmation
        N = 20
        if current_index >= N:
            avg_vol = df.iloc[current_index - N : current_index]["volume"].astype(float).mean()
            if float(bar["volume"]) < avg_vol * self.volume_mult:
                return None

        if self.use_regime and self.current_regime_sma is not None:
            if close < self.current_regime_sma:
                return None

        self._signal_fired = True
        return {
            "signal": "BUY", "symbol": None,
            "entry_price": close, "k": float(k_val), "d": float(d_val),
            "bar_time": bar["timestamp"],
            "reason": f"Stoch %K crossed %D from oversold ({k_val:.1f},{d_val:.1f})",
            "volume_confirmed": True,
        }
