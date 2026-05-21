"""Inside Bar Breakout — pure price-action pattern.

Logic
-----
Inside bar = current bar's high < prior high AND current low > prior low (compression).
Pattern setup: bar_t-1 is "mother bar", bar_t is inside bar.
Trigger: at bar_t+1, BUY if close > mother bar's high (breakout).

Pure pattern — no oscillators, no MAs. Different family from everything else.
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


class InsideBarBreakoutStrategy(BaseStrategy):
    """Inside-bar setup, breakout entry."""

    def __init__(self, config: Dict):
        super().__init__(config)
        params = config.get("strategy_params", {}).get("inside_bar_breakout", {})
        strategy_cfg = config.get("strategy", {})
        self.volume_mult = float(strategy_cfg.get("volume_multiplier", 1.2))
        self.use_regime  = bool(strategy_cfg.get("use_regime_filter", False))

        self._signal_fired = False
        self._setup_mother_high: Optional[float] = None
        self._setup_mother_low:  Optional[float] = None
        self.current_regime_sma: Optional[float] = None

    @property
    def name(self) -> str:
        return "InsideBar"

    def reset_session(self) -> None:
        self._signal_fired = False
        self._setup_mother_high = None
        self._setup_mother_low  = None

    def set_regime_sma(self, sma: Optional[float]) -> None:
        self.current_regime_sma = sma

    def generate_signal(self, df: pd.DataFrame, current_index: int) -> Optional[Dict]:
        if self._signal_fired:
            return None
        if current_index < 2:
            return None
        bar = df.iloc[current_index]
        bar_time = bar["timestamp"].astimezone(_ET).time()
        if not (_MARKET_OPEN <= bar_time <= _MARKET_CLOSE):
            return None

        prev = df.iloc[current_index - 1]
        prev_prev = df.iloc[current_index - 2]

        # First check: did we just have an inside bar (prev relative to prev_prev)?
        prev_high   = float(prev["high"])
        prev_low    = float(prev["low"])
        mother_high = float(prev_prev["high"])
        mother_low  = float(prev_prev["low"])
        is_inside_setup = (prev_high < mother_high) and (prev_low > mother_low)

        if is_inside_setup and self._setup_mother_high is None:
            # Arm: store mother bar's range
            self._setup_mother_high = mother_high
            self._setup_mother_low  = mother_low

        # Trigger: current bar breaks above stored mother high
        if self._setup_mother_high is None:
            return None
        close = float(bar["close"])
        if close <= self._setup_mother_high:
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

        # Fire and disarm
        mother_high = self._setup_mother_high
        self._signal_fired = True
        self._setup_mother_high = None
        self._setup_mother_low  = None
        return {
            "signal": "BUY", "symbol": None,
            "entry_price": close, "mother_high": float(mother_high),
            "bar_time": bar["timestamp"],
            "reason": f"Inside-bar breakout above mother high ({mother_high:.2f}→{close:.2f})",
            "volume_confirmed": True,
        }
