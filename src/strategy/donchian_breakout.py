"""Donchian Channel Breakout — N-bar rolling high breakout.

Logic
-----
1. Rolling N-bar (default 20) highest high (excluding current bar)
2. Buy when current close breaks above that high
3. Volume confirmation

Different from ORB (fixed opening range) — fires anytime during the day,
catches breakouts ORB misses (after 10:00 ET cutoff).
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


class DonchianBreakoutStrategy(BaseStrategy):
    """N-bar high breakout."""

    def __init__(self, config: Dict):
        super().__init__(config)
        params = config.get("strategy_params", {}).get("donchian_breakout", {})
        strategy_cfg = config.get("strategy", {})
        self.period      = int(params.get("period", 20))
        self.volume_mult = float(strategy_cfg.get("volume_multiplier", 1.5))
        self.use_regime  = bool(strategy_cfg.get("use_regime_filter", False))

        self._signal_fired = False
        self.current_regime_sma: Optional[float] = None

    @property
    def name(self) -> str:
        return "Donchian"

    def reset_session(self) -> None:
        self._signal_fired = False

    def set_regime_sma(self, sma: Optional[float]) -> None:
        self.current_regime_sma = sma

    def generate_signal(self, df: pd.DataFrame, current_index: int) -> Optional[Dict]:
        if self._signal_fired:
            return None
        if current_index < self.period:
            return None
        bar = df.iloc[current_index]
        bar_time = bar["timestamp"].astimezone(_ET).time()
        if not (_MARKET_OPEN <= bar_time <= _MARKET_CLOSE):
            return None

        # Highest high of prior N bars (excluding current)
        highest = df.iloc[current_index - self.period: current_index]["high"].astype(float).max()
        close = float(bar["close"])
        if close <= highest:
            return None

        # Volume confirmation
        avg_vol = df.iloc[current_index - self.period: current_index]["volume"].astype(float).mean()
        if float(bar["volume"]) < avg_vol * self.volume_mult:
            return None

        if self.use_regime and self.current_regime_sma is not None:
            if close < self.current_regime_sma:
                return None

        self._signal_fired = True
        return {
            "signal": "BUY", "symbol": None,
            "entry_price": close, "donchian_high": float(highest),
            "bar_time": bar["timestamp"],
            "reason": f"Donchian breakout above {self.period}-bar high ({highest:.2f}→{close:.2f})",
            "volume_confirmed": True,
        }
