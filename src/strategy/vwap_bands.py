"""VWAP Bands Strategy — volume-weighted breakout.

Logic
-----
1. Compute intraday VWAP from market open
2. Compute rolling standard deviation of (price − VWAP)
3. Trigger BUY when:
     - close breaks above VWAP + 1σ band
     - on rising volume (current bar > avg of last N)
     - and price is in upper half of bar (close > midpoint)

Different from VWAP Reversion (mean-revert to VWAP) because it bets on
continuation breakouts WITH volume confirmation, not reversion to VWAP.
"""
import logging
from datetime import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pytz

from src.strategy.base import BaseStrategy

logger = logging.getLogger(__name__)
_ET = pytz.timezone("America/New_York")

_MARKET_OPEN  = time(9, 30)
_MARKET_CLOSE = time(15, 50)


class VWAPBandsStrategy(BaseStrategy):
    """VWAP + 1σ band breakout with volume confirmation."""

    def __init__(self, config: Dict):
        super().__init__(config)
        params = config.get("strategy_params", {}).get("vwap_bands", {})
        strategy_cfg = config.get("strategy", {})

        self.band_sigma         = float(params.get("band_sigma", 1.0))
        self.min_warmup_bars    = int(params.get("min_warmup_bars", 15))   # need ≥15 bars for σ
        self.volume_multiplier  = float(strategy_cfg.get("volume_multiplier", 1.5))
        self.use_regime_filter  = bool(strategy_cfg.get("use_regime_filter", False))
        self.regime_sma_days    = int(strategy_cfg.get("regime_sma_days", 20))
        self.afternoon_entries  = bool(config.get("afternoon_entries", True))

        self._signal_fired = False
        self.current_regime_sma: Optional[float] = None

    @property
    def name(self) -> str:
        return "VWAPBands"

    def reset_session(self) -> None:
        self._signal_fired = False

    def set_regime_sma(self, sma: Optional[float]) -> None:
        self.current_regime_sma = sma

    def generate_signal(self, df: pd.DataFrame, current_index: int) -> Optional[Dict]:
        if self._signal_fired:
            return None
        if current_index < self.min_warmup_bars:
            return None

        bar = df.iloc[current_index]
        bar_time = bar["timestamp"].astimezone(_ET).time()
        if not (_MARKET_OPEN <= bar_time <= _MARKET_CLOSE):
            return None

        # Use bars from market open up to and including current
        day_bars = df.iloc[: current_index + 1]
        # VWAP: cumulative price*volume / cumulative volume
        typical = (day_bars["high"].astype(float)
                   + day_bars["low"].astype(float)
                   + day_bars["close"].astype(float)) / 3.0
        vol = day_bars["volume"].astype(float)
        cum_pv = (typical * vol).cumsum()
        cum_v  = vol.cumsum().replace(0, 1e-10)
        vwap   = cum_pv / cum_v

        # σ of (close − vwap)
        diff = day_bars["close"].astype(float) - vwap
        sigma = diff.std()
        if pd.isna(sigma) or sigma == 0:
            return None

        cur_vwap  = float(vwap.iloc[-1])
        upper_band = cur_vwap + self.band_sigma * sigma
        close = float(bar["close"])

        # Breakout check
        if close <= upper_band:
            return None

        # Bar in upper half (close > midpoint)
        midpoint = (float(bar["high"]) + float(bar["low"])) / 2.0
        if close <= midpoint:
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
            if close < self.current_regime_sma:
                return None

        self._signal_fired = True
        return {
            "signal":      "BUY",
            "symbol":      None,
            "entry_price": close,
            "vwap":        cur_vwap,
            "upper_band":  upper_band,
            "sigma":       float(sigma),
            "bar_time":    bar["timestamp"],
            "reason":      f"Break above VWAP+{self.band_sigma}σ ({close:.2f}>{upper_band:.2f})",
            "volume_confirmed": True,
        }
