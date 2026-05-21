"""Bollinger Band Reversal — mean reversion via price MA ± Nσ.

Logic
-----
1. Compute rolling 20-bar SMA + 2σ band
2. Wait for price to touch / dip below LOWER band
3. Buy when price closes back ABOVE lower band AND current bar is bullish

Different from VWAP Reversion (volume-weighted MA) and RSI Reversion (oscillator).
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


class BollingerReversalStrategy(BaseStrategy):
    """Touch lower BB, recover, bullish bar → BUY."""

    def __init__(self, config: Dict):
        super().__init__(config)
        params = config.get("strategy_params", {}).get("bollinger_reversal", {})
        strategy_cfg = config.get("strategy", {})
        self.period      = int(params.get("period", 20))
        self.num_std     = float(params.get("num_std", 2.0))
        self.volume_mult = float(strategy_cfg.get("volume_multiplier", 1.0))
        self.use_regime  = bool(strategy_cfg.get("use_regime_filter", False))

        self._signal_fired = False
        self._was_below_lower = False
        self.current_regime_sma: Optional[float] = None

    @property
    def name(self) -> str:
        return "Bollinger"

    def reset_session(self) -> None:
        self._signal_fired = False
        self._was_below_lower = False

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

        closes = df.iloc[: current_index + 1]["close"].astype(float)
        sma   = closes.rolling(self.period).mean().iloc[-1]
        std   = closes.rolling(self.period).std().iloc[-1]
        if pd.isna(sma) or pd.isna(std):
            return None
        lower = sma - self.num_std * std

        close = float(bar["close"])
        low   = float(bar["low"])
        opn   = float(bar["open"])

        # Track if we touched/dipped below lower band
        if low <= lower:
            self._was_below_lower = True
            return None

        if not self._was_below_lower:
            return None
        # Recovery + bullish bar
        if not (close > lower and close > opn):
            return None

        # Volume confirmation
        N = 20
        if current_index >= N:
            avg_vol = df.iloc[current_index - N: current_index]["volume"].astype(float).mean()
            if float(bar["volume"]) < avg_vol * self.volume_mult:
                return None

        if self.use_regime and self.current_regime_sma is not None:
            if close < self.current_regime_sma:
                return None

        self._signal_fired = True
        self._was_below_lower = False
        return {
            "signal": "BUY", "symbol": None,
            "entry_price": close, "lower_band": float(lower), "sma": float(sma),
            "bar_time": bar["timestamp"],
            "reason": f"BB recovery (low={low:.2f}<lower={lower:.2f}, close={close:.2f})",
            "volume_confirmed": True,
        }
