"""
Momentum Strategy — trade strong opening momentum confirmed by volume.
"""
import logging
from datetime import time
from typing import Dict, List, Optional

import pandas as pd
import pytz

from src.strategy.base import BaseStrategy

logger = logging.getLogger(__name__)

_ET = pytz.timezone("America/New_York")

_OR_START  = time(9, 30)
_OR_END    = time(9, 44)
_ENTRY_BAR = time(9, 45)

# Afternoon (PM) opening range — only used when afternoon_entries=True
_PM_OR_START  = time(15, 30)
_PM_OR_END    = time(15, 44)
_PM_ENTRY_BAR = time(15, 45)


class MomentumStrategy(BaseStrategy):
    """Opening 15-Minute Momentum.

    Logic
    -----
    1. Track price during the opening range (9:30–9:44 ET, 15 bars).
    2. At the 9:45 bar: compute
           momentum_pct = (or_close − session_open) / session_open × 100
    3. Signal fires if momentum_pct >= momentum_threshold AND
       total OR volume > or_volume_threshold (proxy for SPY).
    4. One signal per session.

    Config (reads from strategy_params.momentum, falls back to strategy):
        momentum_threshold: 0.3   minimum % move in the opening 15 min
        or_volume_threshold: 500000  total OR volume proxy (e.g. SPY typical level)
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        params    = config.get("strategy_params", {}).get("momentum", {})
        strat_cfg = config.get("strategy", {})  # backward compat
        self.momentum_threshold: float = float(params.get(
            "momentum_threshold",
            strat_cfg.get("momentum_threshold_pct", 0.3),
        ))
        self.or_volume_threshold: float = float(params.get("or_volume_threshold", 500_000))
        self.volume_multiplier: float = float(params.get(
            "volume_multiplier",
            strat_cfg.get("volume_multiplier", 1.0),
        ))

        self.afternoon_entries: bool = bool(config.get("afternoon_entries", False))

        self._signal_fired:    bool  = False
        self._open_price:      Optional[float] = None
        self._or_close:        Optional[float] = None
        self._or_volumes:      List[float] = []
        self._prev_or_volume:  Optional[float] = None

        # Afternoon state
        self._pm_signal_fired: bool = False
        self._pm_open_price:   Optional[float] = None
        self._pm_or_close:     Optional[float] = None
        self._pm_or_volumes:   List[float] = []

    @property
    def name(self) -> str:
        return "Momentum"

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def reset_session(self) -> None:
        self._signal_fired   = False
        self._open_price     = None
        self._or_close       = None
        self._or_volumes     = []
        self._pm_signal_fired = False
        self._pm_open_price   = None
        self._pm_or_close     = None
        self._pm_or_volumes   = []
        # _prev_or_volume persists across sessions (set externally by engine)

    def set_prev_or_volume(self, volume: float) -> None:
        """Set previous session's OR volume for relative volume comparison."""
        self._prev_or_volume = volume

    # ------------------------------------------------------------------
    # Core signal logic
    # ------------------------------------------------------------------

    def generate_signal(self, df: pd.DataFrame, current_index: int) -> Optional[Dict]:
        if current_index < 0 or current_index >= len(df):
            return None

        bar      = df.iloc[current_index]
        bar_time = bar["timestamp"].astimezone(_ET).time()

        # ── Accumulate MORNING opening range ────────────────────────────
        if _OR_START <= bar_time <= _OR_END:
            close = float(bar["close"])
            if self._open_price is None:
                self._open_price = float(bar["open"])
            self._or_close = close
            self._or_volumes.append(float(bar["volume"]))
            return None

        # ── Accumulate AFTERNOON opening range (if enabled) ─────────────
        if self.afternoon_entries and _PM_OR_START <= bar_time <= _PM_OR_END:
            close = float(bar["close"])
            if self._pm_open_price is None:
                self._pm_open_price = float(bar["open"])
            self._pm_or_close = close
            self._pm_or_volumes.append(float(bar["volume"]))
            return None

        # ── MORNING signal evaluation at 9:45 ───────────────────────────
        if bar_time == _ENTRY_BAR and not self._signal_fired:
            if self._open_price is None or self._or_close is None:
                return None
            momentum_pct = (self._or_close - self._open_price) / self._open_price * 100.0
            if momentum_pct < self.momentum_threshold:
                return None

            total_or_vol = sum(self._or_volumes)
            if self._prev_or_volume is not None:
                required_vol = self._prev_or_volume * self.volume_multiplier
                vol_ok = total_or_vol >= required_vol
            else:
                required_vol = self.or_volume_threshold
                vol_ok = total_or_vol > required_vol
            if not vol_ok:
                return None

            close = float(bar["close"])
            self._signal_fired = True
            signal = {
                "signal":        "BUY",
                "symbol":        None,
                "entry_price":   close,
                "momentum_pct":  round(momentum_pct, 4),
                "open_price":    round(self._open_price, 4),
                "bar_time":      bar["timestamp"],
                "reason":        "opening momentum (AM)",
            }
            logger.info(
                "Momentum BUY signal (AM) | entry=%.4f  momentum_pct=%.4f%%  time=%s",
                close, momentum_pct, bar["timestamp"],
            )
            return signal

        # ── AFTERNOON signal evaluation at 15:45 (if enabled) ───────────
        if (self.afternoon_entries
                and bar_time == _PM_ENTRY_BAR
                and not self._pm_signal_fired):
            if self._pm_open_price is None or self._pm_or_close is None:
                return None
            momentum_pct = (
                (self._pm_or_close - self._pm_open_price)
                / self._pm_open_price * 100.0
            )
            if momentum_pct < self.momentum_threshold:
                return None

            # Volume confirmation against AM OR (proxy for "normal" volume)
            total_or_vol = sum(self._pm_or_volumes)
            if self._prev_or_volume is not None:
                required_vol = self._prev_or_volume * self.volume_multiplier
                vol_ok = total_or_vol >= required_vol
            else:
                required_vol = self.or_volume_threshold
                vol_ok = total_or_vol > required_vol
            if not vol_ok:
                return None

            close = float(bar["close"])
            self._pm_signal_fired = True
            signal = {
                "signal":        "BUY",
                "symbol":        None,
                "entry_price":   close,
                "momentum_pct":  round(momentum_pct, 4),
                "open_price":    round(self._pm_open_price, 4),
                "bar_time":      bar["timestamp"],
                "reason":        "opening momentum (PM)",
            }
            logger.info(
                "Momentum BUY signal (PM) | entry=%.4f  momentum_pct=%.4f%%  time=%s",
                close, momentum_pct, bar["timestamp"],
            )
            return signal

        return None
