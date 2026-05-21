"""
EMA Crossover Strategy — golden cross on 1-min bars.

Supports two entry windows:
  Morning:   9:45–10:30 ET
  Afternoon: 15:30–15:50 ET (optional, controlled by afternoon_entry_enabled)
"""
import logging
from datetime import time
from typing import Dict, List, Optional

import pandas as pd
import pytz

from src.strategy.base import BaseStrategy

logger = logging.getLogger(__name__)

_ET = pytz.timezone("America/New_York")

_SESSION_START   = time(9, 30)
_AM_ENTRY_START  = time(9, 45)
_DEFAULT_AM_ENTRY_END = time(10, 30)
_PM_ENTRY_START  = time(15, 30)
_PM_ENTRY_END    = time(15, 50)


class EMACrossoverStrategy(BaseStrategy):
    """EMA Crossover (fast/slow golden cross).

    Logic
    -----
    1. From 9:30 ET, accumulate 1-min close prices.
    2. Compute fast and slow EMAs via pandas Series.ewm(span=N, adjust=False).mean().
    3. BUY signal when fast EMA crosses ABOVE slow EMA (golden cross):
           previous bar: fast_ema < slow_ema
           current  bar: fast_ema > slow_ema
    4. Morning entry window: 9:45–10:30 ET.
    5. Afternoon entry window: 15:30–15:50 ET (if afternoon_entry_enabled=true).
    6. Volume confirmation: bar volume >= volume_multiplier × avg session volume so far.
    7. One signal per window (morning and afternoon fire independently).

    Config (reads from strategy_params.ema_crossover, falls back to strategy):
        ema_fast: 9
        ema_slow: 21
        volume_multiplier: 1.2
        afternoon_entry_enabled: false
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        params    = config.get("strategy_params", {}).get("ema_crossover", {})
        strat_cfg = config.get("strategy", {})  # backward compat
        self.fast_period:       int   = int(params.get("ema_fast",
                                            strat_cfg.get("ema_fast_period", 9)))
        self.slow_period:       int   = int(params.get("ema_slow",
                                            strat_cfg.get("ema_slow_period", 21)))
        self.volume_multiplier: float = float(params.get("volume_multiplier",
                                              strat_cfg.get("volume_multiplier", 1.2)))
        # Read afternoon_entries from top-level (set by engine override) OR
        # from per-strategy params (config.yaml). Either enables PM window.
        self.afternoon_entry_enabled: bool = bool(
            config.get("afternoon_entries", False) or
            params.get("afternoon_entry_enabled", False) or
            params.get("afternoon_entries", False)
        )

        # Configurable AM entry window end — minutes after 9:45 ET.
        # Default: 45 min → cutoff at 10:30 ET.
        cutoff_min = int(params.get("entry_window_minutes",
                         config.get("entry_window_minutes", 45)))
        h, m = divmod(9 * 60 + 45 + cutoff_min, 60)
        self.am_entry_end = time(h, m)

        self._reset()

    @property
    def name(self) -> str:
        return "EMA Crossover"

    # ------------------------------------------------------------------
    # Internal reset helper
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        self._am_signal_fired: bool = False
        self._pm_signal_fired: bool = False
        self._prices:  List[float] = []
        self._volumes: List[float] = []

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def reset_session(self) -> None:
        self._reset()

    # ------------------------------------------------------------------
    # Core signal logic
    # ------------------------------------------------------------------

    def _in_entry_window(self, bar_time: time) -> Optional[str]:
        """Return 'am' or 'pm' if bar_time is inside an entry window, else None."""
        if _AM_ENTRY_START <= bar_time <= self.am_entry_end:
            return "am"
        if (self.afternoon_entry_enabled
                and _PM_ENTRY_START <= bar_time <= _PM_ENTRY_END):
            return "pm"
        return None

    def generate_signal(self, df: pd.DataFrame, current_index: int) -> Optional[Dict]:
        if current_index < 0 or current_index >= len(df):
            return None

        bar      = df.iloc[current_index]
        bar_time = bar["timestamp"].astimezone(_ET).time()

        # Only accumulate bars from 9:30 onward
        if bar_time < _SESSION_START:
            return None

        close   = float(bar["close"])
        bar_vol = float(bar["volume"])
        self._prices.append(close)
        self._volumes.append(bar_vol)
        n = len(self._prices)

        # Need at least slow_period + 1 bars to detect a crossover
        if n < self.slow_period + 1:
            return None

        # Check if we're in an entry window
        window = self._in_entry_window(bar_time)
        if window is None:
            return None

        # One signal per window
        if window == "am" and self._am_signal_fired:
            return None
        if window == "pm" and self._pm_signal_fired:
            return None

        # Compute full EMA series via pandas ewm (adjust=False = recursive formula)
        series         = pd.Series(self._prices)
        ema_fast_series = series.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow_series = series.ewm(span=self.slow_period, adjust=False).mean()

        ema_fast      = float(ema_fast_series.iloc[-1])
        ema_slow      = float(ema_slow_series.iloc[-1])
        prev_ema_fast = float(ema_fast_series.iloc[-2])
        prev_ema_slow = float(ema_slow_series.iloc[-2])

        # Golden cross: fast crossed above slow
        if not (prev_ema_fast <= prev_ema_slow and ema_fast > ema_slow):
            return None

        # Volume confirmation
        avg_vol = sum(self._volumes) / len(self._volumes)
        if bar_vol < avg_vol * self.volume_multiplier:
            logger.debug(
                "EMA crossover volume rejected: bar_vol=%.0f  avg=%.0f  required=%.0f",
                bar_vol, avg_vol, avg_vol * self.volume_multiplier,
            )
            return None

        if window == "am":
            self._am_signal_fired = True
        else:
            self._pm_signal_fired = True

        signal = {
            "signal":      "BUY",
            "symbol":      None,
            "entry_price": close,
            "ema_fast":    round(ema_fast, 4),
            "ema_slow":    round(ema_slow, 4),
            "bar_time":    bar["timestamp"],
            "reason":      "EMA golden cross",
        }
        logger.info(
            "EMA BUY signal (%s) | entry=%.4f  ema_fast=%.4f  ema_slow=%.4f  time=%s",
            window.upper(), close, ema_fast, ema_slow, bar["timestamp"],
        )
        return signal
