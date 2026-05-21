import logging
from datetime import time
from typing import Dict, Optional

import pandas as pd
import pytz

from src.strategy.base import BaseStrategy

logger = logging.getLogger(__name__)

_ET = pytz.timezone("America/New_York")

# Morning opening range: 9:30 (inclusive) through 9:44 (inclusive) = 15 bars
_ORB_START = time(9, 30)
_ORB_END   = time(9, 44)

# Morning entry window: default cutoff is 10:00 ET (configurable)
_DEFAULT_ENTRY_CUTOFF = time(10, 0)

# Afternoon "second OR" — only used when afternoon_entries=True
_PM_OR_START   = time(15, 30)
_PM_OR_END     = time(15, 44)
_PM_ENTRY_END  = time(15, 50)


class ORBStrategy(BaseStrategy):
    """Opening Range Breakout strategy.

    Logic
    -----
    1. Opening range  : first 15 minutes (9:30–9:44 ET) → defines opening_high / opening_low
    2. Breakout check : on each bar after 9:44, if close exceeds opening_high by at least
                        min_breakout_pct → emit a BUY signal (one per session, AM window only)
    3. Afternoon      : 15:30–16:00 window is reserved for exits; no new entry signals
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        strategy_cfg = config.get("strategy", {})
        params       = config.get("strategy_params", {}).get("orb", {})
        self.min_breakout_pct:  float = strategy_cfg.get("min_breakout_pct", 0.1) / 100.0
        self.volume_multiplier: float = strategy_cfg.get("volume_multiplier", 1.5)
        self.use_regime_filter: bool  = bool(strategy_cfg.get("use_regime_filter", False))
        self.regime_sma_days:   int   = int(strategy_cfg.get("regime_sma_days", 20))
        self.afternoon_entries: bool = bool(config.get("afternoon_entries", False))

        # Configurable entry cutoff — how many minutes after 9:45 are entries allowed?
        # Default: 15 min → cutoff at 10:00 ET. Set via config or engine override.
        cutoff_min = int(params.get("entry_window_minutes",
                         config.get("entry_window_minutes", 15)))
        h, m = divmod(9 * 60 + 45 + cutoff_min, 60)
        self.entry_cutoff = time(h, m)

        self._signal_fired: bool = False   # AM signal flag
        self._opening_high: Optional[float] = None
        self._opening_low:  Optional[float] = None
        self._or_volumes:   list  = []     # AM OR bar volumes

        # Afternoon (PM) state — only used when afternoon_entries=True
        self._pm_signal_fired: bool = False
        self._pm_opening_high: Optional[float] = None
        self._pm_opening_low:  Optional[float] = None
        self._pm_or_volumes:   list = []

        self.current_regime_sma: Optional[float] = None  # set by engine before each day

    @property
    def name(self) -> str:
        return "ORB"

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def reset_session(self) -> None:
        """Call once per trading day before iterating bars."""
        self._signal_fired = False
        self._opening_high = None
        self._opening_low  = None
        self._or_volumes   = []
        # PM state
        self._pm_signal_fired = False
        self._pm_opening_high = None
        self._pm_opening_low  = None
        self._pm_or_volumes   = []
        # current_regime_sma is intentionally NOT reset here — the engine sets it
        # for each day after reset_session() via set_regime_sma().

    def set_regime_sma(self, sma: Optional[float]) -> None:
        """Set the 20-day SMA value the engine pre-computed for today's date.

        Call this once per trading day before iterating bars.
        Pass None to disable the filter for a day (e.g. during SMA warmup).
        """
        self.current_regime_sma = sma

    # ------------------------------------------------------------------
    # Core signal logic
    # ------------------------------------------------------------------

    def generate_signal(self, df: pd.DataFrame, current_index: int) -> Optional[Dict]:
        """Evaluate bar at current_index and return a BUY signal dict or None."""
        if current_index < 0 or current_index >= len(df):
            return None

        bar      = df.iloc[current_index]
        bar_time = bar["timestamp"].astimezone(_ET).time()
        close    = float(bar["close"])

        # ---- Build / update MORNING opening range ------------------------
        if _ORB_START <= bar_time <= _ORB_END:
            if self._opening_high is None or close > self._opening_high:
                self._opening_high = close
            if self._opening_low is None or close < self._opening_low:
                self._opening_low = close
            self._or_volumes.append(float(bar["volume"]))
            return None

        # ---- Build / update AFTERNOON opening range (if enabled) ---------
        if self.afternoon_entries and _PM_OR_START <= bar_time <= _PM_OR_END:
            if self._pm_opening_high is None or close > self._pm_opening_high:
                self._pm_opening_high = close
            if self._pm_opening_low is None or close < self._pm_opening_low:
                self._pm_opening_low = close
            self._pm_or_volumes.append(float(bar["volume"]))
            return None

        # ---- Determine which entry window we're in -----------------------
        in_am = (bar_time > _ORB_END and bar_time < self.entry_cutoff)
        in_pm = (
            self.afternoon_entries
            and bar_time > _PM_OR_END
            and bar_time <= _PM_ENTRY_END
        )
        if not (in_am or in_pm):
            return None

        # Pick the right OR (morning vs afternoon)
        if in_am:
            if self._signal_fired:
                return None
            if self._opening_high is None:
                return None
            opening_high = self._opening_high
            opening_low  = self._opening_low
            or_volumes   = self._or_volumes
        else:
            if self._pm_signal_fired:
                return None
            if self._pm_opening_high is None:
                return None
            opening_high = self._pm_opening_high
            opening_low  = self._pm_opening_low
            or_volumes   = self._pm_or_volumes

        threshold = opening_high * (1.0 + self.min_breakout_pct)
        if close <= threshold:
            return None

        # Volume confirmation
        if or_volumes:
            avg_or_volume = sum(or_volumes) / len(or_volumes)
            if float(bar["volume"]) < avg_or_volume * self.volume_multiplier:
                logger.debug(
                    "Volume filter rejected: bar_vol=%.0f  avg_or_vol=%.0f  required=%.0f",
                    float(bar["volume"]), avg_or_volume, avg_or_volume * self.volume_multiplier,
                )
                return None

        # Market-regime filter (applies to both AM and PM)
        if self.use_regime_filter and self.current_regime_sma is not None:
            if close < self.current_regime_sma:
                logger.debug(
                    "Regime filter rejected signal: close=%.4f < SMA(%.0f)=%.4f",
                    close, self.regime_sma_days, self.current_regime_sma,
                )
                return None

        if in_am:
            self._signal_fired = True
            window = "AM"
        else:
            self._pm_signal_fired = True
            window = "PM"

        signal = {
            "signal":           "BUY",
            "symbol":           None,
            "entry_price":      close,
            "opening_high":     opening_high,
            "opening_low":      opening_low,
            "bar_time":         bar["timestamp"],
            "reason":           f"ORB breakout above opening range ({window})",
            "volume_confirmed": True,
        }
        logger.info(
            "ORB BUY signal (%s) | entry=%.4f  OR_high=%.4f  OR_low=%.4f  time=%s",
            window, close, opening_high, opening_low, bar["timestamp"],
        )
        return signal

    # ------------------------------------------------------------------
    # Convenience: run over a full day's DataFrame
    # ------------------------------------------------------------------

    def run_day(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """Iterate all bars in df for a single trading day; return first signal or None."""
        self.reset_session()
        for i in range(len(df)):
            signal = self.generate_signal(df, i)
            if signal is not None:
                signal["symbol"] = symbol
                return signal
        return None
