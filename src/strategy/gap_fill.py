"""
Gap Fill Strategy — trade gap-down opens expecting price to recover to previous close.
"""
import logging
from datetime import time
from typing import Dict, Optional

import pandas as pd
import pytz

from src.strategy.base import BaseStrategy

logger = logging.getLogger(__name__)

_ET = pytz.timezone("America/New_York")

_ENTRY_START  = time(9, 30)
_ENTRY_CUTOFF = time(10, 0)   # first 30 minutes only


class GapFillStrategy(BaseStrategy):
    """Gap Fill (long-only).

    Logic
    -----
    1. On the first bar at/after 9:30 ET, compute:
           gap_pct = (bar_open − prev_close) / prev_close × 100
    2. Only trade gap-DOWN days (gap_pct ≤ −min_gap_pct).
    3. Only trade moderate gaps: abs(gap_pct) ≤ max_gap_pct (skip extreme gaps).
    4. BUY signal fires immediately; entry_price = bar close.
    5. gap_fill_target = prev_close (the level the price should fill back to).
    6. One signal per session; only in first 30 minutes.

    Config (reads from strategy_params.gap_fill, falls back to strategy):
        min_gap_pct: 0.3    minimum gap % to trade
        max_gap_pct: 2.0    maximum gap % (skip extreme gaps)

    The engine must call set_prev_close() before iterating each day's bars.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        params    = config.get("strategy_params", {}).get("gap_fill", {})
        strat_cfg = config.get("strategy", {})  # backward compat
        self.min_gap_pct: float = float(params.get("min_gap_pct", strat_cfg.get("min_gap_pct", 0.3)))
        self.max_gap_pct: float = float(params.get("max_gap_pct", strat_cfg.get("max_gap_pct", 2.0)))
        # afternoon_entries accepted for grid uniformity, but Gap Fill is
        # inherently a morning-open strategy — gaps only exist at the open.
        self.afternoon_entries: bool = bool(config.get("afternoon_entries", False))

        self._signal_fired: bool  = False
        self._gap_computed: bool  = False
        self._prev_close: Optional[float] = None

    @property
    def name(self) -> str:
        return "Gap Fill"

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def reset_session(self) -> None:
        self._signal_fired = False
        self._gap_computed = False
        # _prev_close intentionally NOT reset — engine sets it via set_prev_close()

    def set_prev_close(self, prev_close: Optional[float]) -> None:
        """Set previous session's closing price.  Call once before iterating bars."""
        self._prev_close = prev_close

    # ------------------------------------------------------------------
    # Core signal logic
    # ------------------------------------------------------------------

    def generate_signal(self, df: pd.DataFrame, current_index: int) -> Optional[Dict]:
        if current_index < 0 or current_index >= len(df):
            return None

        bar      = df.iloc[current_index]
        bar_time = bar["timestamp"].astimezone(_ET).time()

        # Only first 30 minutes
        if bar_time >= _ENTRY_CUTOFF:
            return None

        if self._signal_fired or self._prev_close is None:
            return None

        # Compute gap on the first qualifying bar
        if not self._gap_computed and bar_time >= _ENTRY_START:
            open_price = float(bar["open"])
            gap_pct    = (open_price - self._prev_close) / self._prev_close * 100.0
            self._gap_computed = True

            logger.debug(
                "Gap computed: open=%.4f  prev_close=%.4f  gap_pct=%.4f%%",
                open_price, self._prev_close, gap_pct,
            )

            # Long-only: only gap-DOWN plays
            if gap_pct > -self.min_gap_pct:
                logger.debug("Gap too small or upward gap — skipping")
                return None

            # Skip extreme gaps
            if abs(gap_pct) > self.max_gap_pct:
                logger.debug("Gap too large (%.4f%% > %.1f%%) — skipping", abs(gap_pct), self.max_gap_pct)
                return None

            close           = float(bar["close"])
            gap_fill_target = self._prev_close

            self._signal_fired = True
            signal = {
                "signal":          "BUY",
                "symbol":          None,
                "entry_price":     close,
                "gap_pct":         round(gap_pct, 4),
                "prev_close":      round(self._prev_close, 4),
                "gap_fill_target": round(gap_fill_target, 4),
                "bar_time":        bar["timestamp"],
                "reason":          "gap fill bounce",
            }
            logger.info(
                "GapFill BUY signal | entry=%.4f  gap_pct=%.4f%%  target=%.4f  time=%s",
                close, gap_pct, gap_fill_target, bar["timestamp"],
            )
            return signal

        return None
