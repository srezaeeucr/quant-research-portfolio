"""
VWAP Reversion Strategy — buy dips back to VWAP during the morning session.
"""
import logging
from datetime import time
from typing import Dict, Optional

import pandas as pd
import pytz

from src.strategy.base import BaseStrategy

logger = logging.getLogger(__name__)

_ET = pytz.timezone("America/New_York")

_VWAP_START     = time(9, 30)
_ENTRY_START    = time(9, 45)
_ENTRY_END      = time(11, 30)
_PM_ENTRY_START = time(15, 30)
_PM_ENTRY_END   = time(15, 50)


class VWAPReversionStrategy(BaseStrategy):
    """VWAP Reversion.

    Logic
    -----
    1. Running VWAP is computed from 9:30 ET using close × volume.
    2. BUY signal when: (vwap - close) / vwap >= deviation_threshold
       AND current bar closes ABOVE previous bar (bounce confirmation).
    3. Entry window: 9:45–11:30 ET only.
    4. One signal per session.

    Config (reads from strategy_params.vwap_reversion, falls back to strategy):
        deviation_threshold: 0.002  (0.2% dip below VWAP)
        bounce_bars: 1
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        params    = config.get("strategy_params", {}).get("vwap_reversion", {})
        strat_cfg = config.get("strategy", {})  # backward compat
        # New key: deviation_threshold (fraction); old key: vwap_dip_pct (percent)
        if "deviation_threshold" in params:
            self.deviation_threshold: float = float(params["deviation_threshold"])
        else:
            self.deviation_threshold = float(strat_cfg.get("vwap_dip_pct", 0.2)) / 100.0

        self.afternoon_entries: bool = bool(
            config.get("afternoon_entries", False) or
            params.get("afternoon_entries", False)
        )

        self._signal_fired: bool = False
        self._pm_signal_fired: bool = False
        self._cum_pv:   float = 0.0   # cumulative close × volume
        self._cum_vol:  float = 0.0   # cumulative volume
        self._prev_close: Optional[float] = None
        self._dip_active: bool = False   # dip condition met on a prior bar
        self._dip_vwap:   float = 0.0   # VWAP at the time dip was detected
        self._dip_dev:    float = 0.0   # deviation at the time dip was detected

    @property
    def name(self) -> str:
        return "VWAP Reversion"

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def reset_session(self) -> None:
        self._signal_fired = False
        self._pm_signal_fired = False
        self._cum_pv   = 0.0
        self._cum_vol  = 0.0
        self._prev_close = None
        self._dip_active = False
        self._dip_vwap   = 0.0
        self._dip_dev    = 0.0

    # ------------------------------------------------------------------
    # Core signal logic
    # ------------------------------------------------------------------

    def generate_signal(self, df: pd.DataFrame, current_index: int) -> Optional[Dict]:
        if current_index < 0 or current_index >= len(df):
            return None

        bar      = df.iloc[current_index]
        bar_time = bar["timestamp"].astimezone(_ET).time()

        # Only accumulate VWAP from 9:30 onward
        if bar_time < _VWAP_START:
            return None

        # Update running VWAP using close × volume
        close   = float(bar["close"])
        bar_vol = float(bar["volume"])
        self._cum_pv  += close * bar_vol
        self._cum_vol += bar_vol

        if self._cum_vol <= 0:
            self._prev_close = close
            return None

        vwap = self._cum_pv / self._cum_vol

        # Determine which window we're in (or None)
        in_am = _ENTRY_START <= bar_time <= _ENTRY_END
        in_pm = (
            self.afternoon_entries
            and _PM_ENTRY_START <= bar_time <= _PM_ENTRY_END
        )
        if not (in_am or in_pm):
            self._prev_close = close
            return None

        # One signal per window
        already_fired = (
            (in_am and self._signal_fired) or
            (in_pm and self._pm_signal_fired)
        )
        if already_fired:
            self._prev_close = close
            return None

        # Dip condition: check whether close is >= deviation_threshold below VWAP.
        deviation = (vwap - close) / vwap  # positive when close < vwap
        if deviation >= self.deviation_threshold:
            self._dip_active = True
            self._dip_vwap   = vwap
            self._dip_dev    = deviation

        # Bounce confirmation
        if self._dip_active and self._prev_close is not None and close > self._prev_close:
            if in_am:
                self._signal_fired = True
            else:
                self._pm_signal_fired = True
            signal = {
                "signal":        "BUY",
                "symbol":        None,
                "entry_price":   close,
                "vwap":          round(self._dip_vwap, 4),
                "deviation_pct": round(self._dip_dev * 100, 4),
                "bar_time":      bar["timestamp"],
                "reason":        "VWAP reversion bounce",
            }
            logger.info(
                "VWAP BUY signal | entry=%.4f  vwap=%.4f  dev=%.4f%%  time=%s",
                close, self._dip_vwap, self._dip_dev * 100, bar["timestamp"],
            )
            self._prev_close = close
            return signal

        self._prev_close = close
        return None
