"""Ensemble Strategy — fires only when ≥N sub-strategies agree.

Validates the GPU Mega Research finding: "When 5/5 strategies agree, 100%
of configs are profitable." This wraps a list of sub-strategies and fires
a BUY signal only when the agreement count reaches a configurable threshold.

Each sub-strategy is queried on every bar, but their internal _signal_fired
flag is reset after the call so they can keep "voting" for the rest of the
session. The ensemble's own _signal_fired flag enforces one entry per session.
"""
import copy
import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.strategy.base import BaseStrategy

# Direct imports of sub-strategy classes (avoids circular import via engine)
from src.strategy.orb import ORBStrategy
from src.strategy.ema_crossover import EMACrossoverStrategy
from src.strategy.momentum import MomentumStrategy
from src.strategy.macd_crossover import MACDCrossoverStrategy
from src.strategy.rsi_reversion import RSIReversionStrategy
from src.strategy.vwap_bands import VWAPBandsStrategy
from src.strategy.bollinger_reversal import BollingerReversalStrategy
from src.strategy.donchian_breakout import DonchianBreakoutStrategy
from src.strategy.stochastic_crossover import StochasticCrossoverStrategy
from src.strategy.inside_bar_breakout import InsideBarBreakoutStrategy
from src.strategy.vwap_reversion import VWAPReversionStrategy
from src.strategy.gap_fill import GapFillStrategy

logger = logging.getLogger(__name__)

_SUB_MAP = {
    "orb":                  ORBStrategy,
    "ema_crossover":        EMACrossoverStrategy,
    "momentum":             MomentumStrategy,
    "macd_crossover":       MACDCrossoverStrategy,
    "rsi_reversion":        RSIReversionStrategy,
    "vwap_bands":           VWAPBandsStrategy,
    "bollinger_reversal":   BollingerReversalStrategy,
    "donchian_breakout":    DonchianBreakoutStrategy,
    "stochastic_crossover": StochasticCrossoverStrategy,
    "inside_bar_breakout":  InsideBarBreakoutStrategy,
    "vwap_reversion":       VWAPReversionStrategy,
    "gap_fill":             GapFillStrategy,
}


class EnsembleStrategy(BaseStrategy):
    """Fires when ≥threshold sub-strategies agree on entry."""

    def __init__(self, config: Dict):
        super().__init__(config)
        params = config.get("strategy_params", {}).get("ensemble", {})
        self.threshold: int = int(params.get("threshold", 2))
        sub_names: List[str] = params.get("sub_strategies",
                                           ["orb", "rsi_reversion", "macd_crossover"])

        self.subs: List[Tuple[str, BaseStrategy]] = []
        for sname in sub_names:
            cls = _SUB_MAP.get(sname)
            if cls is None:
                logger.warning("Ensemble: unknown sub-strategy '%s' — skipping", sname)
                continue
            sub_cfg = copy.deepcopy(config)
            sub_cfg["active_strategy"] = sname
            sub_cfg.setdefault("strategy", {})["active_strategy"] = sname
            self.subs.append((sname, cls(sub_cfg)))

        self._signal_fired = False
        self.current_regime_sma: Optional[float] = None
        # For ORB-style sub-strategies that track _opening_high — surface it for
        # the bot's health check.
        self._opening_high: Optional[float] = None
        self._opening_low: Optional[float] = None

    @property
    def name(self) -> str:
        return "Ensemble"

    def reset_session(self) -> None:
        self._signal_fired = False
        for _, sub in self.subs:
            sub.reset_session()

    def set_regime_sma(self, sma: Optional[float]) -> None:
        self.current_regime_sma = sma
        for _, sub in self.subs:
            if hasattr(sub, "set_regime_sma"):
                sub.set_regime_sma(sma)

    def set_prev_close(self, val) -> None:
        for _, sub in self.subs:
            if hasattr(sub, "set_prev_close"):
                sub.set_prev_close(val)

    def set_prev_or_volume(self, val) -> None:
        for _, sub in self.subs:
            if hasattr(sub, "set_prev_or_volume"):
                sub.set_prev_or_volume(val)

    def generate_signal(self, df: pd.DataFrame, current_index: int) -> Optional[Dict]:
        if self._signal_fired:
            return None
        if current_index < 0 or current_index >= len(df):
            return None

        fired_subs: List[str] = []
        for sname, sub in self.subs:
            try:
                sig = sub.generate_signal(df, current_index)
            except Exception as e:
                logger.debug("Ensemble sub %s error: %s", sname, e)
                sig = None
            if sig:
                fired_subs.append(sname)
                # Reset the sub's flag so it can keep voting on later bars.
                sub._signal_fired = False
                if hasattr(sub, "_pm_signal_fired"):
                    sub._pm_signal_fired = False

        # Mirror first ORB sub's opening range for health checks
        for sname, sub in self.subs:
            if sname == "orb" and hasattr(sub, "_opening_high"):
                self._opening_high = getattr(sub, "_opening_high", None)
                self._opening_low  = getattr(sub, "_opening_low", None)
                break

        if len(fired_subs) < self.threshold:
            return None

        bar = df.iloc[current_index]
        close = float(bar["close"])
        self._signal_fired = True
        return {
            "signal":      "BUY",
            "symbol":      None,
            "entry_price": close,
            "bar_time":    bar["timestamp"],
            "reason":      f"Ensemble {len(fired_subs)}/{len(self.subs)} agreed: {','.join(fired_subs)}",
            "volume_confirmed": True,
        }
