"""Hybrid Filter Strategy — asymmetric ensemble.

Logic
-----
- ONE primary strategy generates the entry trigger
- N filter strategies vote yes/no
- BUY only if primary fires AND ≥min_filters_agree filter strategies also fire

Different from EnsembleStrategy (symmetric N-of-N agreement). Lets a strong
primary keep its trade frequency while filtering out the worst setups.
"""
import copy
import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.strategy.base import BaseStrategy

# Sub-strategy classes (avoid circular import via engine)
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


class HybridFilterStrategy(BaseStrategy):
    """Primary strategy + filter agreement requirement."""

    def __init__(self, config: Dict):
        super().__init__(config)
        params = config.get("strategy_params", {}).get("hybrid_filter", {})
        primary_name: str = params.get("primary", "orb")
        filter_names: List[str] = params.get("filters",
                                              ["rsi_reversion", "macd_crossover",
                                               "bollinger_reversal", "donchian_breakout"])
        self.min_filters_agree: int = int(params.get("min_filters_agree", 1))
        # Filter "lookback" — when primary fires, look back N bars to see if any
        # filter fired recently (since they may have fired moments before)
        self.filter_lookback: int = int(params.get("filter_lookback", 5))

        # Build instances
        primary_cls = _SUB_MAP.get(primary_name)
        if primary_cls is None:
            raise ValueError(f"Unknown primary strategy: {primary_name}")
        primary_cfg = copy.deepcopy(config)
        primary_cfg["active_strategy"] = primary_name
        primary_cfg.setdefault("strategy", {})["active_strategy"] = primary_name
        self.primary_name = primary_name
        self.primary: BaseStrategy = primary_cls(primary_cfg)

        self.filter_strats: List[Tuple[str, BaseStrategy]] = []
        for fname in filter_names:
            cls = _SUB_MAP.get(fname)
            if cls is None: continue
            f_cfg = copy.deepcopy(config)
            f_cfg["active_strategy"] = fname
            f_cfg.setdefault("strategy", {})["active_strategy"] = fname
            self.filter_strats.append((fname, cls(f_cfg)))

        # Track filter "fire history" within session
        self._filter_fire_bars: Dict[str, List[int]] = {fname: [] for fname, _ in self.filter_strats}
        self._signal_fired = False
        self.current_regime_sma: Optional[float] = None
        self._opening_high: Optional[float] = None
        self._opening_low: Optional[float] = None

    @property
    def name(self) -> str:
        return f"Hybrid({self.primary_name})"

    def reset_session(self) -> None:
        self._signal_fired = False
        self.primary.reset_session()
        for _, sub in self.filter_strats:
            sub.reset_session()
        self._filter_fire_bars = {fname: [] for fname, _ in self.filter_strats}

    def set_regime_sma(self, sma: Optional[float]) -> None:
        self.current_regime_sma = sma
        if hasattr(self.primary, "set_regime_sma"):
            self.primary.set_regime_sma(sma)
        for _, sub in self.filter_strats:
            if hasattr(sub, "set_regime_sma"):
                sub.set_regime_sma(sma)

    def set_prev_close(self, val) -> None:
        if hasattr(self.primary, "set_prev_close"):
            self.primary.set_prev_close(val)
        for _, sub in self.filter_strats:
            if hasattr(sub, "set_prev_close"):
                sub.set_prev_close(val)

    def set_prev_or_volume(self, val) -> None:
        if hasattr(self.primary, "set_prev_or_volume"):
            self.primary.set_prev_or_volume(val)
        for _, sub in self.filter_strats:
            if hasattr(sub, "set_prev_or_volume"):
                sub.set_prev_or_volume(val)

    def generate_signal(self, df: pd.DataFrame, current_index: int) -> Optional[Dict]:
        if self._signal_fired:
            return None

        # Update filter fire history (track across the session)
        for fname, sub in self.filter_strats:
            try:
                fsig = sub.generate_signal(df, current_index)
            except Exception:
                fsig = None
            if fsig:
                self._filter_fire_bars[fname].append(current_index)
                # Reset so it can keep firing
                sub._signal_fired = False
                if hasattr(sub, "_pm_signal_fired"):
                    sub._pm_signal_fired = False

        # Check primary
        primary_sig = self.primary.generate_signal(df, current_index)
        # Mirror primary's OR for health check
        if hasattr(self.primary, "_opening_high"):
            self._opening_high = getattr(self.primary, "_opening_high", None)
            self._opening_low  = getattr(self.primary, "_opening_low", None)

        if not primary_sig:
            return None

        # Count filters that fired within lookback of current bar
        agree_count = 0
        agree_names = []
        for fname, _ in self.filter_strats:
            recent_fires = [b for b in self._filter_fire_bars[fname]
                            if current_index - self.filter_lookback <= b <= current_index]
            if recent_fires:
                agree_count += 1
                agree_names.append(fname)

        if agree_count < self.min_filters_agree:
            # Primary fired but not enough filter agreement — reset primary so
            # it could fire again later (in case a filter agrees later).
            # Actually — primary should NOT keep retrying same signal; let it stay armed.
            # But the primary's own _signal_fired flag was set when it fired.
            # We need to reset that to allow re-evaluation if config requires.
            # For now, treat as "missed opportunity" — primary won't fire again.
            return None

        # All conditions met — fire the buy at primary's signal
        self._signal_fired = True
        primary_sig["reason"] = (
            f"Hybrid: {self.primary_name} fired + {agree_count}/{len(self.filter_strats)} filters: "
            f"{','.join(agree_names)}"
        )
        return primary_sig
