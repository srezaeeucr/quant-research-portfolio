import logging
import math
from datetime import datetime, time
from typing import Dict, Optional

import pandas as pd
import pytz

from src.risk.atr import calculate_atr

logger = logging.getLogger(__name__)

_ET = pytz.timezone("America/New_York")
_EOD_EXIT = time(15, 55)


class RiskManager:
    """Enforces per-trade sizing, daily loss limits, and exit rules.

    Stop-loss modes
    ---------------
    - Fixed (default):  stop = entry × stop_loss_pct%
    - ATR (opt-in):     stop = ATR(period) × atr_multiplier
                        clamped to [stop_floor_pct%, stop_ceiling_pct%] of entry
                        falls back to fixed if bars_df missing or has < period+1 bars
    """

    def __init__(self, config: Dict):
        risk_cfg = config.get("risk", {})
        acct_cfg = config.get("account", {})

        self.max_risk_per_trade_pct: float = risk_cfg.get("max_risk_per_trade_pct", 1.0)
        self.daily_loss_limit_pct:   float = risk_cfg.get("daily_loss_limit_pct", 2.0)
        self.reward_risk_ratio:      float = risk_cfg.get("reward_risk_ratio", 2.0)
        self.stop_loss_pct:          float = risk_cfg.get("stop_loss_pct", 0.5)   # % below entry
        self.default_balance:        float = acct_cfg.get("balance", 200.0)

        # ATR dynamic stops (opt-in)
        self.use_atr_stops:    bool  = bool(risk_cfg.get("use_atr_stops", False))
        self.atr_period:       int   = int(risk_cfg.get("atr_period", 14))
        self.atr_multiplier:   float = float(risk_cfg.get("atr_multiplier", 1.5))
        self.stop_floor_pct:   float = float(risk_cfg.get("stop_floor_pct", 0.3))
        self.stop_ceiling_pct: float = float(risk_cfg.get("stop_ceiling_pct", 1.5))

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def calculate_position(self, signal: Dict, account_balance: float,
                           buying_power: Optional[float] = None,
                           bars_df: Optional[pd.DataFrame] = None) -> Optional[Dict]:
        """Size a position from a BUY signal.

        Parameters
        ----------
        signal          : dict from a strategy (must contain 'entry_price', 'symbol')
        account_balance : current account equity in dollars
        buying_power    : available cash (optional); position cost is clamped to this limit
        bars_df         : optional OHLCV DataFrame for ATR calculation (only used
                          when ``use_atr_stops`` is enabled in config)

        Returns
        -------
        Position dict, or None if the trade is not viable (< 0.01 shares).
        """
        entry_price   = float(signal["entry_price"])
        risk_amount   = account_balance * self.max_risk_per_trade_pct / 100.0

        # Stop distance — ATR-based or fixed
        atr_stop_used = False
        atr_value:    Optional[float] = None
        if self.use_atr_stops and bars_df is not None:
            atr_value = calculate_atr(bars_df, period=self.atr_period)
            if atr_value is not None and atr_value > 0:
                stop_distance = atr_value * self.atr_multiplier
                # Floor + ceiling clamps in price units
                floor_dist   = entry_price * self.stop_floor_pct   / 100.0
                ceiling_dist = entry_price * self.stop_ceiling_pct / 100.0
                stop_distance = max(stop_distance, floor_dist)
                stop_distance = min(stop_distance, ceiling_dist)
                atr_stop_used = True
            else:
                # Fall back to fixed
                stop_distance = entry_price * (self.stop_loss_pct / 100.0)
        else:
            stop_distance = entry_price * (self.stop_loss_pct / 100.0)

        stop_loss = entry_price - stop_distance
        target    = entry_price + stop_distance * self.reward_risk_ratio

        if stop_distance <= 0:
            logger.warning("stop_distance is zero for entry %.4f — skipping", entry_price)
            return None

        shares = round(risk_amount / stop_distance, 2)

        if shares < 0.01:
            logger.info(
                "Position too small: risk_amount=%.2f stop_distance=%.4f → %.2f shares",
                risk_amount, stop_distance, shares,
            )
            return None

        buying_power_clamped = False
        if buying_power is not None and shares * entry_price > buying_power:
            shares = math.floor(buying_power / entry_price * 100) / 100  # floor at 2dp
            buying_power_clamped = True
            logger.info(
                "Buying power clamp applied: buying_power=%.2f → shares=%.2f",
                buying_power, shares,
            )
            if shares < 0.01:
                logger.info("Shares < 0.01 after buying power clamp — skipping")
                return None

        position_cost = shares * entry_price
        reward        = stop_distance * self.reward_risk_ratio * shares
        # Recalculate actual risk based on clamped shares
        actual_risk   = stop_distance * shares

        position = {
            "symbol":               signal["symbol"],
            "shares":               shares,
            "entry_price":          entry_price,
            "stop_loss":            round(stop_loss, 4),
            "target":               round(target, 4),
            "risk_amount":          round(actual_risk, 4),
            "position_cost":        round(position_cost, 4),
            "reward":               round(reward, 4),
            "fractional":           shares != int(shares),
            "buying_power_clamped": buying_power_clamped,
            "atr_stop_used":        atr_stop_used,
            "atr_value":            (round(atr_value, 4) if atr_value is not None else None),
            "stop_distance_pct":    round(stop_distance / entry_price * 100, 4),
        }

        logger.info(
            "Position sized | %s  shares=%.2f  entry=%.2f  stop=%.2f  target=%.2f  "
            "risk=$%.2f  reward=$%.2f  bp_clamped=%s",
            signal["symbol"], shares, entry_price, stop_loss, target,
            actual_risk, reward, buying_power_clamped,
        )
        return position

    # ------------------------------------------------------------------
    # Daily loss limit
    # ------------------------------------------------------------------

    def check_daily_loss_limit(self, daily_pnl: float, account_balance: float) -> bool:
        """Return True (halt trading) if daily loss limit has been reached.

        Parameters
        ----------
        daily_pnl       : realised P&L so far today (negative = loss)
        account_balance : current account equity

        Returns
        -------
        True  → stop trading for the day
        False → within limits, continue
        """
        limit = account_balance * self.daily_loss_limit_pct / 100.0
        breached = daily_pnl <= -limit
        if breached:
            logger.warning(
                "Daily loss limit hit: pnl=%.2f  limit=%.2f  balance=%.2f",
                daily_pnl, limit, account_balance,
            )
        return breached

    # ------------------------------------------------------------------
    # Exit checks
    # ------------------------------------------------------------------

    def check_exit(self, position: Dict, current_price: float,
                   current_dt: Optional[datetime] = None) -> Optional[Dict]:
        """Evaluate whether an open position should be closed.

        Parameters
        ----------
        position      : dict returned by calculate_position
        current_price : latest bar close price
        current_dt    : bar timestamp (timezone-aware); uses now() if omitted

        Returns
        -------
        {'action': 'CLOSE', 'reason': ..., 'exit_price': ...}  or None
        """
        if current_dt is None:
            current_dt = datetime.now(_ET)

        bar_time = current_dt.astimezone(_ET).time()

        if current_price <= position["stop_loss"]:
            return {"action": "CLOSE", "reason": "stop_loss",  "exit_price": current_price}

        if current_price >= position["target"]:
            return {"action": "CLOSE", "reason": "target_hit", "exit_price": current_price}

        if bar_time >= _EOD_EXIT:
            return {"action": "CLOSE", "reason": "eod_exit",   "exit_price": current_price}

        return None
