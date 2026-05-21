import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional

import pytz

from src.broker.base import BaseBroker

logger = logging.getLogger(__name__)

_ET = pytz.timezone("America/New_York")


class SimulatorBroker(BaseBroker):
    """Paper / backtest broker that simulates fills in memory.

    All fills are assumed to occur at the price provided (no slippage model).
    """

    def __init__(self, config: Dict):
        acct_cfg = config.get("account", {})
        self.cash:          float            = float(acct_cfg.get("balance", 200.0))
        self.mode:          str              = config.get("mode", "backtest")
        # Multi-symbol: dict of open positions keyed by symbol.
        # `open_position` property returns the single position for backwards compat.
        self.open_positions: Dict[str, Dict]  = {}
        self.daily_pnl:     float            = 0.0
        broker_cfg = config.get("broker", {}) or {}
        self.slippage_pct: float = float(broker_cfg.get("slippage_pct", 0.0)) / 100.0

    @property
    def open_position(self) -> Optional[Dict]:
        """Backwards-compat: return the first open position, or None."""
        if not self.open_positions:
            return None
        return next(iter(self.open_positions.values()))

    @open_position.setter
    def open_position(self, val):
        """Backwards-compat: setting open_position to None clears all."""
        if val is None:
            self.open_positions.clear()
        else:
            self.open_positions[val.get("symbol", "_")] = val

    def has_position(self, symbol: str) -> bool:
        """Check if a specific symbol has an open position."""
        return symbol in self.open_positions

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get the open position for a specific symbol, or None."""
        return self.open_positions.get(symbol)

    # ------------------------------------------------------------------
    # BaseBroker interface
    # ------------------------------------------------------------------

    def submit_order(self, position: Dict) -> Dict:
        """Simulate a market-buy fill at position['entry_price'].

        Deducts position_cost from cash and stores the open position.
        Returns an order record.
        """
        symbol = position["symbol"]
        if symbol in self.open_positions:
            raise RuntimeError(
                f"Cannot open new position for {symbol} — "
                f"already holding {symbol}"
            )

        bar_price    = float(position["entry_price"])
        # Apply slippage: paying MORE than the bar price on entry
        fill_price   = bar_price * (1.0 + self.slippage_pct)
        # Recompute cost from the actual fill price (not the position_cost field
        # which was computed at the bar price by the risk manager)
        shares       = float(position["shares"])
        cost         = shares * fill_price
        order_id     = str(uuid.uuid4())
        entry_dt     = position.get("entry_time") or datetime.now(_ET)

        self.cash -= cost

        order = {
            "order_id":    order_id,
            "symbol":      position["symbol"],
            "shares":      position["shares"],
            "entry_price": fill_price,
            "stop_loss":   position["stop_loss"],
            "target":      position["target"],
            "position_cost": cost,
            "entry_time":  entry_dt,
            "mode":        self.mode,
        }

        self.open_positions[symbol] = order.copy()
        logger.info(
            "ORDER FILLED | %s  %s shares @ $%.2f  cost=$%.2f  id=%s",
            order["symbol"], order["shares"], order["entry_price"], cost, order_id,
        )
        return order

    def close_position(self, position: Dict, exit_price: float, reason: str) -> Dict:
        """Simulate closing an open position at exit_price.

        Looks up the position by symbol from the caller-provided ``position``
        dict (which carries the symbol key from the original submit_order).
        Adds sale proceeds to cash, calculates P&L, returns a trade record.
        """
        symbol = position.get("symbol") or (
            self.open_position["symbol"] if self.open_position else None
        )
        stored = self.open_positions.get(symbol) if symbol else None
        if stored is None:
            raise RuntimeError(
                f"close_position called with no open position for {symbol}"
            )

        shares     = float(stored["shares"])
        entry_price = float(stored["entry_price"])
        cost        = float(stored["position_cost"])
        entry_time  = stored["entry_time"]
        order_id    = stored["order_id"]

        exit_time = position.get("exit_time") or datetime.now(_ET)

        # Apply slippage: receiving LESS than the bar price on exit.
        # Stop-loss exits don't get extra slippage (the stop trigger price IS
        # the worst-case fill in our model), but target/eod exits do.
        bar_exit_price = float(exit_price)
        if reason in ("target_hit", "eod_exit"):
            fill_exit = bar_exit_price * (1.0 - self.slippage_pct)
        else:
            fill_exit = bar_exit_price

        proceeds   = shares * fill_exit
        pnl        = proceeds - cost
        pnl_pct    = (pnl / cost) * 100.0 if cost else 0.0

        # Ensure both datetimes are comparable
        if hasattr(entry_time, "tzinfo") and entry_time.tzinfo is None:
            entry_time = _ET.localize(entry_time)
        if hasattr(exit_time, "tzinfo") and exit_time.tzinfo is None:
            exit_time = _ET.localize(exit_time)

        duration = int((exit_time - entry_time).total_seconds() / 60)

        self.cash       += proceeds
        self.daily_pnl  += pnl
        del self.open_positions[symbol]

        trade = {
            "order_id":         order_id,
            "symbol":           symbol,
            "shares":           shares,
            "entry_price":      round(entry_price, 4),
            "exit_price":       round(fill_exit, 4),
            "pnl":              round(pnl, 4),
            "pnl_pct":          round(pnl_pct, 4),
            "reason":           reason,
            "entry_time":       entry_time,
            "exit_time":        exit_time,
            "duration_minutes": duration,
            "mode":             self.mode,
        }

        logger.info(
            "POSITION CLOSED | %s  %s shares  entry=$%.2f  exit=$%.2f  "
            "pnl=$%.2f (%.2f%%)  reason=%s",
            symbol, shares, entry_price, exit_price, pnl, pnl_pct, reason,
        )
        return trade

    def get_account_balance(self) -> float:
        """Return current cash balance."""
        return self.cash

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def reset_daily(self) -> None:
        """Reset daily P&L counter. Call at the start of each new trading day."""
        self.daily_pnl = 0.0
        logger.debug("Daily P&L reset")
