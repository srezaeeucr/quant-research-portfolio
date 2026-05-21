import logging
import uuid
from datetime import datetime
from typing import Dict, Optional

import pytz

try:
    import alpaca_trade_api as tradeapi
except ImportError:
    tradeapi = None  # type: ignore

from src.broker.base import BaseBroker

logger = logging.getLogger(__name__)

_ET = pytz.timezone("America/New_York")


class AlpacaBroker(BaseBroker):
    """Live / paper broker that routes orders through the Alpaca Markets API.

    Reads credentials from environment (loaded by the caller via dotenv):
        ALPACA_API_KEY
        ALPACA_SECRET_KEY
        ALPACA_BASE_URL  (defaults to paper endpoint if unset)

    All orders are market orders with time_in_force='day'.
    Returns dicts whose keys match SimulatorBroker so the rest of the
    engine (TradeLogger, Notifier, RiskManager) works without modification.
    """

    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL  = "https://api.alpaca.markets"

    def __init__(self, config: Dict):
        if tradeapi is None:
            raise ImportError(
                "alpaca-trade-api is not installed. "
                "Run: pip install alpaca-trade-api"
            )

        import os
        api_key    = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        base_url   = os.environ.get("ALPACA_BASE_URL", self.PAPER_URL)

        if not api_key or not secret_key:
            raise EnvironmentError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env"
            )

        self.mode:           str             = config.get("mode", "paper")
        self.open_positions: Dict[str, Dict] = {}   # keyed by symbol
        self.open_position:  Optional[Dict]  = None  # backwards compat
        self.daily_pnl:      float           = 0.0

        self._api = tradeapi.REST(
            key_id=api_key,
            secret_key=secret_key,
            base_url=base_url,
        )
        logger.info("AlpacaBroker connected  mode=%s  url=%s", self.mode, base_url)

    # ------------------------------------------------------------------
    # BaseBroker interface
    # ------------------------------------------------------------------

    def submit_order(self, position: Dict) -> Dict:
        """Submit a market buy order via Alpaca.

        Parameters
        ----------
        position : Dict
            Must include: symbol, shares, entry_price, stop_loss, target,
            position_cost, entry_time.

        Returns
        -------
        Dict  — same keys as SimulatorBroker.submit_order()
        """
        symbol = position["symbol"]
        if symbol in self.open_positions:
            raise RuntimeError(
                f"Cannot open new position for {symbol} — "
                f"already holding {symbol}"
            )
        shares = float(position["shares"])

        logger.info(
            "SUBMIT ORDER | %s  %.2f shares @ ~$%.2f",
            symbol, shares, position["entry_price"],
        )

        order = self._api.submit_order(
            symbol=symbol,
            qty=shares,
            side="buy",
            type="market",
            time_in_force="day",
        )

        filled_price = float(order.filled_avg_price or position["entry_price"])
        order_id     = str(order.id)
        entry_time   = position.get("entry_time") or datetime.now(_ET)

        logger.info(
            "ORDER FILLED | %s  %.2f shares @ $%.2f  id=%s",
            symbol, shares, filled_price, order_id,
        )

        record = {
            "order_id":      order_id,
            "symbol":        symbol,
            "shares":        shares,
            "entry_price":   filled_price,
            "stop_loss":     position["stop_loss"],
            "target":        position["target"],
            "position_cost": shares * filled_price,
            "entry_time":    entry_time,
            "mode":          self.mode,
        }
        self.open_positions[symbol] = record.copy()
        self.open_position = record.copy()  # backwards compat
        return record

    def close_position(self, position: Dict, exit_price: float, reason: str) -> Dict:
        """Submit a market sell order to close the open position.

        Parameters
        ----------
        position : Dict  — the open position record (from submit_order)
        exit_price : float  — used for P&L if fill price unavailable
        reason : str  — e.g. 'stop_loss', 'target', 'eod_exit'

        Returns
        -------
        Dict  — same keys as SimulatorBroker.close_position()
        """
        symbol = position.get("symbol")
        if symbol and symbol in self.open_positions:
            pos = self.open_positions[symbol]
        elif self.open_position is not None:
            pos = self.open_position
            symbol = pos["symbol"]
        else:
            raise RuntimeError("close_position called with no open position")

        shares      = float(pos["shares"])
        entry_price = float(pos["entry_price"])
        cost        = float(pos["position_cost"])
        entry_time  = pos["entry_time"]
        order_id    = pos["order_id"]

        logger.info(
            "CLOSE ORDER | %s  %.2f shares  reason=%s", symbol, shares, reason
        )

        order = self._api.submit_order(
            symbol=symbol,
            qty=shares,
            side="sell",
            type="market",
            time_in_force="day",
        )

        filled_price = float(order.filled_avg_price or exit_price)
        exit_time    = position.get("exit_time") or datetime.now(_ET)

        proceeds = shares * filled_price
        pnl      = proceeds - cost
        pnl_pct  = (pnl / cost) * 100.0 if cost else 0.0

        # Ensure both datetimes are comparable
        if hasattr(entry_time, "tzinfo") and entry_time.tzinfo is None:
            entry_time = _ET.localize(entry_time)
        if hasattr(exit_time, "tzinfo") and exit_time.tzinfo is None:
            exit_time = _ET.localize(exit_time)

        duration = int((exit_time - entry_time).total_seconds() / 60)

        self.daily_pnl += pnl
        self.open_positions.pop(symbol, None)
        self.open_position = None  # backwards compat

        trade = {
            "order_id":         order_id,
            "symbol":           symbol,
            "shares":           shares,
            "entry_price":      round(entry_price, 4),
            "exit_price":       round(filled_price, 4),
            "pnl":              round(pnl, 4),
            "pnl_pct":          round(pnl_pct, 4),
            "reason":           reason,
            "entry_time":       entry_time,
            "exit_time":        exit_time,
            "duration_minutes": duration,
            "mode":             self.mode,
        }

        logger.info(
            "POSITION CLOSED | %s  %.2f shares  entry=$%.2f  exit=$%.2f  "
            "pnl=$%.2f (%.2f%%)  reason=%s",
            symbol, shares, entry_price, filled_price, pnl, pnl_pct, reason,
        )
        return trade

    def get_account_balance(self) -> float:
        """Return current cash balance from the Alpaca account."""
        account = self._api.get_account()
        balance = float(account.cash)
        logger.debug("Account cash balance: $%.2f", balance)
        return balance

    # ------------------------------------------------------------------
    # Session management (mirrors SimulatorBroker interface)
    # ------------------------------------------------------------------

    def reset_daily(self) -> None:
        """Reset daily P&L counter. Called at the start of each trading day."""
        self.daily_pnl = 0.0
        self.open_positions = {}
        self.open_position = None
        logger.debug("Daily P&L reset")
