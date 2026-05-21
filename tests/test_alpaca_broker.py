"""
Tests for AlpacaBroker — all Alpaca API calls are mocked.

Because alpaca-trade-api is not installed in the test venv, the module is
injected into sys.modules as a MagicMock before the broker is imported.
"""
import os
import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import pytz

# ---------------------------------------------------------------------------
# Inject fake alpaca_trade_api into sys.modules BEFORE importing the broker
# ---------------------------------------------------------------------------
_mock_alpaca_module = MagicMock()
sys.modules.setdefault("alpaca_trade_api", _mock_alpaca_module)

# Now the import in src/broker/alpaca.py will succeed, with tradeapi = MagicMock()
from src.broker.alpaca import AlpacaBroker  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ET = pytz.timezone("America/New_York")

BASE_CONFIG = {
    "mode": "paper",
    "account": {"balance": 200.0, "max_position_pct": 0.95},
}

POSITION = {
    "symbol":        "AAPL",
    "shares":        0.25,
    "entry_price":   180.00,
    "stop_loss":     179.10,
    "target":        182.25,
    "position_cost": 45.00,
    "entry_time":    datetime(2024, 1, 15, 9, 50, tzinfo=_ET),
}


def _make_mock_order(order_id: str, filled_price: float) -> MagicMock:
    order = MagicMock()
    order.id = order_id
    order.filled_avg_price = str(filled_price)
    return order


def _make_mock_account(cash: float = 195.50) -> MagicMock:
    account = MagicMock()
    account.id    = "abc123def456"
    account.cash  = str(cash)
    account.equity = str(cash + 10)
    account.pattern_day_trader = False
    return account


def _make_broker(config=None) -> "tuple[AlpacaBroker, MagicMock]":
    """Return an AlpacaBroker with a mocked REST client."""
    if config is None:
        config = BASE_CONFIG

    env_patch = {
        "ALPACA_API_KEY":    "TESTKEY1234",
        "ALPACA_SECRET_KEY": "TESTSECRET5678",
        "ALPACA_BASE_URL":   "https://paper-api.alpaca.markets",
    }
    mock_api = MagicMock()

    with patch.dict(os.environ, env_patch, clear=False):
        # Make tradeapi.REST(...)  return our mock_api
        _mock_alpaca_module.REST.return_value = mock_api
        broker = AlpacaBroker(config)

    # Swap in the mock directly so test calls land on it
    broker._api = mock_api
    return broker, mock_api


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestAlpacaBrokerInit:
    def test_raises_without_api_key(self):
        env = {"ALPACA_API_KEY": "", "ALPACA_SECRET_KEY": "secret",
               "ALPACA_BASE_URL": "https://paper-api.alpaca.markets"}
        with patch.dict(os.environ, env, clear=False):
            with pytest.raises(EnvironmentError, match="ALPACA_API_KEY"):
                AlpacaBroker(BASE_CONFIG)

    def test_raises_without_secret_key(self):
        env = {"ALPACA_API_KEY": "key", "ALPACA_SECRET_KEY": "",
               "ALPACA_BASE_URL": "https://paper-api.alpaca.markets"}
        with patch.dict(os.environ, env, clear=False):
            with pytest.raises(EnvironmentError, match="ALPACA_SECRET_KEY"):
                AlpacaBroker(BASE_CONFIG)

    def test_raises_when_library_missing(self):
        with patch("src.broker.alpaca.tradeapi", None):
            with pytest.raises(ImportError, match="alpaca-trade-api"):
                AlpacaBroker(BASE_CONFIG)

    def test_mode_set_correctly(self):
        broker, _ = _make_broker()
        assert broker.mode == "paper"

    def test_initial_state(self):
        broker, _ = _make_broker()
        assert broker.open_position is None
        assert broker.daily_pnl == 0.0


# ---------------------------------------------------------------------------
# submit_order
# ---------------------------------------------------------------------------

class TestSubmitOrder:
    def test_returns_expected_keys(self):
        broker, mock_api = _make_broker()
        mock_api.submit_order.return_value = _make_mock_order("order-001", 180.20)

        result = broker.submit_order(POSITION)

        expected_keys = {
            "order_id", "symbol", "shares", "entry_price",
            "stop_loss", "target", "position_cost", "entry_time", "mode",
        }
        assert set(result.keys()) == expected_keys

    def test_symbol_and_mode_correct(self):
        broker, mock_api = _make_broker()
        mock_api.submit_order.return_value = _make_mock_order("order-002", 180.20)

        result = broker.submit_order(POSITION)

        assert result["symbol"] == "AAPL"
        assert result["mode"]   == "paper"

    def test_filled_price_used_when_available(self):
        broker, mock_api = _make_broker()
        mock_api.submit_order.return_value = _make_mock_order("order-003", 180.25)

        result = broker.submit_order(POSITION)

        assert result["entry_price"] == 180.25

    def test_entry_price_fallback_when_fill_is_none(self):
        broker, mock_api = _make_broker()
        order = MagicMock()
        order.id = "order-004"
        order.filled_avg_price = None
        mock_api.submit_order.return_value = order

        result = broker.submit_order(POSITION)

        assert result["entry_price"] == POSITION["entry_price"]

    def test_open_position_stored(self):
        broker, mock_api = _make_broker()
        mock_api.submit_order.return_value = _make_mock_order("order-005", 180.00)

        broker.submit_order(POSITION)

        assert broker.open_position is not None
        assert broker.open_position["symbol"] == "AAPL"

    def test_raises_if_position_already_open(self):
        broker, mock_api = _make_broker()
        mock_api.submit_order.return_value = _make_mock_order("order-006", 180.00)

        broker.submit_order(POSITION)

        with pytest.raises(RuntimeError, match="already holding"):
            broker.submit_order(POSITION)

    def test_api_called_with_market_day(self):
        broker, mock_api = _make_broker()
        mock_api.submit_order.return_value = _make_mock_order("order-007", 180.00)

        broker.submit_order(POSITION)

        call_kwargs = mock_api.submit_order.call_args[1]
        assert call_kwargs["type"]          == "market"
        assert call_kwargs["time_in_force"] == "day"
        assert call_kwargs["side"]          == "buy"
        assert call_kwargs["symbol"]        == "AAPL"


# ---------------------------------------------------------------------------
# close_position
# ---------------------------------------------------------------------------

class TestClosePosition:
    def _open(self, broker: AlpacaBroker, mock_api: MagicMock,
              entry_price: float = 180.00) -> None:
        mock_api.submit_order.return_value = _make_mock_order("order-buy", entry_price)
        broker.submit_order(POSITION)

    def test_returns_expected_keys(self):
        broker, mock_api = _make_broker()
        self._open(broker, mock_api)
        mock_api.submit_order.return_value = _make_mock_order("order-sell", 182.00)

        exit_pos = {"exit_time": datetime(2024, 1, 15, 10, 15, tzinfo=_ET)}
        result = broker.close_position(exit_pos, 182.00, "target")

        expected_keys = {
            "order_id", "symbol", "shares", "entry_price", "exit_price",
            "pnl", "pnl_pct", "reason", "entry_time", "exit_time",
            "duration_minutes", "mode",
        }
        assert set(result.keys()) == expected_keys

    def test_pnl_calculated_correctly(self):
        broker, mock_api = _make_broker()
        self._open(broker, mock_api, entry_price=180.00)
        mock_api.submit_order.return_value = _make_mock_order("order-sell2", 182.00)

        exit_pos = {"exit_time": datetime(2024, 1, 15, 10, 15, tzinfo=_ET)}
        result = broker.close_position(exit_pos, 182.00, "target")

        expected_pnl = round(0.25 * 182.00 - 0.25 * 180.00, 4)
        assert result["pnl"] == pytest.approx(expected_pnl, abs=0.01)

    def test_reason_preserved(self):
        broker, mock_api = _make_broker()
        self._open(broker, mock_api)
        mock_api.submit_order.return_value = _make_mock_order("order-sell3", 179.10)

        exit_pos = {"exit_time": datetime(2024, 1, 15, 9, 55, tzinfo=_ET)}
        result = broker.close_position(exit_pos, 179.10, "stop_loss")

        assert result["reason"] == "stop_loss"

    def test_open_position_cleared(self):
        broker, mock_api = _make_broker()
        self._open(broker, mock_api)
        mock_api.submit_order.return_value = _make_mock_order("order-sell4", 182.00)

        exit_pos = {"exit_time": datetime(2024, 1, 15, 10, 15, tzinfo=_ET)}
        broker.close_position(exit_pos, 182.00, "target")

        assert broker.open_position is None

    def test_daily_pnl_updated(self):
        broker, mock_api = _make_broker()
        self._open(broker, mock_api, entry_price=180.00)
        mock_api.submit_order.return_value = _make_mock_order("order-sell5", 182.00)

        exit_pos = {"exit_time": datetime(2024, 1, 15, 10, 15, tzinfo=_ET)}
        result = broker.close_position(exit_pos, 182.00, "target")

        assert broker.daily_pnl == pytest.approx(result["pnl"], abs=0.01)

    def test_raises_with_no_open_position(self):
        broker, _ = _make_broker()
        with pytest.raises(RuntimeError, match="no open position"):
            broker.close_position({}, 182.00, "target")

    def test_filled_price_used_over_exit_price(self):
        broker, mock_api = _make_broker()
        self._open(broker, mock_api, entry_price=180.00)
        mock_api.submit_order.return_value = _make_mock_order("order-sell6", 182.50)

        exit_pos = {"exit_time": datetime(2024, 1, 15, 10, 15, tzinfo=_ET)}
        result = broker.close_position(exit_pos, 182.00, "target")

        assert result["exit_price"] == 182.50  # filled price wins


# ---------------------------------------------------------------------------
# get_account_balance
# ---------------------------------------------------------------------------

class TestGetAccountBalance:
    def test_returns_cash_from_api(self):
        broker, mock_api = _make_broker()
        mock_api.get_account.return_value = _make_mock_account(cash=193.75)

        balance = broker.get_account_balance()

        assert balance == pytest.approx(193.75)

    def test_api_called_once(self):
        broker, mock_api = _make_broker()
        mock_api.get_account.return_value = _make_mock_account()

        broker.get_account_balance()

        mock_api.get_account.assert_called_once()


# ---------------------------------------------------------------------------
# reset_daily
# ---------------------------------------------------------------------------

class TestResetDaily:
    def test_resets_pnl_to_zero(self):
        broker, _ = _make_broker()
        broker.daily_pnl = 5.50

        broker.reset_daily()

        assert broker.daily_pnl == 0.0
