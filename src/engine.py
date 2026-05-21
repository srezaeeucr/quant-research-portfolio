"""
TradingEngine — orchestrates all components for backtest (and future paper/live) runs.
"""
import logging
import math
from datetime import date, timedelta
from typing import Dict, List, Optional

import pandas as pd
import pytz
import yfinance as yf

from src.data.fetcher import DataFetcher
from src.data.universe import Universe
from src.strategy.base import BaseStrategy
from src.strategy.orb import ORBStrategy
from src.strategy.vwap_reversion import VWAPReversionStrategy
from src.strategy.gap_fill import GapFillStrategy
from src.strategy.ema_crossover import EMACrossoverStrategy
from src.strategy.momentum import MomentumStrategy
from src.strategy.macd_crossover import MACDCrossoverStrategy
from src.strategy.rsi_reversion import RSIReversionStrategy
from src.strategy.vwap_bands import VWAPBandsStrategy
from src.strategy.bollinger_reversal import BollingerReversalStrategy
from src.strategy.donchian_breakout import DonchianBreakoutStrategy
from src.strategy.stochastic_crossover import StochasticCrossoverStrategy
from src.strategy.inside_bar_breakout import InsideBarBreakoutStrategy
from src.strategy.ensemble import EnsembleStrategy
from src.strategy.hybrid_filter import HybridFilterStrategy
from src.risk.manager import RiskManager
from src.broker.simulator import SimulatorBroker
from src.broker.alpaca import AlpacaBroker
from src.logger.trade_log import TradeLogger
from src.notifications.notifier import Notifier

from datetime import time as _time

logger = logging.getLogger(__name__)

_ET          = pytz.timezone("America/New_York")
_OR_BAR_START = _time(9, 30)
_OR_BAR_END   = _time(9, 44)

_STRATEGY_MAP = {
    "orb":             ORBStrategy,
    "vwap_reversion":  VWAPReversionStrategy,
    "gap_fill":        GapFillStrategy,
    "ema_crossover":   EMACrossoverStrategy,
    "momentum":        MomentumStrategy,
    "macd_crossover":  MACDCrossoverStrategy,
    "rsi_reversion":   RSIReversionStrategy,
    "vwap_bands":      VWAPBandsStrategy,
    "bollinger_reversal":     BollingerReversalStrategy,
    "donchian_breakout":      DonchianBreakoutStrategy,
    "stochastic_crossover":   StochasticCrossoverStrategy,
    "inside_bar_breakout":    InsideBarBreakoutStrategy,
    "ensemble":               EnsembleStrategy,
    "hybrid_filter":          HybridFilterStrategy,
}


def _build_strategy(config: Dict) -> BaseStrategy:
    """Instantiate the strategy named in config.

    Checks (in order):
      1. config['active_strategy']          (new top-level key)
      2. config['strategy']['active_strategy']  (legacy path)
    """
    name = (
        config.get("active_strategy") or
        config.get("strategy", {}).get("active_strategy", "orb")
    ).lower()
    cls = _STRATEGY_MAP.get(name)
    if cls is None:
        logger.warning("Unknown strategy '%s' — falling back to ORB", name)
        cls = ORBStrategy
    return cls(config)


class TradingEngine:
    """Drives the full trade cycle for backtest mode.

    Components wired together:
        DataFetcher → Strategy → RiskManager
        → SimulatorBroker → TradeLogger → Notifier
    """

    def __init__(self, config: Dict):
        self.config  = config
        self.mode    = config.get("mode", "backtest")

        acct_cfg          = config.get("account", {})
        self._start_cash  = float(acct_cfg.get("balance", 200.0))
        self._max_pos_pct = float(acct_cfg.get("max_position_pct", 0.95))

        self.universe = Universe(config)
        self.fetcher  = DataFetcher(config)
        self.strategy = _build_strategy(config)
        self.risk_mgr = RiskManager(config)
        if self.mode in ("paper", "live"):
            self.broker = AlpacaBroker(config)
        else:
            self.broker = SimulatorBroker(config)
        self.tlog     = TradeLogger()
        notifier_cfg = config
        if config.get("mode", "backtest") == "backtest":
            import copy
            notifier_cfg = copy.deepcopy(config)
            notifier_cfg.setdefault("notifications", {}).setdefault("telegram", {})["enabled"] = False
        self.notifier = Notifier(notifier_cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_backtest(self, start_date: str, end_date: str,
                     symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Run a backtest over [start_date, end_date] for each symbol.

        Returns
        -------
        DataFrame of all logged trade records.
        """
        symbols = symbols or self.universe.get_symbols()

        self.tlog.clear()

        print(f"\n{'='*62}")
        print(f"  BACKTEST  {start_date} → {end_date}")
        print(f"  Symbols  : {symbols}")
        print(f"  Balance  : ${self._start_cash:.2f}")
        print(f"  Mode     : {self.mode.upper()}")
        print(f"{'='*62}")

        starting_balance = self.broker.cash
        peak_balance     = starting_balance
        max_drawdown_pct = 0.0

        # ── Filters (VIX + max_trades_per_day) ───────────────────────────
        filters_cfg  = self.config.get("filters", {})
        strategy_cfg = self.config.get("strategy", {})   # legacy compat
        vix_threshold: Optional[float] = None
        _vt = filters_cfg.get("vix_threshold", strategy_cfg.get("vix_threshold"))
        if _vt is not None:
            vix_threshold = float(_vt)
        max_trades: int = int(filters_cfg.get(
            "max_trades_per_day",
            strategy_cfg.get("max_trades_per_day", 9999)
        ))

        # VIX data — fetched once for the whole run (market-wide)
        vix_data: Dict = {}
        if vix_threshold is not None:
            print(f"  Fetching VIX data for filter (threshold={vix_threshold})…")
            vix_data = self._fetch_vix_data(start_date, end_date)
            print(f"  VIX data available for {len(vix_data)} dates.")

        for symbol in symbols:
            print(f"\n  Fetching 1-min bars for {symbol}…")
            full_df = self._fetch_all_bars(symbol, start_date, end_date)

            if full_df.empty:
                print(f"\n  ⚠  No 1-min data returned for {symbol}.")
                print(f"     yfinance free tier only keeps 1-min bars for the last "
                      f"~30 calendar days.")
                print(f"     Range {start_date} → {end_date} is beyond that window.")
                print(f"     To backtest older data, swap DataFetcher for a paid "
                      f"data source (Alpaca, Polygon, etc.).")
                continue

            days = self._split_by_day(full_df)
            print(f"  {len(days)} trading days loaded.")

            # Pre-compute regime SMA
            use_regime = bool(
                filters_cfg.get("regime_filter",
                                strategy_cfg.get("use_regime_filter", False))
            )
            sma_days = int(
                filters_cfg.get("regime_sma_days",
                                strategy_cfg.get("regime_sma_days", 20))
            )
            daily_sma: Dict = {}
            if use_regime:
                print(f"  Pre-computing {sma_days}-day SMA for regime filter…")
                daily_sma = self._compute_daily_sma(symbol, start_date, end_date, sma_days)
                print(f"  SMA available for {len(daily_sma)} dates.\n")
            else:
                print()

            # Per-symbol state carried across days
            prev_close:     Optional[float] = None
            prev_or_volume: Optional[float] = None

            for trading_date, day_df in sorted(days.items()):
                self.broker.reset_daily()
                self.strategy.reset_session()
                daily_trade_count = 0

                # Strategy-specific per-day context (use hasattr for safety)
                if use_regime and hasattr(self.strategy, "set_regime_sma"):
                    self.strategy.set_regime_sma(daily_sma.get(trading_date))
                if hasattr(self.strategy, "set_prev_close"):
                    self.strategy.set_prev_close(prev_close)
                if hasattr(self.strategy, "set_prev_or_volume"):
                    self.strategy.set_prev_or_volume(prev_or_volume)

                # VIX filter for today
                today_vix  = vix_data.get(trading_date)
                vix_blocked = (
                    vix_threshold is not None and
                    today_vix is not None and
                    float(today_vix) > vix_threshold
                )
                if vix_blocked:
                    logger.info(
                        "VIX filter: skipping entries on %s (VIX=%.1f > %.1f)",
                        trading_date, today_vix, vix_threshold,
                    )

                self.notifier.notify_heartbeat(
                    str(trading_date), self.broker.cash, self.mode
                )

                open_position    = None
                daily_loss_halted = False

                for i in range(len(day_df)):
                    bar       = day_df.iloc[i]
                    bar_price = float(bar["close"])
                    bar_dt    = bar["timestamp"]

                    # ── Daily loss limit ──────────────────────────────────
                    if self.risk_mgr.check_daily_loss_limit(
                            self.broker.daily_pnl, self.broker.cash):
                        if not daily_loss_halted:
                            self.notifier.notify_halt(
                                "Daily loss limit reached",
                                self.broker.daily_pnl,
                                self.mode,
                            )
                            daily_loss_halted = True
                            logger.info("Daily loss limit hit — halting %s", trading_date)
                        if open_position is not None:
                            open_position["exit_time"] = bar_dt
                            trade = self.broker.close_position(
                                open_position, bar_price, "eod_exit"
                            )
                            self.tlog.log_trade(trade)
                            self.notifier.notify_trade_closed(
                                trade, self.broker.cash, self.mode
                            )
                            open_position = None
                        break

                    # ── No open position: look for entry ─────────────────
                    if (open_position is None
                            and not vix_blocked
                            and daily_trade_count < max_trades):
                        signal = self.strategy.generate_signal(day_df, i)
                        if signal:
                            signal["symbol"] = symbol
                            buying_power = math.floor(
                                self.broker.cash * self._max_pos_pct * 100
                            ) / 100
                            # Pass bars up to and including this bar for ATR calc
                            position = self.risk_mgr.calculate_position(
                                signal, self.broker.cash,
                                buying_power=buying_power,
                                bars_df=day_df.iloc[:i + 1],
                            )
                            if position:
                                # Strategy-specific target override (e.g. gap_fill_target)
                                if signal.get("gap_fill_target") is not None:
                                    position["target"] = float(signal["gap_fill_target"])
                                position["entry_time"] = bar_dt
                                self.broker.submit_order(position)
                                self.notifier.notify_trade_opened(
                                    position, self.mode
                                )
                                open_position = position
                                daily_trade_count += 1
                                logger.info(
                                    "ENTRY  %s  %.2f shares @ $%.2f",
                                    trading_date, position["shares"],
                                    position["entry_price"],
                                )

                    # ── Open position: check for exit ─────────────────────
                    elif open_position is not None:
                        exit_result = self.risk_mgr.check_exit(
                            open_position, bar_price, bar_dt
                        )
                        if exit_result:
                            open_position["exit_time"] = bar_dt
                            trade = self.broker.close_position(
                                open_position,
                                exit_result["exit_price"],
                                exit_result["reason"],
                            )
                            self.tlog.log_trade(trade)
                            self.notifier.notify_trade_closed(
                                trade, self.broker.cash, self.mode
                            )
                            logger.info(
                                "EXIT   %s  reason=%s  pnl=$%.4f",
                                trading_date, exit_result["reason"], trade["pnl"],
                            )
                            open_position = None

                # ── EOD force-close ───────────────────────────────────────
                if open_position is not None:
                    last_bar = day_df.iloc[-1]
                    open_position["exit_time"] = last_bar["timestamp"]
                    trade = self.broker.close_position(
                        open_position, float(last_bar["close"]), "eod_exit"
                    )
                    self.tlog.log_trade(trade)
                    self.notifier.notify_trade_closed(
                        trade, self.broker.cash, self.mode
                    )
                    logger.info("EOD force-close %s  pnl=$%.4f", trading_date, trade["pnl"])
                    open_position = None

                # ── Per-day state for next day's strategies ───────────────
                prev_close  = float(day_df.iloc[-1]["close"])
                _or_volumes = [
                    float(row["volume"])
                    for _, row in day_df.iterrows()
                    if _OR_BAR_START
                    <= row["timestamp"].astimezone(_ET).time()
                    <= _OR_BAR_END
                ]
                if _or_volumes:
                    prev_or_volume = sum(_or_volumes)

                # ── Drawdown tracking ─────────────────────────────────────
                bal = self.broker.cash
                if bal > peak_balance:
                    peak_balance = bal
                if peak_balance > 0:
                    dd = (peak_balance - bal) / peak_balance * 100.0
                    if dd > max_drawdown_pct:
                        max_drawdown_pct = dd

                # ── Daily summary notification ────────────────────────────
                summary = self.tlog.get_daily_summary(str(trading_date))
                self.notifier.notify_daily_summary(
                    summary, self.broker.cash, self.mode
                )

                if summary["total_trades"] > 0:
                    sign = "+" if summary["total_pnl"] >= 0 else ""
                    print(
                        f"  {trading_date}  "
                        f"trades={summary['total_trades']}  "
                        f"pnl={sign}${summary['total_pnl']:.4f}  "
                        f"balance=${self.broker.cash:.2f}"
                    )

        all_trades = self.tlog.get_all_trades()
        self._print_summary(all_trades, starting_balance, max_drawdown_pct)
        return all_trades

    # ------------------------------------------------------------------
    # Grid-search API
    # ------------------------------------------------------------------

    def run_backtest_config(
        self,
        config_override: Dict,
        start_date: str,
        end_date: str,
        symbol: str,
        cached_bars: Optional[pd.DataFrame] = None,
        cached_sma: Optional[Dict] = None,
        cached_vix: Optional[Dict] = None,
    ) -> Dict:
        """Run a silent backtest with overridden parameters.

        Applies config_override on top of self.config, creates fresh
        strategy / risk_mgr / broker instances, runs the bar loop without
        printing or writing to trades.db, and returns a metrics dict.

        Parameters
        ----------
        config_override : dict with any subset of:
            stop_loss_pct, reward_risk, volume_mult, regime_filter,
            active_strategy, vix_threshold, max_trades_per_day
        cached_bars : pre-fetched all-hours DataFrame (avoids re-fetch)
        cached_sma  : pre-computed {date: sma_value} dict (avoids re-fetch)
        cached_vix  : pre-fetched {date: vix_close} dict (avoids re-fetch;
                      pass {} to disable VIX fetching even when threshold is set)

        Returns
        -------
        dict with keys:
            total_trades, win_rate, total_pnl, max_drawdown, profit_factor,
            avg_duration_min, sharpe_ratio, final_balance, trades_list
        """
        import copy, math as _math

        # ── Build merged config ───────────────────────────────────────
        cfg = copy.deepcopy(self.config)
        cfg.setdefault("risk",     {})
        cfg.setdefault("strategy", {})
        cfg.setdefault("filters",  {})

        if "stop_loss_pct" in config_override:
            cfg["risk"]["stop_loss_pct"] = float(config_override["stop_loss_pct"])
        if "reward_risk" in config_override:
            cfg["risk"]["reward_risk_ratio"] = float(config_override["reward_risk"])
        if "volume_mult" in config_override:
            cfg["strategy"]["volume_multiplier"] = float(config_override["volume_mult"])
        if "regime_filter" in config_override:
            cfg["strategy"]["use_regime_filter"] = bool(config_override["regime_filter"])
            cfg["filters"]["regime_filter"] = bool(config_override["regime_filter"])
        if "active_strategy" in config_override:
            cfg["strategy"]["active_strategy"] = config_override["active_strategy"]
            cfg["active_strategy"] = config_override["active_strategy"]
        if "vix_threshold" in config_override:
            cfg["filters"]["vix_threshold"] = float(config_override["vix_threshold"])
        if "max_trades_per_day" in config_override:
            cfg["filters"]["max_trades_per_day"] = int(config_override["max_trades_per_day"])
        if "afternoon_entries" in config_override:
            cfg["afternoon_entries"] = bool(config_override["afternoon_entries"])
        if "use_atr_stops" in config_override:
            cfg["risk"]["use_atr_stops"] = bool(config_override["use_atr_stops"])
        if "atr_period" in config_override:
            cfg["risk"]["atr_period"] = int(config_override["atr_period"])
        if "atr_multiplier" in config_override:
            cfg["risk"]["atr_multiplier"] = float(config_override["atr_multiplier"])
        if "stop_floor_pct" in config_override:
            cfg["risk"]["stop_floor_pct"] = float(config_override["stop_floor_pct"])
        if "stop_ceiling_pct" in config_override:
            cfg["risk"]["stop_ceiling_pct"] = float(config_override["stop_ceiling_pct"])
        if "max_risk_per_trade_pct" in config_override:
            cfg["risk"]["max_risk_per_trade_pct"] = float(
                config_override["max_risk_per_trade_pct"]
            )
        if "slippage_pct" in config_override:
            cfg.setdefault("broker", {})["slippage_pct"] = float(
                config_override["slippage_pct"]
            )
        if "entry_window_minutes" in config_override:
            cfg["entry_window_minutes"] = int(config_override["entry_window_minutes"])

        # ── Fresh components (do not touch self.broker / self.tlog) ───
        from src.risk.manager import RiskManager as _RM
        from src.broker.simulator import SimulatorBroker as _SB

        strategy = _build_strategy(cfg)
        risk_mgr = _RM(cfg)
        broker   = _SB(cfg)

        start_cash  = float(cfg.get("account", {}).get("balance", 200.0))
        max_pos_pct = float(cfg.get("account", {}).get("max_position_pct", 0.95))

        filters_cfg  = cfg.get("filters", {})
        strategy_cfg = cfg.get("strategy", {})
        use_regime   = bool(
            filters_cfg.get("regime_filter",
                            strategy_cfg.get("use_regime_filter", False))
        )
        sma_days = int(
            filters_cfg.get("regime_sma_days",
                            strategy_cfg.get("regime_sma_days", 20))
        )
        vix_threshold: Optional[float] = None
        _vt = filters_cfg.get("vix_threshold", strategy_cfg.get("vix_threshold"))
        if _vt is not None:
            vix_threshold = float(_vt)
        max_trades = int(filters_cfg.get(
            "max_trades_per_day",
            strategy_cfg.get("max_trades_per_day", 9999)
        ))

        # ── Data ──────────────────────────────────────────────────────
        if cached_bars is not None and not cached_bars.empty:
            full_df = cached_bars
        else:
            full_df = self.fetcher.fetch_historical(
                symbol, start_date, end_date, filter_windows=False
            )

        if full_df.empty:
            return self._empty_metrics(start_cash)

        days = self._split_by_day(full_df)

        daily_sma: Dict = {}
        if use_regime and hasattr(strategy, "set_regime_sma"):
            if cached_sma is not None:
                daily_sma = cached_sma
            else:
                daily_sma = self._compute_daily_sma(symbol, start_date, end_date, sma_days)

        # VIX data
        vix_data: Dict = {}
        if vix_threshold is not None:
            if cached_vix is not None:
                vix_data = cached_vix
            else:
                vix_data = self._fetch_vix_data(start_date, end_date)

        peak_balance     = start_cash
        max_drawdown_pct = 0.0
        trades_list: List[Dict] = []
        prev_close:     Optional[float] = None
        prev_or_volume: Optional[float] = None

        for trading_date, day_df in sorted(days.items()):
            broker.reset_daily()
            strategy.reset_session()
            daily_trade_count = 0

            if use_regime and hasattr(strategy, "set_regime_sma"):
                strategy.set_regime_sma(daily_sma.get(trading_date))
            if hasattr(strategy, "set_prev_close"):
                strategy.set_prev_close(prev_close)
            if hasattr(strategy, "set_prev_or_volume"):
                strategy.set_prev_or_volume(prev_or_volume)

            # VIX filter for today
            today_vix  = vix_data.get(trading_date)
            vix_blocked = (
                vix_threshold is not None and
                today_vix is not None and
                float(today_vix) > vix_threshold
            )

            open_position     = None
            daily_loss_halted = False

            for i in range(len(day_df)):
                bar       = day_df.iloc[i]
                bar_price = float(bar["close"])
                bar_dt    = bar["timestamp"]

                if risk_mgr.check_daily_loss_limit(broker.daily_pnl, broker.cash):
                    if not daily_loss_halted:
                        daily_loss_halted = True
                    if open_position is not None:
                        open_position["exit_time"] = bar_dt
                        trade = broker.close_position(open_position, bar_price, "eod_exit")
                        trades_list.append(trade)
                        open_position = None
                    break

                if (open_position is None
                        and not vix_blocked
                        and daily_trade_count < max_trades):
                    signal = strategy.generate_signal(day_df, i)
                    if signal:
                        signal["symbol"] = symbol
                        buying_power = _math.floor(
                            broker.cash * max_pos_pct * 100
                        ) / 100
                        position = risk_mgr.calculate_position(
                            signal, broker.cash,
                            buying_power=buying_power,
                            bars_df=day_df.iloc[:i + 1],
                        )
                        if position:
                            if signal.get("gap_fill_target") is not None:
                                position["target"] = float(signal["gap_fill_target"])
                            position["entry_time"] = bar_dt
                            broker.submit_order(position)
                            open_position = position
                            daily_trade_count += 1
                elif open_position is not None:
                    exit_result = risk_mgr.check_exit(open_position, bar_price, bar_dt)
                    if exit_result:
                        open_position["exit_time"] = bar_dt
                        trade = broker.close_position(
                            open_position,
                            exit_result["exit_price"],
                            exit_result["reason"],
                        )
                        trades_list.append(trade)
                        open_position = None

            if open_position is not None:
                last_bar = day_df.iloc[-1]
                open_position["exit_time"] = last_bar["timestamp"]
                trade = broker.close_position(
                    open_position, float(last_bar["close"]), "eod_exit"
                )
                trades_list.append(trade)
                open_position = None

            prev_close = float(day_df.iloc[-1]["close"])
            _or_vols = [
                float(r["volume"])
                for _, r in day_df.iterrows()
                if _OR_BAR_START <= r["timestamp"].astimezone(_ET).time() <= _OR_BAR_END
            ]
            if _or_vols:
                prev_or_volume = sum(_or_vols)

            bal = broker.cash
            if bal > peak_balance:
                peak_balance = bal
            if peak_balance > 0:
                dd = (peak_balance - bal) / peak_balance * 100.0
                if dd > max_drawdown_pct:
                    max_drawdown_pct = dd

        # ── Compute aggregate metrics ─────────────────────────────────
        return self._compute_metrics(trades_list, start_cash, broker.cash, max_drawdown_pct)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_metrics(start_cash: float) -> Dict:
        return {
            "total_trades": 0, "win_rate": 0.0, "total_pnl": 0.0,
            "max_drawdown": 0.0, "profit_factor": 0.0,
            "avg_duration_min": 0.0, "sharpe_ratio": 0.0,
            "final_balance": start_cash, "trades_list": [],
        }

    @staticmethod
    def _compute_metrics(trades_list: List[Dict], start_cash: float,
                         final_cash: float, max_drawdown_pct: float) -> Dict:
        import math as _math
        if not trades_list:
            return TradingEngine._empty_metrics(start_cash)

        pnls      = [float(t["pnl"]) for t in trades_list]
        total     = len(pnls)
        wins      = sum(1 for p in pnls if p > 0)
        win_rate  = wins / total * 100.0
        total_pnl = sum(pnls)
        gross_win = sum(p for p in pnls if p > 0)
        gross_los = abs(sum(p for p in pnls if p <= 0))
        pf        = gross_win / gross_los if gross_los > 0 else float("inf")
        avg_dur   = sum(t.get("duration_minutes", 0) for t in trades_list) / total

        # Sharpe: group P&L by calendar date, annualise
        daily_pnl: Dict = {}
        for t in trades_list:
            et_str = str(t.get("entry_time", ""))[:10]
            daily_pnl[et_str] = daily_pnl.get(et_str, 0.0) + float(t["pnl"])

        dpnls  = list(daily_pnl.values())
        if len(dpnls) >= 2:
            mu  = sum(dpnls) / len(dpnls)
            var = sum((x - mu) ** 2 for x in dpnls) / len(dpnls)
            std = _math.sqrt(var) if var > 0 else 0.0
            sharpe = (mu / std * _math.sqrt(252)) if std > 0 else 0.0
        else:
            sharpe = 0.0

        return {
            "total_trades":    total,
            "win_rate":        round(win_rate, 4),
            "total_pnl":       round(total_pnl, 6),
            "max_drawdown":    round(max_drawdown_pct, 4),
            "profit_factor":   round(pf, 4) if pf != float("inf") else 9999.0,
            "avg_duration_min": round(avg_dur, 2),
            "sharpe_ratio":    round(sharpe, 4),
            "final_balance":   round(final_cash, 4),
            "trades_list":     trades_list,
        }

    def _fetch_all_bars(self, symbol: str, start_date: str,
                        end_date: str) -> pd.DataFrame:
        """Fetch ALL intraday 1-min bars without window filter.

        Uses Alpaca if ALPACA_API_KEY is set, yfinance otherwise.
        Returns all hours — the exit checker needs mid-day bars to detect
        stop-loss and target hits that occur outside trading windows.
        """
        return self.fetcher.fetch_historical(
            symbol, start_date, end_date, filter_windows=False
        )

    def _compute_daily_sma(self, symbol: str, start_date: str,
                           end_date: str, sma_days: int) -> Dict:
        """Return {date: sma_value} for each trading day in range.

        Uses yfinance daily bars (no 30-day limit).  SMA is computed through
        D-1 (shift(1)) so there is zero lookahead bias.
        """
        fetch_start = (
            date.fromisoformat(start_date) - timedelta(days=sma_days * 3)
        ).isoformat()
        raw = yf.Ticker(symbol).history(
            start=fetch_start, end=end_date, interval="1d", auto_adjust=True
        )
        if raw.empty:
            logger.warning("No daily bars for SMA computation: %s", symbol)
            return {}
        closes = raw["Close"]
        sma = closes.rolling(window=sma_days).mean().shift(1)
        result: Dict = {}
        for dt_key, val in sma.items():
            if pd.notna(val):
                d = dt_key.date() if hasattr(dt_key, "date") else dt_key
                result[d] = float(val)
        logger.info("Daily SMA(%d) computed for %d dates", sma_days, len(result))
        return result

    def _fetch_vix_data(self, start_date: str, end_date: str) -> Dict:
        """Return {date: vix_close} for each trading day in range.

        Uses yfinance daily bars for the ^VIX index (no 30-day limit).
        Returns an empty dict on any failure so the caller can degrade gracefully.
        """
        try:
            raw = yf.Ticker("^VIX").history(
                start=start_date, end=end_date, interval="1d", auto_adjust=True
            )
        except Exception as exc:
            logger.warning("VIX fetch failed: %s", exc)
            return {}
        if raw.empty:
            logger.warning("No VIX data returned for %s → %s", start_date, end_date)
            return {}
        result: Dict = {}
        for dt_key, val in raw["Close"].items():
            if pd.notna(val):
                d = dt_key.date() if hasattr(dt_key, "date") else dt_key
                result[d] = float(val)
        logger.info("VIX data loaded for %d dates", len(result))
        return result

    def _split_by_day(self, df: pd.DataFrame) -> Dict:
        """Group all-hours DataFrame by ET calendar date."""
        df = df.copy()
        df["_date"] = df["timestamp"].dt.tz_convert(_ET).dt.date
        return {
            d: grp.drop(columns="_date").reset_index(drop=True)
            for d, grp in df.groupby("_date")
        }

    def _print_summary(self, df: pd.DataFrame, starting_balance: float,
                        max_drawdown_pct: float) -> None:
        print(f"\n{'='*62}")
        print("  BACKTEST SUMMARY")
        print(f"{'='*62}")

        if df.empty:
            print("  No trades executed in this period.\n")
            final = self.broker.cash
            print(f"  Starting balance : ${starting_balance:.2f}")
            print(f"  Final balance    : ${final:.2f}")
            print(f"  Total P&L        : ${final - starting_balance:+.4f}")
            print(f"  Max drawdown     : {max_drawdown_pct:.2f}%")
            print()
            return

        total   = len(df)
        wins    = int((df["pnl"] > 0).sum())
        losses  = total - wins
        total_pnl = float(df["pnl"].sum())
        win_rate  = wins / total * 100 if total else 0.0
        final_bal = self.broker.cash
        avg_win   = float(df.loc[df["pnl"] > 0, "pnl"].mean()) if wins  else 0.0
        avg_loss  = float(df.loc[df["pnl"] <= 0, "pnl"].mean()) if losses else 0.0
        avg_dur   = float(df["duration_minutes"].mean())

        reason_counts = df["reason"].value_counts().to_dict()

        print(f"  Total trades     : {total}")
        print(f"  Wins / Losses    : {wins} / {losses}  "
              f"(win rate {win_rate:.1f}%)")
        print(f"  Avg win          : ${avg_win:+.4f}")
        print(f"  Avg loss         : ${avg_loss:+.4f}")
        print(f"  Avg duration     : {avg_dur:.0f} min")
        print(f"  Exit breakdown   : "
              + "  ".join(f"{k}={v}" for k, v in sorted(reason_counts.items())))
        print(f"  ─────────────────────────────────────────────────────")
        print(f"  Starting balance : ${starting_balance:.2f}")
        print(f"  Final balance    : ${final_bal:.2f}")
        print(f"  Total P&L        : ${total_pnl:+.4f}")
        print(f"  Max drawdown     : {max_drawdown_pct:.2f}%")
        print()
