import os
import logging
from pathlib import Path
from datetime import time
from typing import Dict, List, Optional, Tuple

import requests
import yaml
import pytz
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap: load .env and config
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env")

with open(_ROOT / "config" / "config.yaml") as _f:
    _CFG = yaml.safe_load(_f)

_ET = pytz.timezone("America/New_York")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_windows(cfg: Dict) -> List[Tuple[time, time]]:
    """Return list of (start_time, end_time) tuples from config trading_windows."""
    windows = []
    for w in cfg.get("trading_windows", []):
        start = time(*map(int, w["start"].split(":")))
        end   = time(*map(int, w["end"].split(":")))
        windows.append((start, end))
    return windows


def _filter_windows(df: pd.DataFrame, windows: List[Tuple[time, time]]) -> pd.DataFrame:
    """Keep only rows whose timestamp falls inside at least one trading window."""
    if df.empty:
        return df
    ts = df["timestamp"].dt.tz_convert(_ET)
    mask = pd.Series(False, index=df.index)
    for start, end in windows:
        mask |= ts.dt.time.between(start, end)
    return df[mask].copy()


def _standardise(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to lowercase standard names and ensure ET-aware timestamp."""
    df = df.rename(columns={c: c.lower() for c in df.columns})
    rename_map = {"datetime": "timestamp", "date": "timestamp"}
    df = df.rename(columns=rename_map)

    if "timestamp" not in df.columns:
        df = df.reset_index().rename(columns={"index": "timestamp",
                                              "Datetime": "timestamp",
                                              "Date": "timestamp"})

    # Ensure column order and required set
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df[[c for c in required if c in df.columns]]

    # Make timestamp timezone-aware in ET
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(_ET)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(_ET)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# DataFetcher
# ---------------------------------------------------------------------------

_ALPACA_DATA_URL = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"


class DataFetcher:
    """Fetches 1-minute OHLCV bars and filters them to configured trading windows.

    Data source selection (fetch_historical):
      - ALPACA_API_KEY present in environment → Alpaca Data API (years of history)
      - Key absent → yfinance fallback (~30-day 1-min window only)

    Modes
    -----
    backtest  — historical data (Alpaca or yfinance)
    paper/live — Alpaca Markets REST API
    """

    def __init__(self, cfg: Optional[Dict] = None):
        self.cfg = cfg or _CFG
        self.mode = self.cfg.get("mode", "backtest")
        self.windows = _parse_windows(self.cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_historical(self, symbol: str, start_date: str, end_date: str,
                         filter_windows: bool = True) -> pd.DataFrame:
        """Download 1-min bars for [start_date, end_date].

        Automatically selects the data source:
          - ALPACA_API_KEY set → Alpaca Data API (unlimited history)
          - key missing       → yfinance (last ~30 days only)

        Parameters
        ----------
        filter_windows : if True (default), only bars inside configured trading
                         windows are returned.  Pass False to get all-hours bars
                         (needed by TradingEngine for mid-day exit monitoring).

        Returns
        -------
        DataFrame: timestamp, open, high, low, close, volume
        """
        api_key = os.environ.get("ALPACA_API_KEY", "")
        if api_key:
            logger.info(
                "ALPACA_API_KEY found — using Alpaca data source for %s (%s → %s)",
                symbol, start_date, end_date,
            )
            return self.fetch_historical_alpaca(symbol, start_date, end_date,
                                                filter_windows=filter_windows)

        logger.info(
            "No ALPACA_API_KEY — falling back to yfinance for %s (%s → %s)",
            symbol, start_date, end_date,
        )
        return self._fetch_historical_yfinance(symbol, start_date, end_date,
                                               filter_windows=filter_windows)

    def fetch_historical_alpaca(self, symbol: str, start_date: str,
                                end_date: str,
                                filter_windows: bool = True) -> pd.DataFrame:
        """Download 1-min bars from the Alpaca Data API.

        Handles pagination automatically — keeps fetching until
        next_page_token is None.

        Parameters
        ----------
        symbol         : ticker, e.g. 'SPY'
        start_date     : 'YYYY-MM-DD'
        end_date       : 'YYYY-MM-DD'  (exclusive on Alpaca side)
        filter_windows : if True (default), only trading-window bars returned.
                         Pass False for all-hours bars (engine exit monitoring).

        Returns
        -------
        DataFrame: timestamp (ET-aware), open, high, low, close, volume
        """
        api_key    = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")

        if not api_key or not secret_key:
            raise EnvironmentError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env "
                "to use fetch_historical_alpaca()"
            )

        headers = {
            "APCA-API-KEY-ID":     api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }
        url = _ALPACA_DATA_URL.format(symbol=symbol)

        all_bars: list = []
        page_token: Optional[str] = None
        page_num = 0

        while True:
            params: Dict = {
                "timeframe":  "1Min",
                "start":      start_date,
                "end":        end_date,
                "limit":      10000,
                "adjustment": "raw",
                "feed":       "iex",   # free-tier feed (SIP requires paid plan)
            }
            if page_token:
                params["page_token"] = page_token

            resp = requests.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            bars = data.get("bars") or []
            all_bars.extend(bars)
            page_num += 1
            logger.debug("Alpaca page %d: %d bars (running total %d)",
                         page_num, len(bars), len(all_bars))

            page_token = data.get("next_page_token") or None
            if page_token is None:
                break

        logger.info(
            "Alpaca fetch complete: %d raw bars over %d page(s) for %s (%s → %s)",
            len(all_bars), page_num, symbol, start_date, end_date,
        )

        if not all_bars:
            logger.warning("Alpaca returned no bars for %s (%s → %s)",
                           symbol, start_date, end_date)
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        df = pd.DataFrame(all_bars)
        df = df.rename(columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        })
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]

        # Alpaca timestamps are UTC ISO 8601 → convert to ET
        df["timestamp"] = (
            pd.to_datetime(df["timestamp"], utc=True)
              .dt.tz_convert(_ET)
        )

        total = len(df)
        if filter_windows:
            df = _filter_windows(df, self.windows)
            logger.info("Bars after window filter: %d / %d", len(df), total)
        else:
            logger.info("Returning %d bars (all hours — no window filter)", total)

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_historical_yfinance(self, symbol: str, start_date: str,
                                   end_date: str,
                                   filter_windows: bool = True) -> pd.DataFrame:
        """Download 1-min bars from yfinance (last ~30 days only)."""
        logger.info("yfinance fetch: %s (%s → %s)", symbol, start_date, end_date)

        ticker = yf.Ticker(symbol)
        raw = ticker.history(start=start_date, end=end_date, interval="1m", auto_adjust=True)

        if raw.empty:
            logger.warning("yfinance returned no data for %s", symbol)
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        raw = raw.reset_index()
        df = _standardise(raw)

        total = len(df)
        if filter_windows:
            df = _filter_windows(df, self.windows)
            logger.info("Bars fetched: %d total | %d after window filter", total, len(df))
        else:
            logger.info("Returning %d bars (all hours — no window filter)", total)
        return df

    def fetch_live(self, symbol: str) -> pd.DataFrame:
        """Fetch the most recent completed 1-min bar.

        Strategy
        --------
        1. Try Alpaca (IEX feed) up to 3 times with exponential backoff
           (1s, 3s, 8s). Catches both empty responses AND connection errors.
        2. If Alpaca exhausted, try yfinance as fallback (last day's 1-min,
           take the most recent bar).
        3. If everything fails, return an empty DataFrame (caller skips).

        Requires ALPACA_API_KEY, ALPACA_SECRET_KEY in environment.
        """
        try:
            import alpaca_trade_api as tradeapi  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("alpaca-trade-api is required for paper/live mode") from exc

        api_key    = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        base_url   = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

        if not api_key or not secret_key:
            raise EnvironmentError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env")

        logger.info("Fetching live 1-min bar for %s via Alpaca (%s)", symbol, base_url)

        api = tradeapi.REST(api_key, secret_key, base_url, api_version="v2")
        from datetime import datetime as _dt, timedelta as _td
        import time as _time
        # Use 3 minutes ago as start so we get the most recent completed bar
        # (slightly wider than before to tolerate clock skew).
        start_ts = (_dt.now(_ET) - _td(minutes=3)).isoformat()

        # Retry policy: 3 attempts with exponential backoff.
        backoffs = [0, 1.5, 5.0]    # sleep BEFORE attempt N (0 = immediate first try)
        bars = None
        last_error: Optional[Exception] = None
        for attempt, sleep_s in enumerate(backoffs, start=1):
            if sleep_s > 0:
                _time.sleep(sleep_s)
            try:
                bars = api.get_bars(symbol, tradeapi.TimeFrame.Minute,
                                     start=start_ts, limit=1, feed='iex').df
                if not bars.empty:
                    break  # success
                logger.info("Alpaca returned empty for %s (attempt %d/%d)",
                            symbol, attempt, len(backoffs))
            except Exception as e:
                last_error = e
                logger.warning("Alpaca fetch error for %s (attempt %d/%d): %s",
                               symbol, attempt, len(backoffs), e)
                bars = None

        if bars is None or bars.empty:
            logger.warning(
                "Alpaca exhausted for %s (last_error=%s) — trying yfinance fallback",
                symbol, last_error,
            )
            df = self._fetch_live_yfinance_fallback(symbol)
            if not df.empty:
                logger.info("yfinance fallback succeeded for %s  ts=%s",
                            symbol, df["timestamp"].iloc[-1])
            else:
                logger.warning("yfinance fallback also returned no bars for %s", symbol)
            return df

        bars = bars.reset_index()
        df   = _standardise(bars)
        logger.info("Live bar fetched (Alpaca): %d row(s)  ts=%s",
                    len(df),
                    df["timestamp"].iloc[-1] if not df.empty else "N/A")
        return df

    def _fetch_live_yfinance_fallback(self, symbol: str) -> pd.DataFrame:
        """Take the most-recent 1-min bar from yfinance.

        yfinance returns the last day of 1-min data; we take the last row.
        Slower than Alpaca and rate-limited but useful when Alpaca's IEX
        feed has gaps or connection issues.
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1d", interval="1m")
            if df is None or df.empty:
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            # Take only the last row (most recent bar)
            df = df.tail(1).reset_index()
            # yfinance returns a 'Datetime' column; rename to match our schema
            if "Datetime" in df.columns:
                df = df.rename(columns={"Datetime": "timestamp"})
            elif "Date" in df.columns:
                df = df.rename(columns={"Date": "timestamp"})
            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })
            # Ensure tz-aware ET timestamps
            ts = pd.to_datetime(df["timestamp"])
            if ts.dt.tz is None:
                ts = ts.dt.tz_localize("UTC")
            df["timestamp"] = ts.dt.tz_convert(_ET)
            return df[["timestamp", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.warning("yfinance fallback error for %s: %s", symbol, e)
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
