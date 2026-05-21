"""
ATR (Average True Range) — used for dynamic stop-loss sizing.

True Range for bar t:
    TR = max(
        high - low,
        abs(high - prev_close),
        abs(low  - prev_close),
    )

ATR = simple rolling mean of TR over `period` bars.
"""
from typing import Optional

import pandas as pd


def calculate_atr(bars_df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Return the most recent ATR value (in price units), or None if data
    is insufficient.

    Parameters
    ----------
    bars_df : DataFrame with columns ``high``, ``low``, ``close`` (case-sensitive)
    period  : rolling window length (default 14)

    Returns
    -------
    float ATR or None if df has fewer than ``period + 1`` valid rows
    """
    if bars_df is None or len(bars_df) < period + 1:
        return None

    required = {"high", "low", "close"}
    if not required.issubset(bars_df.columns):
        return None

    high       = bars_df["high"].astype(float)
    low        = bars_df["low"].astype(float)
    close      = bars_df["close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean().iloc[-1]
    if pd.isna(atr):
        return None
    return float(atr)
