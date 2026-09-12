"""Per-day regime tags: trend (bull/bear), vol (calm/high), rate (rising/falling).
Rate is a placeholder ('falling') until Schwab yield ingest is wired; safe for
2007-2020 window which was falling-rate overall."""
from __future__ import annotations
import numpy as np
import pandas as pd

REGIME_COLS = ("regime_trend", "regime_vol", "regime_rate")

def tag_regime(bars: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
    close = bars[benchmark]["Close"]
    ma200 = close.rolling(200, min_periods=1).mean()
    trend = np.where(close > ma200, "bull", "bear")
    daily_ret = close.pct_change()
    rv = daily_ret.rolling(20, min_periods=5).std() * np.sqrt(252)
    vol = np.where(rv > 0.20, "high", "calm")
    rate = np.repeat("falling", len(bars))
    return pd.DataFrame(
        {"regime_trend": trend, "regime_vol": vol, "regime_rate": rate},
        index=bars.index,
    )
