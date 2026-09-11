"""Daily closes for the regime advisor, from data already on disk.
No network. Long bars fixture first (2010+), chain underlying (2017+) second.
Chain path comes from options.data.chain_path — the binding filename
convention lives there, never duplicated here."""
from __future__ import annotations
import os
import pandas as pd
from ..options.data import chain_path

BARS = "fixtures/bars_etf_universe_2010_2026.parquet"

def closes_for(symbol: str, bars_path: str = BARS) -> pd.Series:
    symbol = symbol.upper()
    if os.path.exists(bars_path):
        bars = pd.read_parquet(bars_path)
        if isinstance(bars.columns, pd.MultiIndex) and symbol in bars.columns.get_level_values(0):
            s = bars[(symbol, "Close")].dropna()
            s.index = pd.to_datetime(s.index)
            s = s.sort_index()
            # de-dup dates to match the chain branch's groupby("date").first();
            # a duplicate date would shift every rolling window by one bar.
            return s[~s.index.duplicated(keep="last")].rename(symbol)
    chain = chain_path(symbol)
    if os.path.exists(chain):
        df = pd.read_parquet(chain, columns=["date", "underlying"])
        s = df.groupby("date")["underlying"].first()
        s.index = pd.to_datetime(s.index)
        return s.sort_index().rename(symbol)
    raise FileNotFoundError(f"no daily closes on disk for {symbol}")
