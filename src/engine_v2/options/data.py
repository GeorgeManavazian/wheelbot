"""Per-ticker data discovery.

Filename convention (binding, per the API contract):

    data/options/{ticker_lower}_greeks_eod_all.parquet   # EOD greeks — always required
    data/options/{ticker_lower}_ohlc_1h_all.parquet      # hourly OHLC — optional
"""
from __future__ import annotations

import glob
import os

_EOD_SUFFIX = "_greeks_eod_all.parquet"
_INTRADAY_SUFFIX = "_ohlc_1h_all.parquet"


def available_tickers(data_dir: str = "data/options") -> list[str]:
    """Tickers with an EOD chain on disk (upper-case, sorted)."""
    paths = glob.glob(os.path.join(data_dir, f"*{_EOD_SUFFIX}"))
    return sorted(
        os.path.basename(p)[: -len(_EOD_SUFFIX)].upper() for p in paths
    )


def intraday_path(ticker: str, data_dir: str = "data/options") -> str | None:
    """Path to the hourly OHLC parquet for `ticker`, or None if absent."""
    p = os.path.join(data_dir, f"{ticker.lower()}{_INTRADAY_SUFFIX}")
    return p if os.path.exists(p) else None


def chain_path(ticker: str, data_dir: str = "data/options") -> str:
    """Path to the EOD greeks parquet for `ticker` (existence not checked)."""
    return os.path.join(data_dir, f"{ticker.lower()}{_EOD_SUFFIX}")
