"""Pluggable bar source. Workbench code depends on this Protocol, never on a
concrete loader, so an intraday source (ThetaData / Schwab) drops in later.
The default source is the real 20-ETF daily universe; the small fixture is the
fast unit-test bed only."""
from __future__ import annotations
from typing import Protocol
import pandas as pd
from .loader import load_bars

FIXTURE_PATH = "fixtures/bars_2007_2010_small.parquet"
UNIVERSE_PATH = "fixtures/bars_etf_universe_2010_2026.parquet"
INTRADAY_PATH = "data/intraday/intraday_1m.parquet"
INTRADAY_FIXTURE_PATH = "fixtures/intraday_1m_small.parquet"

class DataSource(Protocol):
    def load(self, tickers, start, end) -> pd.DataFrame: ...
    def available_tickers(self) -> list[str]: ...
    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]: ...

class ParquetSource:
    """DataSource backed by a checked-in parquet of MultiIndex (ticker, field) bars."""
    def __init__(self, path: str = UNIVERSE_PATH):
        self.path = path
        self._df = pd.read_parquet(path)

    def load(self, tickers, start, end) -> pd.DataFrame:
        return load_bars(list(tickers), start, end, source="parquet", path=self.path)

    def available_tickers(self) -> list[str]:
        return sorted(set(self._df.columns.get_level_values(0)))

    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self._df.index.min(), self._df.index.max()

def default_source() -> ParquetSource:
    """The real 20-ETF 2010–2026 universe — what the dashboard runs on."""
    return ParquetSource(UNIVERSE_PATH)

def intraday_source(path: str = INTRADAY_PATH) -> ParquetSource:
    """1-minute SPY/QQQ bars (Databento, RTH, tz-naive ET). Requires the cache
    built by scripts/build_intraday_cache.py; the committed fixture path is for
    tests."""
    return ParquetSource(path)
