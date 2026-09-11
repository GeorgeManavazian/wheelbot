"""The published risk-free curve. Data-only: parses a saved FRED CSV.

WHY THIS EXISTS. The purchased option store (ThetaData EOD) ships no rate column,
but our solver turns a price into a vol with one. The store spans 2021-08 -> 2026-08,
across which the 1-month bill went from ~0% to ~5%. Solving that whole span at a
single rate does not add noise, it TILTS: every ticker's later observations read
systematically low against their own past, which is a fake ranking signal shared by
the entire universe. So the rate has to be joined per date.

Tenor is DGS1MO -- the wheel sells 7-11 DTE puts, so the 1-month constant-maturity
Treasury is the closest published point.
"""
from __future__ import annotations

import io

import pandas as pd


def parse_fred_csv(text: str) -> pd.Series:
    """FRED's two-column CSV -> date -> decimal rate.

    Two conversions, both load-bearing:
      - percent to decimal (FRED's 4.32 means 4.32%),
      - '.' (FRED's not-published marker) DROPPED, never coerced to 0.0. An
        absent date is carried forward from the last published rate at join
        time; a zero would be used as a real rate and silently mis-solve the day.
    """
    df = pd.read_csv(io.StringIO(text))
    date_col, value_col = df.columns[0], df.columns[1]
    values = pd.to_numeric(df[value_col], errors="coerce")
    out = pd.Series(values.values / 100.0,
                    index=pd.to_datetime(df[date_col])).dropna()
    out.index.name = None
    return out.sort_index()


def load_rate_series(path) -> pd.Series:
    """The persisted FRED series. Same parse as the wire format."""
    with open(path) as f:
        return parse_fred_csv(f.read())
