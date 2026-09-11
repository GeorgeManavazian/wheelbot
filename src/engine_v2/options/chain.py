"""Normalize ThetaData greeks/eod CSV into the tidy OptionsChain frame, plus the
Contract / Mark value types."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

_RIGHT = {"CALL": "C", "PUT": "P", "C": "C", "P": "P"}
COLUMNS = ["date","expiry","dte","strike","right","bid","ask","mid",
           "close","delta","iv","underlying"]

@dataclass(frozen=True)
class Contract:
    root: str
    expiry: pd.Timestamp
    strike: float
    right: str

@dataclass(frozen=True)
class Mark:
    bid: float
    ask: float
    mid: float
    # C1: per-leg quote time (ms epoch) when the source provides one; None on
    # backtest frames (no column) and legacy payloads. Trailing default so
    # positional construction and equality with 3-arg Marks stay intact.
    quote_time: float | None = None

def normalize_greeks_eod(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    date = pd.to_datetime(df["underlying_timestamp"]).dt.normalize()
    expiry = pd.to_datetime(df["expiration"]).dt.normalize()
    out = pd.DataFrame({
        "date": date,
        "expiry": expiry,
        "dte": (expiry - date).dt.days,
        "strike": df["strike"].astype(float),
        "right": df["right"].map(_RIGHT),
        "bid": df["bid"].astype(float),
        "ask": df["ask"].astype(float),
        "mid": (df["bid"].astype(float) + df["ask"].astype(float)) / 2.0,
        "close": df["close"].astype(float),
        "delta": df["delta"].astype(float),
        "iv": df["implied_vol"].astype(float),
        "underlying": df["underlying_price"].astype(float),
    })
    out = out[(out["bid"] > 0) & (out["ask"] > 0) & out["delta"].notna()]
    return out[COLUMNS].sort_values(["date","expiry","strike","right"]).reset_index(drop=True)
