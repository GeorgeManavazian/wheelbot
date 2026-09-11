"""Halt-day + gap handling. Halt = zero volume or flat OHLC; order deferred to
next open. Gaps > 5% still fill but flagged for logging."""
from __future__ import annotations
from dataclasses import replace
import pandas as pd
from .sim import Order

GAP_FLAG_PCT = 0.05

def is_halt_day(bar_row: pd.Series) -> bool:
    vol = bar_row.get("Volume", 0)
    return bool(vol == 0)

def defer_to_next_open(order: Order) -> Order:
    return replace(order, note="deferred_next_open")

def flag_gap(prev_close: float, open_price: float) -> tuple[bool, float]:
    if prev_close <= 0:
        return False, 0.0
    pct = (open_price - prev_close) / prev_close
    return abs(pct) > GAP_FLAG_PCT, pct
