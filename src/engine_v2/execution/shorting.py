"""Short-specific cost model: borrow fee + hard-to-borrow (HTB) registry.
apply_short_guard() returns None when a sell (short) hits an HTB name."""
from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd
from .sim import Order

@dataclass
class HTBRegistry:
    entries: dict = field(default_factory=dict)  # (Timestamp, ticker) -> bool

    def is_htb(self, ticker: str, date) -> bool:
        return bool(self.entries.get((pd.Timestamp(date), ticker), False))

def daily_borrow_fee(notional: float, borrow_bps: float) -> float:
    return notional * borrow_bps / 1e4 / 252

def apply_short_guard(order: Order, date, registry: HTBRegistry):
    if order.side == "sell" and registry.is_htb(order.ticker, date):
        return None
    return order
