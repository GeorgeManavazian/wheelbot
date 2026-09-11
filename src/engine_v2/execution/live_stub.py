"""Live-executor stub. Same-code-path rule: order->fill translation is a single
function; live and backtest share it. When real Schwab wiring lands, only the
market-data source changes, NOT the fill logic."""
from __future__ import annotations
from .sim import Order, Fill, fill_order

def live_fill(order: Order, bid: float, ask: float, adv: float, **kw) -> Fill | None:
    return fill_order(order, bid, ask, adv, **kw)
