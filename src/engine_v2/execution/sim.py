"""Execution simulator: market crosses full spread + slippage,
limit fills at midprice iff price sits inside spread (else no fill)."""
from __future__ import annotations
from dataclasses import dataclass
from math import sqrt

@dataclass
class Order:
    ticker: str
    side: str        # "buy" | "sell"
    kind: str        # "market" | "limit"
    qty: float
    limit_price: float | None = None
    note: str = ""

@dataclass
class Fill:
    ticker: str
    qty: float
    price: float
    slippage_bps: float
    note: str = ""

def bid_ask(close: float, half_spread_bps: float) -> tuple[float, float]:
    """Bracket the close by half the bid-ask spread on each side. A market order
    then crosses the full spread (Chan's rule): buy pays the ask, sell hits the bid.
    Daily bars carry no quotes, so the spread is an assumption, not observed."""
    h = half_spread_bps / 1e4
    return close * (1 - h), close * (1 + h)


def slippage_bps(order_notional: float, adv: float, a: float, b: float) -> float:
    if adv <= 0:
        return a
    return a + b * sqrt(max(order_notional, 0.0) / adv)

def fill_order(order: Order, bid: float, ask: float, adv: float,
               slippage_a: float = 1.0, slippage_b: float = 5.0) -> Fill | None:
    mid = 0.5 * (bid + ask)
    if order.kind == "limit":
        if order.limit_price is None:
            return None
        if bid <= order.limit_price <= ask:
            return Fill(order.ticker, order.qty, mid, 0.0, note="limit_mid")
        return None
    # market
    ref_price = ask if order.side == "buy" else bid
    slip = slippage_bps(order.qty * ref_price, adv, slippage_a, slippage_b) / 1e4
    price = ref_price * (1 + slip) if order.side == "buy" else ref_price * (1 - slip)
    return Fill(order.ticker, order.qty, price, slip * 1e4, note="market_full_spread")
