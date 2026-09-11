import math
import pytest
from src.engine_v2.execution.sim import Order, Fill, fill_order, slippage_bps

def test_market_buy_crosses_full_spread():
    o = Order("SPY", "buy", "market", qty=100, limit_price=None)
    f = fill_order(o, bid=100.00, ask=100.10, adv=1e9)
    assert isinstance(f, Fill)
    assert f.price >= 100.10  # ask + slippage

def test_market_sell_crosses_full_spread():
    o = Order("SPY", "sell", "market", qty=100, limit_price=None)
    f = fill_order(o, bid=100.00, ask=100.10, adv=1e9)
    assert f.price <= 100.00  # bid minus slippage

def test_limit_buy_inside_spread_fills_at_midprice():
    o = Order("SPY", "buy", "limit", qty=100, limit_price=100.05)
    f = fill_order(o, bid=100.00, ask=100.10, adv=1e9)
    assert f.price == pytest.approx(100.05)

def test_limit_buy_below_bid_no_fill():
    o = Order("SPY", "buy", "limit", qty=100, limit_price=99.90)
    f = fill_order(o, bid=100.00, ask=100.10, adv=1e9)
    assert f is None

def test_limit_sell_above_ask_no_fill():
    o = Order("SPY", "sell", "limit", qty=100, limit_price=100.20)
    f = fill_order(o, bid=100.00, ask=100.10, adv=1e9)
    assert f is None

def test_slippage_bps_monotonic_in_size():
    small = slippage_bps(1_000, 1e9, a=1.0, b=5.0)
    big   = slippage_bps(1_000_000, 1e9, a=1.0, b=5.0)
    assert big > small

def test_slippage_bps_formula():
    v = slippage_bps(1_000_000, 1e9, a=1.0, b=5.0)
    expected = 1.0 + 5.0 * math.sqrt(1_000_000 / 1e9)
    assert v == pytest.approx(expected)
