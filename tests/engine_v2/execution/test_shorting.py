import pytest
import pandas as pd
from src.engine_v2.execution.sim import Order
from src.engine_v2.execution.shorting import (
    HTBRegistry, daily_borrow_fee, apply_short_guard,
)

def test_daily_borrow_fee_math():
    # $100k notional, 300 bps annual, per-day
    assert daily_borrow_fee(100_000, 300) == pytest.approx(100_000 * 0.03 / 252)

def test_htb_registry_lookup():
    reg = HTBRegistry({(pd.Timestamp("2020-03-16"), "GME"): True})
    assert reg.is_htb("GME", pd.Timestamp("2020-03-16")) is True
    assert reg.is_htb("SPY", pd.Timestamp("2020-03-16")) is False

def test_short_on_htb_blocked():
    reg = HTBRegistry({(pd.Timestamp("2020-03-16"), "GME"): True})
    o = Order("GME", "sell", "market", qty=100)
    assert apply_short_guard(o, pd.Timestamp("2020-03-16"), reg) is None

def test_short_on_easy_borrow_allowed():
    reg = HTBRegistry({})
    o = Order("SPY", "sell", "market", qty=100)
    assert apply_short_guard(o, pd.Timestamp("2020-03-16"), reg) is o

def test_buy_never_blocked_by_htb():
    reg = HTBRegistry({(pd.Timestamp("2020-03-16"), "GME"): True})
    o = Order("GME", "buy", "market", qty=100)
    assert apply_short_guard(o, pd.Timestamp("2020-03-16"), reg) is o
