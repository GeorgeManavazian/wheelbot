import pandas as pd
import pytest
from src.engine_v2.execution.sim import Order
from src.engine_v2.execution.events import (
    is_halt_day, defer_to_next_open, flag_gap, GAP_FLAG_PCT,
)

def test_halt_when_zero_volume():
    row = pd.Series({"Open": 100, "High": 100, "Low": 100, "Close": 100, "Volume": 0})
    assert is_halt_day(row) is True

def test_normal_day_not_halt():
    row = pd.Series({"Open": 100, "High": 101, "Low": 99, "Close": 100.5, "Volume": 1_000_000})
    assert is_halt_day(row) is False

def test_defer_marks_order():
    o = Order("SPY", "buy", "market", qty=100)
    d = defer_to_next_open(o)
    assert "deferred" in getattr(d, "note", "")

def test_gap_flag_true_when_over_threshold():
    is_g, pct = flag_gap(prev_close=100, open_price=106)
    assert is_g is True
    assert pct == pytest.approx(0.06)

def test_gap_flag_false_when_below_threshold():
    is_g, _ = flag_gap(100, 102)
    assert is_g is False

def test_gap_threshold_constant():
    assert GAP_FLAG_PCT == 0.05
