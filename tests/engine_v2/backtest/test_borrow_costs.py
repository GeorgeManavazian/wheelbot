"""A short position must pay a daily borrow fee.

daily_borrow_fee (execution/shorting.py) existed but was never called anywhere in
the engine, so shorts were financed for free. This is the first plugin that goes
short (Oxford Gap Pattern), so the cost has to be real before any short result
can be trusted.
"""
import numpy as np
import pandas as pd
import pytest

from src.engine_v2.backtest.orchestrator import position_history, BacktestConfig
from src.engine_v2.execution.shorting import daily_borrow_fee


def _bars(closes, ticker="SPY"):
    idx = pd.date_range("2010-01-01", periods=len(closes), freq="B")
    cols = pd.MultiIndex.from_product([[ticker], ["Open", "High", "Low", "Close", "Volume"]])
    df = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for fld in ("Open", "High", "Low", "Close"):
        df[(ticker, fld)] = closes
    df[(ticker, "Volume")] = 1e8
    return df


def _series(n=250, seed=11):
    rng = np.random.default_rng(seed)
    return 100 * np.cumprod(1 + rng.normal(0.0, 0.003, n))  # ~flat drift, real vol


class ConstantShort:
    display_name = "Const Short"
    mechanism = "test-only"
    parameter_grid: dict = {}
    holding_period_cap = 10 ** 9

    def __init__(self):
        pass

    def forecast(self, bars, asof):
        return pd.Series({"SPY": -10.0})


class ConstantLong(ConstantShort):
    def forecast(self, bars, asof):
        return pd.Series({"SPY": 10.0})


def _final_equity(strategy_cls, borrow_bps):
    cfg = BacktestConfig(spread_bps_per_side=1.0, borrow_bps_annual=borrow_bps)
    bars = _bars(_series())
    return position_history(strategy_cls, {}, bars, bars.index, cfg)["equity"].iloc[-1]


def test_first_short_borrow_charge_matches_the_formula_exactly():
    """Bar-0 positions are identical across runs (same starting equity), so the first
    borrow charge -- levied at bar 1 on the bar-0 short -- is exact. (Later bars
    diverge because borrow lowers equity and re-sizes subsequent trades.)"""
    cfg0 = BacktestConfig(spread_bps_per_side=1.0, borrow_bps_annual=0.0)
    cfg = BacktestConfig(spread_bps_per_side=1.0, borrow_bps_annual=300.0)
    bars = _bars(_series())
    h0 = position_history(ConstantShort, {}, bars, bars.index, cfg0)
    h = position_history(ConstantShort, {}, bars, bars.index, cfg)

    qty0 = h0["qty"].iloc[0]
    close1 = bars["SPY"]["Close"].iloc[1]
    assert qty0 < 0, "strategy did not establish a short at bar 0"
    expected = daily_borrow_fee(abs(qty0) * close1, 300.0)

    gap_at_bar1 = h0["equity"].iloc[1] - h["equity"].iloc[1]
    assert gap_at_bar1 == pytest.approx(expected, rel=1e-3), (
        f"first borrow charge {gap_at_bar1:.4f}, formula says {expected:.4f}"
    )


def test_longs_never_pay_borrow():
    assert _final_equity(ConstantLong, 0.0) == _final_equity(ConstantLong, 300.0)


def test_higher_borrow_costs_a_short_more():
    assert _final_equity(ConstantShort, 300.0) < _final_equity(ConstantShort, 50.0)


def test_default_borrow_is_nonzero():
    """A short must not be silently free -- the whole point."""
    assert _final_equity(ConstantShort, 0.0) != _final_equity(ConstantShort, BacktestConfig().borrow_bps_annual)
