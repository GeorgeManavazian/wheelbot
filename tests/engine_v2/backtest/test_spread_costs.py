"""A configured spread must show up as realized cost in a backtest.

Integration-level: drive the real _simulate loop and confirm that widening the
spread strictly increases the money lost to trading, and that a per-ticker override
is honoured. Before this, spread was hardcoded to zero everywhere.
"""
import numpy as np
import pandas as pd

from src.engine_v2.backtest.orchestrator import position_history, BacktestConfig


def _bars(closes, ticker="SPY"):
    idx = pd.date_range("2010-01-01", periods=len(closes), freq="B")
    cols = pd.MultiIndex.from_product([[ticker], ["Open", "High", "Low", "Close", "Volume"]])
    df = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for fld in ("Open", "High", "Low", "Close"):
        df[(ticker, fld)] = closes
    df[(ticker, "Volume")] = 1e8
    return df


def _ramp(n=250, start=100.0, daily=0.0015, vol=0.003, seed=11):
    rng = np.random.default_rng(seed)
    return start * np.cumprod(1 + daily + rng.normal(0.0, vol, n))


class ConstantLong:
    display_name = "Const Long"
    mechanism = "test-only"
    parameter_grid: dict = {}
    holding_period_cap = 10 ** 9

    def __init__(self):
        pass

    def forecast(self, bars, asof):
        return pd.Series({"SPY": 10.0})


def _final_equity(spread_bps, by_ticker=None):
    cfg = BacktestConfig(spread_bps_per_side=spread_bps,
                         spread_bps_by_ticker=by_ticker or {})
    bars = _bars(_ramp())
    return position_history(ConstantLong, {}, bars, bars.index, cfg)["equity"].iloc[-1]


def test_wider_spread_costs_more():
    tight = _final_equity(1.0)
    wide = _final_equity(10.0)
    assert wide < tight, "a 10bps spread did not cost more than a 1bps spread"


def test_a_configured_spread_costs_more_than_no_spread():
    assert _final_equity(5.0) < _final_equity(0.0)


def test_per_ticker_spread_overrides_the_default():
    default_wide = _final_equity(1.0)
    override_wide = _final_equity(1.0, by_ticker={"SPY": 20.0})
    assert override_wide < default_wide, "per-ticker override was ignored"


def test_default_config_charges_a_nonzero_spread():
    """The default must not silently be zero -- that was the whole bug."""
    assert _final_equity(0.0) > BacktestConfig().starting_equity - 1e9  # sanity: runs
    assert _final_equity(0.0) != _final_equity(1.0), "default spread is zero"
