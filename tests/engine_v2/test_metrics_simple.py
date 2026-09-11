import numpy as np
import pandas as pd
import pytest
from src.engine_v2.backtest import metrics_simple as m

def _daily(n=252, start="2010-01-01"):
    return pd.date_range(start, periods=n, freq="B")

def test_infer_periods_per_year_daily():
    idx = _daily(300)
    assert m.infer_periods_per_year(idx) == pytest.approx(252, abs=6)

def test_infer_periods_per_year_minute():
    idx = pd.date_range("2010-01-04 09:30", periods=400, freq="min")
    # ~390 trading minutes/day * 252 days
    assert m.infer_periods_per_year(idx) > 90_000

def test_cagr_and_maxdd_on_known_curve():
    idx = _daily(253)
    equity = pd.Series(np.linspace(100.0, 110.0, 253), index=idx)  # +10% over ~1y
    assert m.cagr(equity, 252) == pytest.approx(0.10, abs=0.02)
    assert m.max_drawdown(equity) == pytest.approx(0.0, abs=1e-9)

def test_sharpe_frequency_scales():
    idx = _daily(252)
    rets = pd.Series(np.full(252, 0.001), index=idx)  # constant positive, zero std
    # zero variance -> guarded to 0.0, not inf/nan
    assert m.sharpe(rets, 252) == 0.0

def test_yearly_returns_splits_by_year():
    idx = pd.date_range("2010-01-01", "2011-12-31", freq="B")
    equity = pd.Series(np.linspace(100, 121, len(idx)), index=idx)
    yr = m.yearly_returns(equity)
    assert set(yr.index) == {2010, 2011}

def test_yearly_sharpe_one_per_year():
    idx = pd.date_range("2010-01-01", "2011-12-31", freq="B")
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)
    ys = m.yearly_sharpe(rets, 252)
    assert set(ys.index) == {2010, 2011}
    assert ys.notna().all()

def test_buy_hold_matches_hand_calc():
    idx = _daily(3)
    cols = pd.MultiIndex.from_tuples([("SPY", "Close")])
    bars = pd.DataFrame([[100.0], [110.0], [121.0]], index=idx, columns=cols)
    eq = m.buy_hold_equity(bars, {"SPY": 1.0}, 100_000.0)
    assert eq.iloc[-1] == pytest.approx(121_000.0)

def test_sharpe_too_short_is_nan():
    assert np.isnan(m.sharpe(pd.Series([], dtype=float), 252))
    assert np.isnan(m.sharpe(pd.Series([0.01]), 252))

def test_monthly_returns_shape_and_values():
    # 100 -> 110 in Jan, 110 -> 99 in Feb
    idx = pd.to_datetime(["2024-01-15", "2024-01-31", "2024-02-15", "2024-02-29"])
    eq = pd.Series([100.0, 110.0, 105.0, 99.0], index=idx)
    out = m.monthly_returns(eq)
    assert list(out.columns) == list(range(1, 13))
    assert out.index.tolist() == [2024]
    # Jan is the first month: return measured off the opening equity (100 -> 110)
    assert out.loc[2024, 1] == pytest.approx(0.10)
    # Feb: 110 -> 99
    assert out.loc[2024, 2] == pytest.approx(-0.10)
    assert np.isnan(out.loc[2024, 5])  # no data in May

def test_monthly_returns_spans_years():
    idx = pd.date_range("2023-11-01", "2024-02-29", freq="D")
    eq = pd.Series(np.linspace(100.0, 120.0, len(idx)), index=idx)
    out = m.monthly_returns(eq)
    assert out.index.tolist() == [2023, 2024]
    assert not np.isnan(out.loc[2023, 12])
    assert np.isnan(out.loc[2023, 1])

def test_monthly_returns_empty_series_returns_empty_frame():
    out = m.monthly_returns(pd.Series(dtype=float))
    assert out.empty

def test_monthly_returns_month_after_gap_is_not_nan():
    # Jan and Feb have data, March has NO data at all, April has data.
    # The gap month (March) must be NaN, but April -- which HAS real data --
    # must report a correct return measured off the last month that had data
    # (Feb), not off the NaN gap month.
    idx = pd.to_datetime([
        "2024-01-15", "2024-01-31",
        "2024-02-15", "2024-02-29",
        "2024-04-15", "2024-04-30",
    ])
    eq = pd.Series([100.0, 110.0, 115.0, 120.0, 130.0, 140.0], index=idx)
    out = m.monthly_returns(eq)
    assert np.isnan(out.loc[2024, 3])  # gap month itself: no data -> NaN
    assert out.loc[2024, 4] == pytest.approx(0.16667, rel=1e-4)  # 140/120 - 1
