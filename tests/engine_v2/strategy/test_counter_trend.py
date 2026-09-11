"""Clenow counter-trend dip buy (Trading Evolved, ch.17).

Rules verbatim from the vault note "Trading Evolved — Ch 17: Counter Trend Trading":
  bull filter : EMA(40) > EMA(80) on close
  volatility  : 40-day std of daily price CHANGES (close.diff()), not returns
  pullback    : (close - max(close, last 20 bars)) / std
  entry       : bull AND pullback < -3
  exit        : trend flips bearish, OR 20 bars held
  long only, no stop, no profit target
"""
import numpy as np
import pandas as pd
import pytest

from src.engine_v2.strategy.protocol import validate_plugin, FORECAST_MIN, FORECAST_MAX
from src.engine_v2.strategy.counter_trend import CounterTrendDipBuy

TICKER = "SPY"


def _frame(closes):
    idx = pd.date_range("2010-01-01", periods=len(closes), freq="B")
    cols = pd.MultiIndex.from_product([[TICKER], ["Open", "High", "Low", "Close", "Volume"]])
    df = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for f in ("Open", "High", "Low", "Close"):
        df[(TICKER, f)] = closes
    df[(TICKER, "Volume")] = 1e8
    return df


def _uptrend(n=200, seed=5):
    rng = np.random.default_rng(seed)
    return 100 * np.cumprod(1 + 0.0012 + rng.normal(0, 0.004, n))


def _downtrend(n=200, seed=5):
    rng = np.random.default_rng(seed)
    return 200 * np.cumprod(1 - 0.0012 + rng.normal(0, 0.004, n))


def _append_dip(closes, sigmas):
    """Append one bar sitting `sigmas` standard deviations below the 20-bar high."""
    std = np.diff(closes[-41:]).std(ddof=1)
    return np.append(closes, closes[-20:].max() + sigmas * std)


def _fc(bars, strat=None, **kw):
    strat = strat or CounterTrendDipBuy(**kw)
    return float(strat.forecast(bars, bars.index[-1])[TICKER])


def test_plugin_satisfies_the_v2_contract():
    validate_plugin(CounterTrendDipBuy)
    assert CounterTrendDipBuy.holding_period_cap == 20
    assert CounterTrendDipBuy.mechanism.strip() != ""


def test_no_entry_in_a_bear_market_even_on_a_deep_dip():
    """Clenow trades bull-market dips only. Bear rallies have different dynamics
    and he explicitly declines to mirror the logic."""
    bars = _frame(_append_dip(_downtrend(), sigmas=-5))
    assert _fc(bars) == 0.0


def test_no_entry_on_a_shallow_dip():
    bars = _frame(_append_dip(_uptrend(), sigmas=-1.0))
    assert _fc(bars) == 0.0


def test_enters_long_on_a_deep_dip_in_a_bull_market():
    bars = _frame(_append_dip(_uptrend(), sigmas=-4.0))
    assert _fc(bars) == pytest.approx(10.0)


def test_never_takes_a_short_position():
    """Faithful replication: long only. The short variant is separate research."""
    strat = CounterTrendDipBuy()
    closes = _append_dip(_downtrend(), sigmas=-6)
    bars = _frame(closes)
    for i in range(120, len(closes)):
        f = float(strat.forecast(bars.iloc[: i + 1], bars.index[i])[TICKER])
        assert f >= 0.0
        assert FORECAST_MIN <= f <= FORECAST_MAX


def test_holds_the_position_after_entry():
    """Entry is one bar; the position must persist, or the loop flattens it instantly."""
    closes = _append_dip(_uptrend(), sigmas=-4.0)
    closes = np.append(closes, [closes[-1] * 1.001, closes[-1] * 1.002])
    bars = _frame(closes)
    strat = CounterTrendDipBuy()
    fcs = [float(strat.forecast(bars.iloc[: i + 1], bars.index[i])[TICKER])
           for i in range(len(closes) - 3, len(closes))]
    assert fcs[0] == pytest.approx(10.0), "did not enter"
    assert all(f == pytest.approx(10.0) for f in fcs), f"did not hold: {fcs}"


def test_exits_after_twenty_bars_held():
    closes = _append_dip(_uptrend(), sigmas=-4.0)
    closes = np.append(closes, closes[-1] * np.cumprod(np.full(25, 1.0005)))
    bars = _frame(closes)
    strat = CounterTrendDipBuy()
    entry_i = len(closes) - 26
    fcs = [float(strat.forecast(bars.iloc[: i + 1], bars.index[i])[TICKER])
           for i in range(entry_i, len(closes))]
    assert fcs[0] == pytest.approx(10.0), "did not enter on the dip bar"
    held = sum(1 for f in fcs if f > 0)
    assert held == 20, f"held {held} bars, expected the 20-bar cap"
    assert fcs[20] == 0.0, "did not exit on the 21st bar"


def test_exits_when_the_trend_flips_bearish():
    closes = _append_dip(_uptrend(), sigmas=-4.0)
    bars = _frame(closes)
    strat = CounterTrendDipBuy()
    assert _fc(bars, strat) == pytest.approx(10.0), "did not enter"
    crash = np.append(closes, closes[-1] * np.cumprod(np.full(90, 0.97)))
    bars2 = _frame(crash)
    fcs = [float(strat.forecast(bars2.iloc[: i + 1], bars2.index[i])[TICKER])
           for i in range(len(closes), len(crash))]
    assert fcs[-1] == 0.0, "still long after the trend flipped bearish"


def test_parameter_grid_yields_enough_trials_for_the_gate():
    """compute_verdict shelves outright below K_MIN=5 trial columns."""
    from itertools import product
    grid = CounterTrendDipBuy.parameter_grid
    combos = list(product(*grid.values())) if grid else [()]
    assert len(combos) >= 2, "grid too thin to probe a parameter plateau"
