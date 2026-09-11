"""The backtest loop must compute profit and loss.

Before this file existed, `_fold_trial_returns` only ever subtracted slippage from
equity -- price movement never entered the equity curve. An always-long strategy on
a rising asset produced zero positive days. test_orchestrator.py asserted the shape
of the verdict dict and never its values, so nothing caught it.

Same failure mode as the gate bug: the accept path was never asserted.
"""
import numpy as np
import pandas as pd
import pytest

from src.engine_v2.backtest.orchestrator import _fold_trial_returns, BacktestConfig


def _bars(closes, ticker="SPY"):
    idx = pd.date_range("2010-01-01", periods=len(closes), freq="B")
    cols = pd.MultiIndex.from_product([[ticker], ["Open", "High", "Low", "Close", "Volume"]])
    df = pd.DataFrame(index=idx, columns=cols, dtype=float)
    df[(ticker, "Close")] = closes
    df[(ticker, "Open")] = closes
    df[(ticker, "High")] = closes
    df[(ticker, "Low")] = closes
    df[(ticker, "Volume")] = 1e8
    return df


def _ramp(n=250, start=100.0, daily=0.0015, vol=0.003, seed=11):
    """Drifting asset with real volatility. A noiseless ramp has sigma=0, which makes
    size_position return 0 -- no position, no P&L, nothing to test. Deterministic seed."""
    rng = np.random.default_rng(seed)
    rets = daily + rng.normal(0.0, vol, n)
    return start * np.cumprod(1 + rets)


class _Fixed:
    """Constant forecast plugin. `_forecast` set per subclass."""
    display_name = "Fixed"
    mechanism = "test-only"
    parameter_grid: dict = {}
    holding_period_cap = 10_000  # effectively no time exit
    _forecast = 10.0

    def __init__(self):
        pass

    def forecast(self, bars, asof):
        return pd.Series({"SPY": self._forecast})


class ConstantLong(_Fixed):
    _forecast = 10.0


class ConstantShort(_Fixed):
    _forecast = -10.0


class ShortHold(_Fixed):
    _forecast = 10.0
    holding_period_cap = 3


def _run(strategy_cls, closes):
    bars = _bars(closes)
    return _fold_trial_returns(strategy_cls, {}, bars, bars.index, BacktestConfig())


def test_holding_a_rising_asset_makes_money():
    """The assertion that never existed. Long a steadily rising asset must profit."""
    rets = _run(ConstantLong, _ramp())
    total = (1 + rets).prod() - 1
    assert (rets > 0).any(), "not a single positive day -- P&L is not being computed"
    assert total > 0, f"long a rising asset lost money: {total:+.4%}"


def test_holding_a_falling_asset_loses_money():
    rets = _run(ConstantLong, _ramp(daily=-0.002))
    assert (1 + rets).prod() - 1 < 0


def test_shorting_a_falling_asset_makes_money():
    """Engine v2 exists to trade long AND short. The short side must earn."""
    rets = _run(ConstantShort, _ramp(daily=-0.002))
    assert (1 + rets).prod() - 1 > 0


def test_shorting_a_rising_asset_loses_money():
    rets = _run(ConstantShort, _ramp(daily=+0.002))
    assert (1 + rets).prod() - 1 < 0


def test_costs_are_charged_on_position_change_not_gross_notional():
    """Opening a position is one big trade. Holding it should trade ~nothing. The old
    loop re-opened the full notional every bar and paid slippage on gross exposure."""
    from src.engine_v2.backtest.orchestrator import position_history

    bars = _bars(_ramp())
    hist = position_history(ConstantLong, {}, bars, bars.index, BacktestConfig())
    opening_trade = hist["traded"].iloc[0]
    steady_state = hist["traded"].iloc[5:].mean()
    assert opening_trade > 0, "no opening trade at all"
    assert steady_state < 0.1 * opening_trade, (
        f"steady-state trading {steady_state:.1f} shares/bar vs opening {opening_trade:.1f} "
        "-- costs are being charged on gross exposure, not on position change"
    )


def test_holding_period_cap_forces_a_time_exit():
    """Clenow's 20-day exit needs this. `holding_period_cap` was read and discarded."""
    from src.engine_v2.backtest.orchestrator import position_history

    bars = _bars(_ramp())
    hist = position_history(ShortHold, {}, bars, bars.index, BacktestConfig())
    assert (hist["qty"] == 0).any(), "position never flattened despite cap=3"
    held_runs = (hist["qty"] != 0).astype(int).groupby(
        (hist["qty"] == 0).cumsum()
    ).sum()
    assert held_runs.max() <= 3, f"held {held_runs.max()} bars with cap=3"


def test_no_lookahead_forecast_sees_only_past_and_present():
    """The loop must never hand a strategy a bar dated after asof."""
    seen = []

    class Spy(_Fixed):
        def forecast(self, bars, asof):
            seen.append((bars.index.max(), asof))
            return pd.Series({"SPY": 1.0})

    bars = _bars(_ramp(n=30))
    _fold_trial_returns(Spy, {}, bars, bars.index, BacktestConfig())
    assert all(last <= asof for last, asof in seen), "strategy saw a future bar"
