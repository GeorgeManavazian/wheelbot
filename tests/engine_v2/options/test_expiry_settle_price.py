"""An expiry must never settle against a carried price.

Found 2026-07-31. On expiry day `step_one_day` took moneyness from
`market.spot()`, which returns the CARRIED `pos["last_spot"]` when today's close
is missing. The `settle_price()` fallback was consulted only on the `d > expiry`
branch, so the ordinary `d == expiry` case had no protection at all: a deep
in-the-money put was booked PUT_EXPIRED (worthless) with no warning and no gap
record. Confirmed against the real TMO leg at $153,750 of settlement notional —
identical market reality, two different ledgers, and the wrong one is silent.

`zombie_check` does not cover this: it fires at 50% of the universe and one
ticker is 0.18%. The 2026-07-22 log shows 7 of 547 closes failing on an
ordinary day.
"""
import pandas as pd

from src.engine_v2.options.chain import Contract
from src.engine_v2.options.portfolio import PortfolioState, step_one_day
from src.engine_v2.options.wheel import WheelConfig

EXP = pd.Timestamp("2026-07-31")


class _Market:
    """Holds a close series; `spot` degrades to the caller's fallback exactly as
    LiveMarket and BatchMarket both do."""
    universe = []

    def __init__(self, closes):
        self._closes = closes

    def chain(self, t, d): return None
    def regime_row(self, t, d): return None
    def eligible(self, t, d): return True

    def spot(self, t, d, fallback):
        d = pd.Timestamp(d)
        return float(self._closes[d]) if d in self._closes.index else fallback

    def settle_price(self, t, expiry):
        pre = self._closes[self._closes.index <= pd.Timestamp(expiry)]
        return float(pre.iloc[-1]) if len(pre) else None


def _cfg():
    return WheelConfig(ticker="TMO", put_delta=0.30, call_delta=0.50, target_dte=11,
                       take_profit_pct=0.60, starting_capital=100_000.0,
                       call_min_strike="basis")


def _state(last_spot=576.0):
    pos = {"ticker": "TMO", "shares": 0, "phase": "PUT", "basis": None,
           "premium": 2400.0, "campaign": 1, "last_spot": last_spot,
           "short": {"contract": Contract("TMO", EXP, 512.5, "P"), "contracts": 3,
                     "credit": 8.0, "last_mid": 8.0, "last_ask": 8.2}}
    return PortfolioState(cash=100_000.0, positions=[pos], campaign=1)


def test_settles_normally_when_the_expiry_day_close_is_present():
    closes = pd.Series([576.0], index=[EXP])
    state = _state()
    r = step_one_day(state, _Market(closes), EXP, _cfg(), selector="chop", n_slots=1)
    assert [t.action for t in r.trades] == ["PUT_EXPIRED"]
    assert state.cash == 100_000.0


def test_assigns_when_the_expiry_day_close_is_below_the_strike():
    closes = pd.Series([480.0], index=[EXP])
    state = _state()
    r = step_one_day(state, _Market(closes), EXP, _cfg(), selector="chop", n_slots=1)
    assert [t.action for t in r.trades] == ["ASSIGNED"]
    assert state.cash == 100_000.0 - 512.5 * 100 * 3
    assert state.positions[0]["shares"] == 300


def test_refuses_to_settle_when_the_expiry_day_close_is_missing():
    """The defect. The carried spot said 576 (worthless); the true close was 480
    (assignment). With no close for the day, the engine must decline to decide
    rather than book the carried price's answer."""
    closes = pd.Series([576.0], index=[EXP - pd.Timedelta(days=1)])   # yesterday only
    state = _state(last_spot=576.0)
    r = step_one_day(state, _Market(closes), EXP, _cfg(), selector="chop", n_slots=1)

    assert [t.action for t in r.trades] == []          # nothing booked
    assert state.cash == 100_000.0                     # no phantom settlement
    assert state.positions[0]["short"] is not None     # leg still open
    assert any(w[1] == "expiry_unsettleable" for w in r.warnings)


def test_a_stale_prior_close_is_not_accepted_as_the_settlement_price():
    """settle_price() returns the last close ON OR BEFORE the expiry, which on an
    outage day is a DIFFERENT day's price. Consulting it here would swap one
    stale number for another -- the fix must require the expiry day itself."""
    closes = pd.Series([576.0, 574.0],
                       index=[EXP - pd.Timedelta(days=2), EXP - pd.Timedelta(days=1)])
    state = _state()
    r = step_one_day(state, _Market(closes), EXP, _cfg(), selector="chop", n_slots=1)
    assert [t.action for t in r.trades] == []
    assert state.positions[0]["short"] is not None


def test_a_late_resolved_expiry_still_settles_on_the_expiry_day_close():
    """Missed days are real (see gaps.jsonl). When the bot returns, the expiry
    day's close IS in the history it pulls, so the leg must settle then -- at the
    expiry day's price, not at the day it happened to notice."""
    closes = pd.Series([480.0, 600.0], index=[EXP, EXP + pd.Timedelta(days=3)])
    state = _state()
    later = EXP + pd.Timedelta(days=3)
    r = step_one_day(state, _Market(closes), later, _cfg(), selector="chop", n_slots=1)
    assert [t.action for t in r.trades] == ["ASSIGNED"]     # 480 < 512.5
    assert any(w[1] == "expiry_resolved_late" for w in r.warnings)
