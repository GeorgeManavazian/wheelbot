"""C1: a carried (frozen-in-place) mark was unfalsifiable -- TMO sat at
$11.30 for six snapshots and nothing distinguished "today's price" from "the
last price we ever got". Section 3 stamps `mark_asof` (the step date) and
`mark_quote_time` (the chain row's per-leg quote time, when the live path
provides one) EVERY time it re-marks a leg -- and never on the carried path,
so `mark_asof < today` IS the carried-mark discriminator the dashboard
renders. Frozen legs (A10f) are never stamped: they are deliberately carried.
"""
import pandas as pd
import pytest

from src.engine_v2.options.chain import Contract
from src.engine_v2.options.portfolio import PortfolioState, step_one_day
from src.engine_v2.options.wheel import WheelConfig

COLS = ["date", "expiry", "dte", "strike", "right", "bid", "ask", "mid",
        "close", "delta", "iv", "underlying"]
D1 = pd.Timestamp("2026-07-21")
D2 = pd.Timestamp("2026-07-22")
EXP = pd.Timestamp("2026-08-21")


def _chain(d):
    ch = pd.DataFrame([[d, EXP, (EXP - d).days, 30.0, "P", 2.00, 2.10, 2.05,
                        2.05, -0.30, 0.2, 32.0]], columns=COLS)
    ch["date"] = pd.to_datetime(ch["date"])
    ch["expiry"] = pd.to_datetime(ch["expiry"])
    return ch


class _M:
    universe = []

    def __init__(self, chains):
        self._ch = chains

    def chain(self, tk, d):
        return self._ch.get(d)

    def spot(self, tk, d, fb):
        return 32.0

    def settle_price(self, tk, e):
        return 32.0

    def regime_row(self, tk, d):
        return None

    def eligible(self, tk, d):
        return True


def _state():
    pos = {"ticker": "GDX", "shares": 0, "phase": "PUT", "basis": None,
           "premium": 210.0, "campaign": 1, "last_spot": 32.0,
           "short": {"contract": Contract("GDX", EXP, 30.0, "P"),
                     "contracts": 1, "credit": 2.10, "last_mid": 2.05}}
    return PortfolioState(cash=100_210.0, positions=[pos], campaign=1)


def _cfg():
    return WheelConfig(ticker="GDX", put_delta=0.20, call_delta=0.30,
                       target_dte=30, take_profit_pct=None,
                       starting_capital=100_000.0, commission_per_contract=0.0)


def test_remark_stamps_asof_and_carried_day_does_not_advance_it():
    st = _state()
    mkt = _M({D1: _chain(D1)})          # D2 has NO chain -> carried
    step_one_day(st, mkt, D1, _cfg(), selector="plain", n_slots=1)
    sh = st.positions[0]["short"]
    assert sh.get("mark_asof") == str(D1.date()), \
        "C1: a re-marked leg must stamp mark_asof with the step date"
    step_one_day(st, mkt, D2, _cfg(), selector="plain", n_slots=1)
    assert st.positions[0]["short"]["mark_asof"] == str(D1.date()), \
        "C1: a CARRIED mark advanced its as-of -- the discriminator is dead"


def test_frozen_leg_is_never_stamped():
    st = _state()
    st.positions[0]["ca_frozen"] = {"date": str(D1.date()), "ratio": 0.5}
    step_one_day(st, _M({D1: _chain(D1)}), D1, _cfg(),
                 selector="plain", n_slots=1)
    assert "mark_asof" not in st.positions[0]["short"], \
        "C1 x A10f: a frozen leg was stamped as freshly marked"


def test_mark_quote_time_rides_along_when_the_chain_has_it():
    st = _state()
    ch = _chain(D1)
    ch["quote_time"] = 1754140000000.0
    step_one_day(st, _M({D1: ch}), D1, _cfg(), selector="plain", n_slots=1)
    assert st.positions[0]["short"].get("mark_quote_time") == 1754140000000.0


def test_backtest_frames_without_the_column_stamp_none_quote_time():
    """Backtest chains have no quote_time column -- mark_asof still stamps
    (it is the step date, always known); mark_quote_time stays absent/None.
    Trades and cash are pinned byte-identical by the whole existing suite."""
    st = _state()
    step_one_day(st, _M({D1: _chain(D1)}), D1, _cfg(),
                 selector="plain", n_slots=1)
    assert st.positions[0]["short"].get("mark_quote_time") is None
