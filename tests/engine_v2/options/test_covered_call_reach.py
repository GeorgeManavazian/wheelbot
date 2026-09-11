"""A4: after the drawdown that caused an assignment, the basis floor sits
above the 12-strike window's reach (~+3.8% over spot), select_contract
returns None, and the covered-call branch was a SILENT no-op -- the income
half of the wheel never starts, shares sit naked, and the only trace was a
counter nobody reads. The engine must say so, loudly, every day it happens.
(The pull-side fix -- every listed OTM call spliced into the snapshot for
CALL-phase holdings -- lives in live/run_chain_snapshot.py.)"""
import pandas as pd

from src.engine_v2.options.portfolio import PortfolioState, step_one_day
from src.engine_v2.options.wheel import WheelConfig

D = pd.Timestamp("2026-07-21")
COLS = ["date", "expiry", "dte", "strike", "right", "bid", "ask", "mid",
        "close", "delta", "iv", "underlying"]


def _chain(strikes, right="C", und=70.0):
    rows = []
    for k, delta in strikes:
        rows.append([D, D + pd.Timedelta(days=11), 11, k, right, 1.00, 1.10,
                     1.05, 1.05, delta, 0.2, und])
    ch = pd.DataFrame(rows, columns=COLS)
    ch["date"] = pd.to_datetime(ch["date"])
    ch["expiry"] = pd.to_datetime(ch["expiry"])
    return ch


class M:
    universe = ["GDX"]

    def __init__(self, ch):
        self._ch = ch

    def chain(self, tk, d):
        return self._ch

    def spot(self, tk, d, fb):
        return 70.0

    def settle_price(self, tk, expiry):
        return 70.0

    def regime_row(self, tk, day):
        return None

    def eligible(self, tk, day):
        return True


def _call_state(basis=80.0, premium=100.0, shares=100):
    return PortfolioState(cash=10_000.0, positions=[{
        "ticker": "GDX", "shares": shares, "phase": "CALL", "basis": basis,
        "premium": premium, "campaign": 1, "last_spot": 70.0, "short": None}])


def _cfg():
    return WheelConfig(ticker="GDX", starting_capital=100_000.0, put_delta=0.30,
                       call_delta=0.50, target_dte=11, take_profit_pct=0.60,
                       call_min_strike="basis")


def test_unreachable_floor_is_loud_not_silent():
    """Assigned at 80, spot 70, banked $1/share -> floor 79. Window top 72.5:
    no strike qualifies. The old behavior: nothing sold, nothing said."""
    ch = _chain([(70.0, 0.55), (71.0, 0.50), (72.5, 0.45)])
    st = _call_state(basis=80.0, premium=100.0)
    r = step_one_day(st, M(ch), D, _cfg(), selector="plain", n_slots=1)
    assert [t.action for t in r.trades] == []
    assert any(w[1] == "covered_call_unreachable" and w[2] == "GDX"
               for w in r.warnings), \
        "A4: shares sit naked and the engine said nothing"


def test_reachable_floor_sells_and_stays_quiet():
    # floor 79, chain reaches 80 -> call written at 80, no warning
    ch = _chain([(70.0, 0.55), (75.0, 0.40), (80.0, 0.25)])
    st = _call_state(basis=80.0, premium=100.0)
    r = step_one_day(st, M(ch), D, _cfg(), selector="plain", n_slots=1)
    assert [t.action for t in r.trades] == ["SELL_CALL"]
    assert r.trades[0].contract.strike == 80.0
    assert not any(w[1].startswith("covered_call") for w in r.warnings)


def test_no_floor_means_no_warning():
    # plain-wheel config (call_min_strike=None): no floor exists, silence is
    # correct even when nothing sells (chain empty of calls)
    cfg = WheelConfig(ticker="GDX", starting_capital=100_000.0, put_delta=0.30,
                      call_delta=0.50, target_dte=11, take_profit_pct=0.60,
                      call_min_strike=None)
    ch = _chain([(70.0, -0.30)], right="P")   # no call rows at all
    st = _call_state(basis=None, premium=100.0)
    r = step_one_day(st, M(ch), D, cfg, selector="plain", n_slots=1)
    assert not any(w[1].startswith("covered_call") for w in r.warnings)


def test_floor_row_on_wrong_date_is_unreachable():
    # the only floor-reaching row belongs to a DIFFERENT date, so selection
    # cannot see it: same defect, same loud reason. (A "no_mark" branch was
    # deleted -- the skeptic proved a selected contract always marks, so
    # "unreachable" is the one true failure mode here.)
    ch = _chain([(70.0, 0.55), (80.0, 0.25)])
    ch.loc[ch["strike"] == 80.0, "date"] = D + pd.Timedelta(days=1)
    st = _call_state(basis=80.0, premium=100.0)
    r = step_one_day(st, M(ch), D, _cfg(), selector="plain", n_slots=1)
    assert [t.action for t in r.trades] == []
    assert any(w[1] == "covered_call_unreachable" for w in r.warnings)
