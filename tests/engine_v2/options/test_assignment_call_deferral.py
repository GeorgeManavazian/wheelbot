"""A9: the covered call must not be sold in the same session as the
assignment. The assignment notice arrives after the close and the shares are
not deliverable until the next session -- selling the call on assignment day
banks a session of premium at prices a real account could never trade
(measured: 180 same-session calls, $433k of phantom premium across the
9-ticker solo backtest). Rule: an assignment BOOKED in the step for session D
makes the first legal covered-call sale the next stepped session. The whole
call block is skipped on D -- selection, warnings and sale -- so a structural
one-day deferral never fires the A4/A3b naked-shares warnings (which would
email the owner daily, once per assignment).
"""
import pandas as pd

from src.engine_v2.options.chain import Contract
from src.engine_v2.options.portfolio import PortfolioState, step_one_day
from src.engine_v2.options.wheel import WheelConfig, run_wheel
from src.engine_v2.options.regime_router import run_regime_router

COLS = ["date", "expiry", "dte", "strike", "right", "bid", "ask", "mid",
        "close", "delta", "iv", "underlying"]

D0 = pd.Timestamp("2024-01-09")     # assignment session
D1 = pd.Timestamp("2024-01-10")     # first legal covered-call session


def _chain(rows):
    ch = pd.DataFrame(rows, columns=COLS)
    ch["date"] = pd.to_datetime(ch["date"])
    ch["expiry"] = pd.to_datetime(ch["expiry"])
    return ch


# ---- portfolio.py (the money path: batch backtest AND live EOD) ----

class _Market:
    """Held GDX, put expiring D0 ITM; healthy call rows on D0 AND D1 so the
    only thing standing between the engine and a same-day sale is the rule."""
    universe = []

    def __init__(self):
        self._ch = {
            D0: _chain([[D0, D0 + pd.Timedelta(days=7), 7, 32.0, "C",
                         1.00, 1.10, 1.05, 1.05, 0.30, 0.2, 28.0]]),
            D1: _chain([[D1, D1 + pd.Timedelta(days=6), 6, 32.0, "C",
                         0.90, 1.00, 0.95, 0.95, 0.30, 0.2, 28.5]]),
        }

    def chain(self, tk, d):
        return self._ch.get(d)

    def spot(self, tk, d, fb):
        return 28.0

    def settle_price(self, tk, expiry):
        return 28.0                      # strike 30 put -> ITM -> assigned

    def regime_row(self, tk, d):
        return None

    def eligible(self, tk, d):
        return True


def _put_state():
    put = Contract("GDX", D0, 30.0, "P")
    pos = {"ticker": "GDX", "shares": 0, "phase": "PUT", "basis": None,
           "premium": 120.0, "campaign": 1, "last_spot": 28.0,
           "short": {"contract": put, "contracts": 1, "credit": 1.2,
                     "last_mid": 1.2}}
    return PortfolioState(cash=100_120.0, positions=[pos], campaign=1)


def _cfg(**kw):
    base = dict(ticker="GDX", put_delta=0.20, call_delta=0.30, target_dte=7,
                take_profit_pct=None, starting_capital=100_000.0,
                commission_per_contract=0.0, call_min_strike="basis")
    base.update(kw)
    return WheelConfig(**base)


def test_portfolio_no_call_on_assignment_day_sells_next_session():
    state = _put_state()
    mkt = _Market()
    r0 = step_one_day(state, mkt, D0, _cfg(), selector="plain", n_slots=1)
    assert [t.action for t in r0.trades] == ["ASSIGNED"], \
        "A9: covered call sold in the same session as the assignment"
    assert state.days_shares_uncovered == 1   # one legit uncovered session
    r1 = step_one_day(state, mkt, D1, _cfg(), selector="plain", n_slots=1)
    assert [t.action for t in r1.trades] == ["SELL_CALL"]
    assert r1.trades[0].date == D1


def test_portfolio_deferral_day_emits_no_naked_shares_warnings():
    """The WHOLE call block is skipped on D0: a structural deferral must not
    fire covered_call_unreachable (A4 daily email) or call_gated_unclosable
    (A3b/C16b class) for a day on which selling was never legal. The fixture
    makes the basis floor UNREACHABLE (floor 30.0, only strike 29) so that a
    mutant which gates only the sale -- letting selection and warnings run --
    fails here loudly instead of passing vacuously."""
    put = Contract("GDX", D0, 30.0, "P")
    pos = {"ticker": "GDX", "shares": 0, "phase": "PUT", "basis": None,
           "premium": 0.0, "campaign": 1, "last_spot": 28.0,
           "short": {"contract": put, "contracts": 1, "credit": 1.2,
                     "last_mid": 1.2}}
    state = PortfolioState(cash=100_120.0, positions=[pos], campaign=1)
    mkt = _Market()
    mkt._ch[D0] = _chain([[D0, D0 + pd.Timedelta(days=7), 7, 29.0, "C",
                           1.00, 1.10, 1.05, 1.05, 0.30, 0.2, 28.0]])
    r0 = step_one_day(state, mkt, D0, _cfg(), selector="plain", n_slots=1)
    assert [t.action for t in r0.trades] == ["ASSIGNED"]
    kinds = [w[1] for w in r0.warnings]
    assert "covered_call_unreachable" not in kinds
    assert "call_gated_unclosable" not in kinds


def test_portfolio_deferral_keys_on_booking_session_not_expiry():
    """A late-booked assignment (expiry_resolved_late, A10-deferred class)
    defers from the session that BOOKED it -- the engine learns after that
    session's close, so the next session is still the first legal sale."""
    state = _put_state()
    mkt = _Market()
    late = D0 + pd.Timedelta(days=1)     # booked one session late == D1
    d2 = D1 + pd.Timedelta(days=1)
    mkt._ch[late] = mkt._ch.pop(D1)      # call row available on booking day
    mkt._ch[late] = mkt._ch[late].assign(date=late)
    mkt._ch[d2] = _chain([[d2, d2 + pd.Timedelta(days=5), 5, 32.0, "C",
                           0.80, 0.90, 0.85, 0.85, 0.30, 0.2, 28.5]])
    r_late = step_one_day(state, mkt, late, _cfg(), selector="plain", n_slots=1)
    acts = [t.action for t in r_late.trades]
    assert acts == ["ASSIGNED"], \
        "A9: late-booked assignment must defer from the booking session"
    r2 = step_one_day(state, mkt, d2, _cfg(), selector="plain", n_slots=1)
    assert [t.action for t in r2.trades] == ["SELL_CALL"]


# ---- wheel.py (solo backtest) ----

def test_wheel_no_call_on_assignment_day_sells_next_session():
    rows = [
        ["2024-01-02", "2024-01-09", 7, 470, "P", 2.00, 2.10, 2.05, 2.05, -0.30, 0.1, 472.0],
        ["2024-01-09", "2024-01-09", 0, 470, "P", 5.00, 5.10, 5.05, 5.05, -0.99, 0.1, 465.0],
        ["2024-01-09", "2024-01-16", 7, 475, "C", 3.00, 3.10, 3.05, 3.05, 0.30, 0.1, 465.0],
        ["2024-01-10", "2024-01-16", 6, 475, "C", 2.80, 2.90, 2.85, 2.85, 0.30, 0.1, 466.0],
        ["2024-01-16", "2024-01-16", 0, 475, "C", 5.00, 5.10, 5.05, 5.05, 0.99, 0.1, 480.0],
    ]
    cfg = WheelConfig(starting_capital=50_000.0, put_delta=0.30, call_delta=0.30,
                      take_profit_pct=None, commission_per_contract=0.0)
    res = run_wheel(_chain(rows), cfg)
    acts = [(str(t.date.date()), t.action) for t in res.trades]
    assert acts == [("2024-01-02", "SELL_PUT"), ("2024-01-09", "ASSIGNED"),
                    ("2024-01-10", "SELL_CALL"), ("2024-01-16", "CALLED_AWAY")], \
        "A9: wheel sold the covered call in the assignment session"
    # cash walk: +200 put credit; -47000 assigned; +280 call credit AT D+1
    # PRICES (not D0's 300 -- that is the phantom session); +47500 called away
    assert res.final_cash == 50_000 + 200 - 47_000 + 280 + 47_500


# ---- regime_router.py ----

def test_router_no_call_on_assignment_day_sells_next_session():
    rows = [
        ["2024-01-02", "2024-01-09", 7, 470, "P", 2.00, 2.10, 2.05, 2.05, -0.30, 0.1, 472.0],
        ["2024-01-09", "2024-01-09", 0, 470, "P", 5.00, 5.10, 5.05, 5.05, -0.99, 0.1, 465.0],
        ["2024-01-09", "2024-01-16", 7, 475, "C", 1.00, 1.10, 1.05, 1.05, 0.30, 0.1, 465.0],
        ["2024-01-10", "2024-01-16", 6, 475, "C", 0.95, 1.05, 1.00, 1.00, 0.30, 0.1, 466.0],
    ]
    states = pd.DataFrame({"trend": "chop", "vol": "normal", "vol_pctile": 0.5},
                          index=pd.DatetimeIndex(["2024-01-01"]))
    cfg = WheelConfig(starting_capital=50_000.0, put_delta=0.30, call_delta=0.30,
                      take_profit_pct=None, commission_per_contract=0.0,
                      call_min_strike="basis")
    res = run_regime_router(_chain(rows), cfg, states)
    acts = [(str(t.date.date()), t.action) for t in res.trades]
    assert ("2024-01-09", "SELL_CALL") not in acts, \
        "A9: router sold the covered call in the assignment session"
    assert ("2024-01-10", "SELL_CALL") in acts
