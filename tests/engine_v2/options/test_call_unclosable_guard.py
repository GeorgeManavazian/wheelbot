"""A3b: extend the TP-exit-feasibility guard to covered calls (owner decision
2026-08-01, middle path). A covered call whose own take-profit exit is
unreachable at the minimum tick or a guaranteed net loss must never be
written; the shares stay honestly naked for the day instead (retried daily,
counted by days_shares_uncovered). Calls stay EXEMPT from the A2 liquidity
gate -- refusing a call leaves shares naked, so only the arithmetic
impossibility (never judgment thresholds) may refuse one. Same parameter-free
floor as A3: credit >= max(tick/(1-tp), 2*commission/(tp*mult)), $0.025 under
FROZEN; tp None/>=1.0 means no TP exit exists, so nothing can be infeasible.

Background: 22.6% of backtest CALL entries (1,073/4,757) sold at bid <=
$0.02, and the A4 splice grows this population (median spliced credit 4-7c
passes the floor; the true pocket lint dies)."""
import pandas as pd

from src.engine_v2.options.portfolio import PortfolioState, step_one_day
from src.engine_v2.options.wheel import WheelConfig

D = pd.Timestamp("2026-07-21")
COLS = ["date", "expiry", "dte", "strike", "right", "bid", "ask", "mid",
        "close", "delta", "iv", "underlying"]


def _call_chain(bid, strike=70.0, delta=0.30, und=65.0):
    ch = pd.DataFrame([[D, D + pd.Timedelta(days=11), 11, strike, "C", bid,
                        bid + 0.01, bid + 0.005, bid + 0.005, delta, 0.2,
                        und]], columns=COLS)
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
        return 65.0

    def settle_price(self, tk, expiry):
        return 65.0

    def regime_row(self, tk, day):
        return None

    def eligible(self, tk, day):
        return True


def _call_state():
    # basis 60, no banked premium -> floor 60; the 70 strike qualifies
    return PortfolioState(cash=10_000.0, positions=[{
        "ticker": "GDX", "shares": 100, "phase": "CALL", "basis": 60.0,
        "premium": 0.0, "campaign": 1, "last_spot": 65.0, "short": None}])


def _cfg(tp=0.60, comm=0.65):
    return WheelConfig(ticker="GDX", starting_capital=100_000.0, put_delta=0.30,
                       call_delta=0.50, target_dte=11, take_profit_pct=tp,
                       commission_per_contract=comm, call_min_strike="basis")


def _step(ch, cfg):
    st = _call_state()
    r = step_one_day(st, M(ch), D, cfg, selector="plain", n_slots=1)
    return st, r


def test_micro_credit_call_is_refused():
    """The A3b defect: a 1c covered call's TP threshold is $0.004 -- below
    any possible ask -- and even a tick-close loses money after commissions.
    Must be refused; the shares stay honestly naked for the day."""
    st, r = _step(_call_chain(bid=0.01), _cfg(tp=0.60))
    assert [t.action for t in r.trades] == [], \
        "A3b: engine wrote a covered call whose TP exit is unsatisfiable"
    assert any(w[1] == "call_gated_unclosable" and w[2] == "GDX"
               for w in r.warnings)
    assert st.positions[0]["short"] is None


def test_boundary_at_tp60():
    # same floor as puts: max(0.01/0.40, 2*0.65/60) = 0.025
    st, r = _step(_call_chain(bid=0.024), _cfg(tp=0.60))
    assert [t.action for t in r.trades] == []
    st, r = _step(_call_chain(bid=0.025), _cfg(tp=0.60))
    assert [t.action for t in r.trades] == ["SELL_CALL"]


def test_no_tp_means_guard_inert_for_calls():
    for tp in (None, 1.0):
        st, r = _step(_call_chain(bid=0.01), _cfg(tp=tp))
        assert [t.action for t in r.trades] == ["SELL_CALL"], tp


def test_healthy_call_unchanged_and_quiet():
    st, r = _step(_call_chain(bid=1.00), _cfg())
    assert [t.action for t in r.trades] == ["SELL_CALL"]
    assert not any(w[1] == "call_gated_unclosable" for w in r.warnings)


def test_refused_call_day_counts_shares_uncovered():
    # the naked day must land in the A4 counter, not vanish
    st = _call_state()
    r = step_one_day(st, M(_call_chain(bid=0.01)), D, _cfg(), selector="plain",
                     n_slots=1)
    assert st.days_shares_uncovered == 1


def test_router_engine_call_gate():
    """Router engine, WHEEL cell: the covered-call entry must refuse a
    micro-credit call identically (third wiring site)."""
    from src.engine_v2.options.regime_router import run_regime_router
    cols = ["date", "expiry", "dte", "strike", "right", "bid", "ask", "mid",
            "close", "delta", "iv", "underlying"]
    rows = [["2024-01-02", "2024-01-09", 7, 470, "P", 2.00, 2.10, 2.05, 2.05,
             -0.30, 0.1, 472.0],
            ["2024-01-03", "2024-01-10", 7, 475, "C", 0.01, 0.02, 0.015,
             0.015, 0.30, 0.1, 470.0]]
    ch = pd.DataFrame(rows, columns=cols)
    ch["date"] = pd.to_datetime(ch["date"])
    ch["expiry"] = pd.to_datetime(ch["expiry"])
    st = pd.DataFrame([("2024-01-01", "uptrend", "calm", 0.2),
                       ("2024-01-02", "chop", "normal", 0.5)],
                      columns=["date", "trend", "vol", "vol_pctile"]
                      ).set_index("date")
    st.index = pd.to_datetime(st.index)
    cfg = WheelConfig(starting_capital=50_000.0, put_delta=0.30,
                      call_delta=0.30, take_profit_pct=0.60,
                      commission_per_contract=0.0, call_min_strike="basis")
    res = run_regime_router(ch, cfg, st)
    assert not [t for t in res.trades if t.action == "SELL_CALL"], \
        "A3b: router engine wrote an unclosable covered call"
    assert any(w[1] == "call_gated_unclosable" for w in (res.warnings or []))
    # skeptic F3: the refused day still counts as shares-uncovered
    assert res.days_shares_uncovered >= 1


def test_router_healthy_call_sells_and_stays_quiet():
    """Skeptic F2: a router mutant that warned call_gated_unclosable on EVERY
    covered-call day survived the whole suite -- only the wheel engine had a
    clean-run quiet assertion. A healthy router call must sell with ZERO gate
    warnings (warning spam would silently corrupt warnings-based checks)."""
    from src.engine_v2.options.regime_router import run_regime_router
    cols = ["date", "expiry", "dte", "strike", "right", "bid", "ask", "mid",
            "close", "delta", "iv", "underlying"]
    rows = [["2024-01-02", "2024-01-09", 7, 470, "P", 2.00, 2.10, 2.05, 2.05,
             -0.30, 0.1, 472.0],
            ["2024-01-03", "2024-01-10", 7, 475, "C", 3.00, 3.10, 3.05, 3.05,
             0.30, 0.1, 470.0]]
    ch = pd.DataFrame(rows, columns=cols)
    ch["date"] = pd.to_datetime(ch["date"])
    ch["expiry"] = pd.to_datetime(ch["expiry"])
    st = pd.DataFrame([("2024-01-01", "uptrend", "calm", 0.2),
                       ("2024-01-02", "chop", "normal", 0.5)],
                      columns=["date", "trend", "vol", "vol_pctile"]
                      ).set_index("date")
    st.index = pd.to_datetime(st.index)
    cfg = WheelConfig(starting_capital=50_000.0, put_delta=0.30,
                      call_delta=0.30, take_profit_pct=0.60,
                      commission_per_contract=0.0, call_min_strike="basis")
    res = run_regime_router(ch, cfg, st)
    assert [t.action for t in res.trades if t.action == "SELL_CALL"], \
        "harness broke: the healthy call should sell"
    assert not any(w[1] == "call_gated_unclosable" for w in (res.warnings or []))


def test_wheel_engine_call_gate():
    """Solo wheel engine: assignment then a micro-credit call day -- the
    solo engine must refuse identically (it is the backtest's only gate)."""
    from src.engine_v2.options.wheel import run_wheel
    d2 = D + pd.Timedelta(days=11)   # put expiry -> assignment (settle 35 < 40)
    d3 = d2 + pd.Timedelta(days=1)
    rows = [
        # day1: healthy 40P sold at $2.00
        [D, d2, 11, 40.0, "P", 2.00, 2.10, 2.05, 2.05, -0.30, 0.2, 41.0],
        # settle row for expiry day (underlying 35 -> assigned)
        [d2, d2, 0, 40.0, "P", 5.00, 5.10, 5.05, 5.05, -0.90, 0.2, 35.0],
        # day3: only a micro-credit call listed above the basis floor
        [d3, d3 + pd.Timedelta(days=11), 11, 40.0, "C", 0.01, 0.02, 0.015,
         0.015, 0.30, 0.2, 35.0],
    ]
    ch = pd.DataFrame(rows, columns=COLS)
    ch["date"] = pd.to_datetime(ch["date"])
    ch["expiry"] = pd.to_datetime(ch["expiry"])
    cfg = WheelConfig(ticker="GDX", starting_capital=100_000.0, put_delta=0.30,
                      call_delta=0.50, target_dte=11, take_profit_pct=0.60,
                      commission_per_contract=0.65, call_min_strike="basis")
    r = run_wheel(ch, cfg)
    assert not any(t.action == "SELL_CALL" for t in r.trades), \
        "A3b: solo engine wrote an unclosable covered call"
    assert any(w[1] == "call_gated_unclosable" for w in (r.warnings or []))
    # skeptic F3: the refused day still counts as shares-uncovered
    assert r.days_shares_uncovered >= 1
