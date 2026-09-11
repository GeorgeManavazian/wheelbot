"""A3: the TP-exit-feasibility guard. An entry whose own take-profit exit is
unreachable at the minimum tick ((1-tp)*credit < $0.01) or a guaranteed net
loss at the worst price the TP rule accepts (credit <= 2*commission/(tp*mult))
must never be sold -- the bot would be opening a position whose winning exit
is unsatisfiable by construction. Real case: WBD 25P sold at a $0.01 credit;
its TP threshold was $0.004, below any possible ask, and even a tick-close
loses money after commissions (banked $0.35, tick-close $1.65).

Parameter-free and always-on: the floor is derived entirely from cfg
arithmetic (tick, commission, multiplier, tp) -- no new config field
(provisional P1, 2026-08-01). Puts + roll destinations only, matching A2's
scope (provisional P2). In FROZEN production A2's rel-spread gate already
subsumes this floor 3.8x over; the guard's value is invariance under any A2
re-tuning and coverage in backtest configs where A2's OI/volume legs are off.
"""
import pandas as pd

from src.engine_v2.options.portfolio import PortfolioState, step_one_day
from src.engine_v2.options.wheel import WheelConfig

D = pd.Timestamp("2026-07-21")
COLS = ["date", "expiry", "dte", "strike", "right", "bid", "ask", "mid",
        "close", "delta", "iv", "underlying"]


def _chain(bid, ask, strike=25.0):
    ch = pd.DataFrame([[D, D + pd.Timedelta(days=11), 11, strike, "P", bid, ask,
                        (bid + ask) / 2, (bid + ask) / 2, -0.30, 0.2,
                        strike + 1.0]], columns=COLS)
    ch["date"] = pd.to_datetime(ch["date"])
    ch["expiry"] = pd.to_datetime(ch["expiry"])
    return ch


class M:
    universe = ["WBD"]

    def __init__(self, ch):
        self._ch = ch

    def chain(self, tk, d):
        return self._ch

    def spot(self, tk, d, fb):
        return 26.0

    def settle_price(self, tk, expiry):
        return 26.0

    def regime_row(self, tk, day):
        return None

    def eligible(self, tk, day):
        return True


def _cfg(tp=0.60, comm=0.65):
    return WheelConfig(ticker="WBD", starting_capital=100_000.0, put_delta=0.30,
                       call_delta=0.50, target_dte=11, take_profit_pct=tp,
                       commission_per_contract=comm, call_min_strike="basis")


def _step(ch, cfg):
    st = PortfolioState(cash=100_000.0, positions=[])
    r = step_one_day(st, M(ch), D, cfg, selector="plain", n_slots=1)
    return st, r


def test_the_real_wbd_entry_is_refused():
    """The defect: the bot sold WBD 25P at a $0.01 credit. TP threshold
    $0.004 -- below any possible ask -- and a tick-close is a guaranteed net
    loss. Must be refused, loudly."""
    st, r = _step(_chain(bid=0.01, ask=1.53), _cfg(tp=0.60))
    assert [t.action for t in r.trades] == [], \
        "A3: engine sold an entry whose TP exit is unsatisfiable"
    assert any(w[1] == "entry_gated_unclosable" for w in r.warnings)


def test_boundary_at_tp60():
    # floor = max(0.01/0.40, 2*0.65/(0.60*100)) = max(0.025, 0.02167) = 0.025
    st, r = _step(_chain(bid=0.024, ask=0.034), _cfg(tp=0.60))
    assert [t.action for t in r.trades] == []
    st, r = _step(_chain(bid=0.025, ask=0.035), _cfg(tp=0.60))
    assert [t.action for t in r.trades] == ["SELL_PUT"]   # thresh==tick PASSES


def test_boundary_at_tp50_commission_conjunct_binds():
    # comm=0.625 keeps the arithmetic binary-exact: floor =
    # max(0.01/0.50, 2*0.625/(0.50*100)) = max(0.020, 0.025) = 0.025 -- the
    # COMMISSION conjunct binds (kills a reachability-only mutant). The
    # production comm=0.65 floor (0.026) sits between representable doubles,
    # so the boundary is pinned here at an exact value instead.
    st, r = _step(_chain(bid=0.024, ask=0.034), _cfg(tp=0.50, comm=0.625))
    assert [t.action for t in r.trades] == []
    st, r = _step(_chain(bid=0.025, ask=0.035), _cfg(tp=0.50, comm=0.625))
    assert [t.action for t in r.trades] == ["SELL_PUT"]


def test_no_tp_exit_means_guard_inert():
    # tp=None / tp>=1.0: hold to expiry -- there is no TP exit to be
    # infeasible, so the $0.01 entry is legitimately sellable
    for tp in (None, 1.0):
        st, r = _step(_chain(bid=0.01, ask=0.02), _cfg(tp=tp))
        assert [t.action for t in r.trades] == ["SELL_PUT"], tp


def test_zero_commission_collapses_to_reachability():
    # floor = max(0.01/0.40, 0) = 0.025 at tp=0.60; kills a hard-coded-0.65
    # mutant (with comm=0 the commission conjunct must vanish, not linger)
    st, r = _step(_chain(bid=0.025, ask=0.035), _cfg(tp=0.60, comm=0.0))
    assert [t.action for t in r.trades] == ["SELL_PUT"]
    st, r = _step(_chain(bid=0.024, ask=0.034), _cfg(tp=0.60, comm=0.0))
    assert [t.action for t in r.trades] == []


def test_normal_credit_unchanged():
    st, r = _step(_chain(bid=0.94, ask=1.04), _cfg())
    assert [t.action for t in r.trades] == ["SELL_PUT"]


def test_tp_zero_refuses_everything():
    """Skeptic F1: tp=0 is a LIVE rule in try_take_profit (fires at any
    ask <= credit) whose worst accepted fill nets -2*commission on every
    entry. The floor must be infinite (refuse all), never 'inert' -- inert
    waved the guaranteed-loss class through under a degenerate config."""
    from src.engine_v2.options.fills import tp_exit_floor, tp_exit_feasible
    cfg = _cfg(tp=0.0)
    assert tp_exit_floor(cfg) == float("inf")
    assert tp_exit_feasible(5.00, cfg)[0] is False
    st, r = _step(_chain(bid=5.00, ask=5.10), cfg)
    assert [t.action for t in r.trades] == []


def test_roll_destination_veto_is_loud():
    """Skeptic F4: a refused roll destination runs the leg to expiry -- the
    warnings must say why instead of the roll silently not happening."""
    from src.engine_v2.options.wheel import run_wheel
    # day1: sell 40P at a healthy credit; day2: spot dives (tested put),
    # destination expiry offered only at a micro credit -> A3 refuses it
    d2 = D + pd.Timedelta(days=1)
    rows = []
    for day, exp, strike, bid, ask, dte, und in (
            (D,  D + pd.Timedelta(days=11), 40.0, 2.00, 2.10, 11, 41.0),
            (d2, D + pd.Timedelta(days=11), 40.0, 4.00, 4.10, 10, 36.0),
            (d2, D + pd.Timedelta(days=22), 40.0, 0.01, 0.02, 21, 36.0)):
        rows.append([day, exp, dte, strike, "P", bid, ask, (bid + ask) / 2,
                     (bid + ask) / 2, -0.30, 0.2, und])
    ch = pd.DataFrame(rows, columns=COLS)
    ch["date"] = pd.to_datetime(ch["date"])
    ch["expiry"] = pd.to_datetime(ch["expiry"])
    cfg = WheelConfig(ticker="WBD", starting_capital=100_000.0, put_delta=0.30,
                      call_delta=0.50, target_dte=11, take_profit_pct=0.60,
                      commission_per_contract=0.65, call_min_strike="basis",
                      roll_tested_puts=True)
    r = run_wheel(ch, cfg)
    assert not any(t.action == "ROLL_OPEN" for t in r.trades)
    assert any(w[1] == "roll_gated_unclosable" for w in (r.warnings or []))
