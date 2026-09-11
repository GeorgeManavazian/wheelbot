"""The WRITE gate needs a minimum credit that does NOT depend on take_profit_pct.

THE DEFECT (found 2026-08-15, while trying to A/B `call_take_profit_pct = 1.0`).
`tp_exit_floor` returns None for any tp >= 1.0 -- correctly, since a
hold-to-expiry short has no take-profit exit to be unreachable -- and
`tp_exit_feasible` then short-circuits to (True, "") for EVERY credit,
including bid = $0.00:

    tp=0.60 -> floor $0.025, feasible(bid=0.00) = (False, 'tp_unreachable')
    tp=1.00 -> floor None,   feasible(bid=0.00) = (True,  '')

On the PUT leg that hole is covered: the A2 liquidity gate and the intrinsic
and yield floors all still run. On the CALL leg it is not. Covered calls are
deliberately EXEMPT from A2 (portfolio.py, "refusing a call leaves shares
naked, so only arithmetic impossibility may refuse one"), so A3b -- which is
tp_exit_feasible -- is the ONLY minimum-credit rule the call leg has. Setting
call_take_profit_pct >= 1.0 deletes it, and the engine will then write calls
at a bid of $0.00 for `sell_proceeds(bid=0, n) = -friction * n`: a guaranteed
loss, by construction, on every contract.

WHY IT MATTERS BEYOND THE BUG. The 2026-08-14 exit sweep reported
`call_gated_unclosable` falling 1,333 -> 0 under call_take_profit_pct=1.0 and
read that as the jam being fixed. It is an arithmetic identity: the counter is
raised by the guard, and the guard has been removed. A refusal class cannot
fire once the rule that raises it is gone. That reading was withdrawn, and this
file is what makes the honest A/B possible -- with the write floor pinned
independently, `call_take_profit_pct = 1.0` becomes a test of "hold vs take
profit" rather than a test of "hold, and also sell worthless calls at a debit".

DEFAULT-IDENTICAL. At every take-profit in the repo (0.50, 0.60) the A3b floor
is already $0.02-$0.025, well above MIN_TICK, so the new floor never binds and
the live config resolves byte-identical. It binds only where A3b has gone
silent, which is exactly the hole.
"""
import pandas as pd
import pytest

from src.engine_v2.options.fills import (MIN_TICK, tp_exit_feasible,
                                         write_credit_floor, write_credit_ok)
from src.engine_v2.options.portfolio import PortfolioState, step_one_day
from src.engine_v2.options.wheel import WheelConfig, sell_proceeds

D = pd.Timestamp("2026-07-21")
COLS = ["date", "expiry", "dte", "strike", "right", "bid", "ask", "mid",
        "close", "delta", "iv", "underlying"]


def _chain(bid, right, strike, delta, und=65.0):
    ch = pd.DataFrame([[D, D + pd.Timedelta(days=11), 11, strike, right, bid,
                        bid + 0.01, bid + 0.005, bid + 0.005, delta, 0.2,
                        und]], columns=COLS)
    ch["date"] = pd.to_datetime(ch["date"])
    ch["expiry"] = pd.to_datetime(ch["expiry"])
    return ch


def _call_chain(bid, strike=70.0):
    return _chain(bid, "C", strike, 0.30)


def _put_chain(bid, strike=60.0):
    return _chain(bid, "P", strike, 0.30)


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


def _held_state():
    """100 shares, basis 60, no banked premium -> basis floor 60."""
    return PortfolioState(cash=10_000.0, positions=[{
        "ticker": "GDX", "shares": 100, "phase": "CALL", "basis": 60.0,
        "premium": 0.0, "campaign": 1, "last_spot": 65.0, "short": None}])


def _flat_state():
    return PortfolioState(cash=100_000.0, positions=[])


def _cfg(tp=0.60, call_tp=None, **kw):
    return WheelConfig(ticker="GDX", starting_capital=100_000.0,
                       put_delta=0.30, call_delta=0.50, target_dte=11,
                       take_profit_pct=tp, call_take_profit_pct=call_tp,
                       commission_per_contract=0.65, fees_per_contract=0.05,
                       call_min_strike="basis", **kw)


def _actions(state, ch, cfg):
    r = step_one_day(state, M(ch), D, cfg, selector="plain", n_slots=1)
    return [t.action for t in r.trades], r


# --------------------------------------------------------------------------
# 1. the floor itself
# --------------------------------------------------------------------------

def test_write_floor_is_one_tick_and_covers_its_own_friction():
    """Two independent reasons a write must clear a floor, and the binding one.

    A quote below MIN_TICK is not a quote at all; and a credit that does not
    cover the one-sided friction it pays is a guaranteed loss. At the live
    contract multiplier the tick is the binding constraint ($0.01 vs $0.007),
    so the floor is one tick -- but it is written as the max of both so a
    nickel-tick or thin-multiplier config cannot slip underneath."""
    cfg = _cfg()
    assert write_credit_floor(cfg) == pytest.approx(MIN_TICK)
    assert write_credit_floor(cfg) >= \
        cfg.friction_per_contract / cfg.contract_multiplier


def test_a_write_at_the_floor_nets_positive_and_below_it_does_not():
    """The zero-crossing this guard exists to sit on. Pinned against
    sell_proceeds itself, so the guard and the fills can never drift."""
    cfg = _cfg()

    class _Mk:
        def __init__(self, bid):
            self.bid = bid
    assert sell_proceeds(_Mk(write_credit_floor(cfg)), 1, cfg) > 0
    assert sell_proceeds(_Mk(0.0), 1, cfg) == \
        pytest.approx(-cfg.friction_per_contract)


def test_write_credit_ok_names_its_own_refusal_reason():
    """NOT the A3b reason. The misreading above happened because one
    counter was read as evidence of a mechanism it did not measure; a second
    rule sharing that counter would repeat it."""
    cfg = _cfg()
    assert write_credit_ok(0.05, cfg) == (True, "")
    ok, why = write_credit_ok(0.0, cfg)
    assert ok is False
    assert why == "no_credit"
    assert "unclosable" not in why


# --------------------------------------------------------------------------
# 2. the defect: hold-to-expiry must not open the floodgate
# --------------------------------------------------------------------------

def test_hold_to_expiry_still_refuses_a_zero_bid_covered_call():
    """THE SPEC. call_take_profit_pct=1.0 disables the take-profit exit, so
    A3b goes silent -- and before this guard existed the engine wrote the
    $0.00 call anyway, banking -$0.70 of friction per contract."""
    st = _held_state()
    acts, _ = _actions(st, _call_chain(bid=0.0), _cfg(tp=0.60, call_tp=1.0))
    assert acts == [], \
        "engine wrote a covered call at a $0.00 bid -- a guaranteed loss"


def test_hold_to_expiry_still_refuses_a_sub_tick_covered_call():
    st = _held_state()
    acts, _ = _actions(st, _call_chain(bid=0.004), _cfg(tp=0.60, call_tp=1.0))
    assert acts == []


def test_hold_to_expiry_does_write_a_call_with_real_credit():
    """The guard must not become a veto on the mechanic it is unblocking.
    A $0.40 call is refused by A3b at tp=0.60 (floor $0.025 is cleared, but
    this is the arm where tp is disabled) and must be WRITTEN at tp=1.0."""
    st = _held_state()
    acts, _ = _actions(st, _call_chain(bid=0.40), _cfg(tp=0.60, call_tp=1.0))
    assert acts == ["SELL_CALL"]


def test_the_refusal_is_counted_under_its_own_name():
    st = _held_state()
    _, r = _actions(st, _call_chain(bid=0.0), _cfg(tp=0.60, call_tp=1.0))
    kinds = [w[1] for w in r.warnings]
    assert "call_gated_no_credit" in kinds
    assert "call_gated_unclosable" not in kinds, \
        "a no-credit refusal must not be filed under the A3b counter"


def test_put_entry_refuses_a_zero_bid_when_the_take_profit_is_disabled():
    """The same hole on the put leg. A2 would normally catch it, but the
    liquidity floors default off and a config can turn them off explicitly,
    so the write gate must stand on its own."""
    st = _flat_state()
    acts, _ = _actions(st, _put_chain(bid=0.0), _cfg(tp=1.0))
    assert acts == []


# --------------------------------------------------------------------------
# 3. default-identical at every take-profit the repo actually uses
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tp", [0.25, 0.50, 0.60, 0.90])
@pytest.mark.parametrize("credit", [0.0, 0.004, 0.01, 0.02, 0.025, 0.03,
                                    0.05, 0.10, 0.40, 1.57])
def test_byte_identical_wherever_a_take_profit_exit_exists(tp, credit):
    """Where a TP exit exists, A3b's floor is already at or above one tick,
    so ANDing the write floor in changes no decision anywhere. This is the
    test that lets the change ship without a re-measurement of the live arm."""
    cfg = _cfg(tp=tp)
    before = tp_exit_feasible(credit, cfg)[0]
    after = before and write_credit_ok(credit, cfg)[0]
    assert after == before, (
        f"write floor changed a decision at tp={tp}, credit={credit}: "
        f"A3b said {before}, combined said {after}")


def test_the_live_take_profit_floor_sits_above_the_write_floor():
    """Why the parametrized test above can hold: state the ordering directly,
    so a future change to either floor that inverts it fails loudly here
    rather than silently changing the live arm."""
    from src.engine_v2.options.fills import tp_exit_floor
    cfg = _cfg(tp=0.60)
    assert tp_exit_floor(cfg) > write_credit_floor(cfg)


# --------------------------------------------------------------------------
# 4. wiring -- the zombie-gate guard
# --------------------------------------------------------------------------

@pytest.mark.parametrize("module", [
    "src/engine_v2/options/portfolio.py",
    "src/engine_v2/options/wheel.py",
    "src/engine_v2/options/regime_router.py",
])
def test_every_engine_that_writes_a_short_consults_the_write_floor(module):
    """The earnings_blackout lesson, applied ahead of time: a gate that is on
    but not wired into every engine is worse than no gate, because the run
    prints a full plausible set of trades and nothing looks wrong. Each of
    these three modules opens short positions; each must consult the floor."""
    src = open(module).read()
    assert "write_credit_ok" in src, (
        f"{module} opens short positions but never consults the write-credit "
        f"floor -- tp_exit_feasible alone is not a minimum-credit rule when "
        f"take_profit_pct >= 1.0")
