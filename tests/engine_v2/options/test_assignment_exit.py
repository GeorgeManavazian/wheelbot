"""The assigned-stock exit ladder (spec 2026-08-14).

The portfolio wheel had no exit: once assigned, the only way out was a covered
call, call_min_strike="basis" forbade writing one below net basis, and
run_portfolio_wheel refuses liquidate_assignment as a solo mechanic -- so no
path sold shares at all. Measured on 530 tickers over 2022-06..2026-08, 1,347
of 5,205 slot-days held assigned shares with no call against them, 1,047 of
them refused by A3b rather than by the floor directly.

    tier 1  floor holds           OTM call at call_delta
    tier 2  tier 1 IMPOSSIBLE     ITM call at exit_call_delta
    tier 3  spot < the tier-2 K   close the call, THEN sell the shares

The invariant that outranks every other requirement here: the account has no
margin, so a short call must never outlive the shares covering it.
"""
import pandas as pd
import pytest

from src.engine_v2.options.portfolio import PortfolioState, step_one_day
from src.engine_v2.options.wheel import WheelConfig

D = pd.Timestamp("2026-07-21")
EXP = D + pd.Timedelta(days=11)
COLS = ["date", "expiry", "dte", "strike", "right", "bid", "ask", "mid",
        "close", "delta", "iv", "underlying",
        "open_interest", "volume", "bid_size", "ask_size"]


def _chain(spot=80.0, strikes=None, call_bid=None):
    """A two-sided ladder around `spot`. Call delta falls as strike rises, so
    select_contract's delta targeting has something real to choose from."""
    strikes = strikes or [60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0]
    rows = []
    for k in strikes:
        # crude but monotone: deep ITM calls ~0.95, far OTM ~0.05
        cdelta = max(0.02, min(0.98, 0.5 + (spot - k) / (spot * 0.55)))
        intrinsic = max(0.0, spot - k)
        cb = call_bid if call_bid is not None else round(intrinsic + 0.60, 2)
        rows.append([D, EXP, 11, k, "C", cb, round(cb + 0.05, 2),
                     round(cb + 0.02, 2), round(cb + 0.02, 2), cdelta, 0.30,
                     spot, 5000.0, 400.0, 10.0, 10.0])
        pintrinsic = max(0.0, k - spot)
        pb = round(pintrinsic + 0.60, 2)
        rows.append([D, EXP, 11, k, "P", pb, round(pb + 0.05, 2),
                     round(pb + 0.02, 2), round(pb + 0.02, 2), -(1 - cdelta),
                     0.30, spot, 5000.0, 400.0, 10.0, 10.0])
    ch = pd.DataFrame(rows, columns=COLS)
    ch["date"] = pd.to_datetime(ch["date"])
    ch["expiry"] = pd.to_datetime(ch["expiry"])
    return ch


class M:
    """Single-name market. `spot` is settable so a decline can be simulated.

    `allow_entry=False` makes the name ineligible for NEW puts, which isolates
    the exit under test: a freed slot is refilled the same session by design
    (see test_tier3_frees_the_slot_for_immediate_reuse), and that refill would
    otherwise mask what tier 3 did to cash and positions.
    """
    universe = ["AA"]

    def __init__(self, ch, spot=80.0, allow_entry=True):
        self._ch, self._spot, self._allow = ch, spot, allow_entry

    def chain(self, tk, d):
        return self._ch

    def spot(self, tk, d, fb):
        return self._spot

    def settle_price(self, tk, expiry):
        return self._spot

    def regime_row(self, tk, day):
        return None

    def eligible(self, tk, day):
        return self._allow


def _cfg(**over):
    base = dict(ticker="AA", starting_capital=100_000.0, put_delta=0.30,
                call_delta=0.50, target_dte=11, take_profit_pct=0.60,
                call_min_strike="basis")
    base.update(over)
    return WheelConfig(**base)


def _held(shares=100, basis=100.0, premium=0.0, short=None, **extra):
    """A position already assigned into shares -- the state the ladder acts on."""
    pos = {"ticker": "AA", "shares": shares, "phase": "CALL", "basis": basis,
           "premium": premium, "campaign": 1, "last_spot": 80.0, "short": short}
    pos.update(extra)
    return pos


def _step(cfg, pos, spot=80.0, chain=None, cash=50_000.0, allow_entry=True):
    st = PortfolioState(cash=cash, positions=[pos])
    mkt = M(chain if chain is not None else _chain(spot=spot), spot=spot,
            allow_entry=allow_entry)
    r = step_one_day(st, mkt, D, cfg, selector="plain", n_slots=1)
    return st, r


# --- tier 2 -----------------------------------------------------------------

def test_tier2_does_not_fire_while_a_floor_compliant_call_is_writable():
    """basis 78 against spot 80: strikes at/above 78 exist and pay real
    premium, so tier 1 works and the ladder must stay out of the way."""
    cfg = _cfg(assignment_exit=True, exit_call_delta=0.80)
    st, r = _step(cfg, _held(basis=78.0), spot=80.0)
    assert [t.action for t in r.trades] == ["SELL_CALL"]
    assert st.positions[0]["short"]["contract"].strike >= 78.0
    assert st.positions[0].get("exit_strike") is None, \
        "tier 2 stamped an exit strike on a position tier 1 handled fine"


def test_tier2_fires_when_the_floor_call_is_unwritable_and_writes_itm():
    """basis 200 against spot 80: the floor sits above every listed strike, so
    tier 1 is impossible and tier 2 must write BELOW spot at exit_call_delta."""
    cfg = _cfg(assignment_exit=True, exit_call_delta=0.80)
    st, r = _step(cfg, _held(basis=200.0), spot=80.0)
    assert [t.action for t in r.trades] == ["SELL_CALL"]
    k = st.positions[0]["short"]["contract"].strike
    assert k < 80.0, f"tier 2 must write ITM, wrote {k} against spot 80"


def test_tier2_stamps_the_exit_strike_on_the_position():
    cfg = _cfg(assignment_exit=True, exit_call_delta=0.80)
    st, r = _step(cfg, _held(basis=200.0), spot=80.0)
    assert st.positions[0]["exit_strike"] == \
        st.positions[0]["short"]["contract"].strike


def test_tier2_is_off_when_the_ladder_is_off():
    """Without assignment_exit the unwritable floor stays unwritable -- today's
    behaviour, and the reason 1,347 slot-days sat uncovered."""
    cfg = _cfg(assignment_exit=False)
    st, r = _step(cfg, _held(basis=200.0), spot=80.0)
    assert [t.action for t in r.trades] == []
    assert any(w[1] == "covered_call_unreachable" for w in r.warnings)


# --- tier 3 -----------------------------------------------------------------

def test_tier3_sells_the_shares_when_spot_falls_below_the_exit_strike():
    cfg = _cfg(assignment_exit=True, exit_stop="spot_below_strike")
    pos = _held(basis=200.0, exit_strike=70.0)
    st, r = _step(cfg, pos, spot=65.0, allow_entry=False)
    assert "SOLD_SHARES" in [t.action for t in r.trades]
    assert st.positions == [], "a fully exited position must free its slot"


def test_tier3_frees_the_slot_for_immediate_reuse():
    """The point of the ladder: a stopped-out slot goes back to work the same
    session. With no cooldown the router re-enters at once, which is why the
    isolation tests above have to switch entry off to see the exit cleanly."""
    cfg = _cfg(assignment_exit=True, exit_stop="spot_below_strike")
    pos = _held(basis=200.0, exit_strike=70.0)
    st, r = _step(cfg, pos, spot=65.0, allow_entry=True)
    actions = [t.action for t in r.trades]
    assert "SOLD_SHARES" in actions
    assert "SELL_PUT" in actions, "the freed slot was not put back to work"
    assert all(p["shares"] == 0 for p in st.positions)


def test_exit_cooldown_blocks_same_day_reentry():
    cfg = _cfg(assignment_exit=True, exit_stop="spot_below_strike",
               exit_cooldown_days=5)
    pos = _held(basis=200.0, exit_strike=70.0)
    st, r = _step(cfg, pos, spot=65.0, allow_entry=True)
    actions = [t.action for t in r.trades]
    assert "SOLD_SHARES" in actions
    assert "SELL_PUT" not in actions, "cooldown did not stop the re-entry"
    assert any(w[1] == "entry_in_exit_cooldown" for w in r.warnings)


def test_tier3_does_not_fire_while_spot_holds_above_the_exit_strike():
    cfg = _cfg(assignment_exit=True, exit_stop="spot_below_strike")
    pos = _held(basis=200.0, exit_strike=70.0)
    st, r = _step(cfg, pos, spot=75.0)
    assert "SOLD_SHARES" not in [t.action for t in r.trades]
    assert st.positions and st.positions[0]["shares"] == 100


def test_tier3_closes_the_short_before_selling_never_naked():
    """THE invariant. The account has no margin: a short call must never
    outlive the shares covering it, not even within a single step."""
    cfg = _cfg(assignment_exit=True, exit_stop="spot_below_strike")
    short = {"contract": type("K", (), {"root": "AA", "expiry": EXP,
                                        "strike": 70.0, "right": "C"})(),
             "contracts": 1, "credit": 1.50, "last_mid": 0.20}
    pos = _held(basis=200.0, exit_strike=70.0, short=short)
    st, r = _step(cfg, pos, spot=65.0, allow_entry=False)
    actions = [t.action for t in r.trades]
    assert "SOLD_SHARES" in actions
    assert "CLOSE_CALL" in actions
    assert actions.index("CLOSE_CALL") < actions.index("SOLD_SHARES"), \
        "shares were sold before the covering call was bought back"
    assert st.positions == []


def test_tier3_refuses_to_sell_when_the_short_cannot_be_closed():
    """Fail closed. An unquotable call means the slot stays stuck one more day
    -- which is the problem we already have -- rather than becoming an
    uncovered short call, which is unlimited liability."""
    cfg = _cfg(assignment_exit=True, exit_stop="spot_below_strike")
    # a strike the chain does not list -> option_mark finds nothing
    short = {"contract": type("K", (), {"root": "AA", "expiry": EXP,
                                        "strike": 61.5, "right": "C"})(),
             "contracts": 1, "credit": 1.50, "last_mid": 0.20}
    pos = _held(basis=200.0, exit_strike=70.0, short=short)
    st, r = _step(cfg, pos, spot=65.0)
    assert "SOLD_SHARES" not in [t.action for t in r.trades]
    assert st.positions[0]["shares"] == 100
    assert any(w[1] == "exit_blocked_unclosable_call" for w in r.warnings)


def test_tier3_books_proceeds_at_the_days_close():
    cfg = _cfg(assignment_exit=True, exit_stop="spot_below_strike")
    pos = _held(shares=200, basis=200.0, exit_strike=70.0)
    st, r = _step(cfg, pos, spot=65.0, cash=1_000.0, allow_entry=False)
    sale = [t for t in r.trades if t.action == "SOLD_SHARES"][0]
    assert sale.contracts == 200
    assert sale.price_per_contract == pytest.approx(65.0)
    assert st.cash == pytest.approx(1_000.0 + 200 * 65.0)


def test_exit_strike_survives_the_take_profit_closing_the_tier2_call():
    """When the stock falls the tier-2 call goes OTM and cheap, so the 60% TP
    buys it back first. The trigger is stamped on the POSITION precisely so it
    is not lost at the moment it is needed."""
    cfg = _cfg(assignment_exit=True, exit_stop="spot_below_strike")
    short = {"contract": type("K", (), {"root": "AA", "expiry": EXP,
                                        "strike": 70.0, "right": "C"})(),
             "contracts": 1, "credit": 10.00, "last_mid": 0.60}
    pos = _held(basis=200.0, exit_strike=70.0, short=short)
    st, r = _step(cfg, pos, spot=65.0)
    actions = [t.action for t in r.trades]
    assert "CLOSE_CALL" in actions and "SOLD_SHARES" in actions


def test_exit_stop_none_leaves_tier2_running_with_no_hard_stop():
    cfg = _cfg(assignment_exit=True, exit_stop="none")
    pos = _held(basis=200.0, exit_strike=70.0)
    st, r = _step(cfg, pos, spot=40.0)
    assert "SOLD_SHARES" not in [t.action for t in r.trades]


def test_pct_below_basis_stop_fires_without_any_exit_strike():
    """The plain stop-loss control: ignores the ladder entirely so the tiered
    design can be benchmarked against it."""
    cfg = _cfg(assignment_exit=True, exit_stop="pct_below_basis",
               exit_stop_buffer=0.20)
    pos = _held(basis=100.0, premium=0.0)         # net basis 100
    st, r = _step(cfg, pos, spot=75.0)            # 25% below
    assert "SOLD_SHARES" in [t.action for t in r.trades]


# --- the anchor -------------------------------------------------------------

def test_ladder_off_by_default_is_byte_identical():
    cfg = _cfg()
    assert cfg.assignment_exit is False
    st, r = _step(cfg, _held(basis=78.0), spot=80.0)
    assert [t.action for t in r.trades] == ["SELL_CALL"]
    assert st.positions[0].get("exit_strike") is None


# --- standing invariants, independent of this feature -----------------------

def test_standing_invariant_every_written_call_is_fully_share_covered():
    """Pins in code the guarantee the engine currently provides only
    structurally: contracts * multiplier never exceeds shares held."""
    for basis, spot in ((78.0, 80.0), (200.0, 80.0), (50.0, 80.0)):
        cfg = _cfg(assignment_exit=True)
        st, r = _step(cfg, _held(shares=250, basis=basis), spot=spot)
        for t in r.trades:
            if t.action == "SELL_CALL":
                assert t.contracts * cfg.contract_multiplier <= 250, \
                    f"naked call: {t.contracts} contracts against 250 shares"


def test_standing_invariant_no_short_call_outlives_its_shares():
    cfg = _cfg(assignment_exit=True, exit_stop="spot_below_strike")
    short = {"contract": type("K", (), {"root": "AA", "expiry": EXP,
                                        "strike": 70.0, "right": "C"})(),
             "contracts": 1, "credit": 1.50, "last_mid": 0.20}
    pos = _held(basis=200.0, exit_strike=70.0, short=short)
    st, r = _step(cfg, pos, spot=65.0)
    for p in st.positions:
        if p.get("short") is not None and p["short"]["contract"].right == "C":
            assert p["short"]["contracts"] * 100 <= p["shares"], \
                "a short call outlived the shares covering it"


# --- config validation ------------------------------------------------------

def test_unknown_exit_trigger_is_refused_loudly():
    from src.engine_v2.options.portfolio import run_portfolio_wheel
    cfg = _cfg(assignment_exit=True, exit_trigger="vibes")
    with pytest.raises(ValueError, match="exit_trigger"):
        run_portfolio_wheel({}, cfg, {}, universe=[])


def test_unknown_exit_stop_is_refused_loudly():
    from src.engine_v2.options.portfolio import run_portfolio_wheel
    cfg = _cfg(assignment_exit=True, exit_stop="vibes")
    with pytest.raises(ValueError, match="exit_stop"):
        run_portfolio_wheel({}, cfg, {}, universe=[])


# --- exit_trigger="never": tier 3 without tier 2 ----------------------------

def test_trigger_never_leaves_tier2_off_but_tier3_armed():
    """The stuck-campaign review (2026-08-14) found the tail is what costs:
    4 of 40 assignments never resolved and consumed 51% of all held-days, with
    PFE held 1,309 days at -57%. That argues for a hard stop that fires deep
    and rarely, with the basis floor otherwise untouched -- which needs tier 3
    WITHOUT tier 2's ITM calls."""
    cfg = _cfg(assignment_exit=True, exit_trigger="never",
               exit_stop="pct_below_basis", exit_stop_buffer=0.40)
    # net basis 110 against spot 80 and a ladder topping out at 100: the floor
    # IS unreachable (so tier 2 would fire if enabled) but the position is only
    # 27% underwater, so the -40% deep stop must NOT fire either. That gap is
    # the whole point of "never" -- keep the floor's patience, add only a tail
    # stop -- and a fixture that fires the stop would prove nothing about it.
    st, r = _step(cfg, _held(basis=110.0), spot=80.0, allow_entry=False)
    assert [t.action for t in r.trades] == [], \
        f"expected no action, got {[t.action for t in r.trades]}"
    assert st.positions[0].get("exit_strike") is None
    assert st.positions[0]["shares"] == 100, "the deep stop fired too early"
    assert any(w[1] == "covered_call_unreachable" for w in r.warnings), \
        "fixture no longer exercises an unreachable floor"


def test_trigger_never_still_lets_the_deep_stop_fire():
    cfg = _cfg(assignment_exit=True, exit_trigger="never",
               exit_stop="pct_below_basis", exit_stop_buffer=0.40)
    pos = _held(basis=100.0, premium=0.0)      # net basis 100
    st, r = _step(cfg, pos, spot=55.0, allow_entry=False)   # 45% below
    assert "SOLD_SHARES" in [t.action for t in r.trades]
    assert st.positions == []
