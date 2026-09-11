"""Collateral-yield floor: refuse a short put that pays a negligible fraction
of the cash it locks up.

The pre-existing minimum credit (`fills.tp_exit_floor`) is ABSOLUTE -- $0.025,
i.e. $2.50 per contract at the live config -- and knows nothing about the
strike. A $1,000-strike put locking $100,000 of collateral clears it on a $2.50
credit. The live bot has already done a version of this: a put sold for a $0.01
bid against a $0.77 mid, locking 50% of a $5k account for 14 days (seen
2026-07-29).

The floor here is a JUNK filter, not a tuning knob. Measured on 25 tickers /
~39.5k ticker-days / DTE 8-15 over 2024-01 -> 2026-07, the median annualised
yield on collateral is 18.3% at 0.20 delta, 30.9% at 0.30 and 45.6% at 0.40;
the least generous legitimate name measured is XLU at 8.4%. A 0.08 floor
therefore sits just under everything a human would knowingly trade while
killing the owner's $10-on-$100k case by two orders of magnitude.

"""
import pandas as pd
import pytest

from src.engine_v2.options.fills import yield_ok
from src.engine_v2.options.portfolio import run_portfolio_wheel
from src.engine_v2.options.wheel import WheelConfig, run_wheel


def _cfg(**kw):
    return WheelConfig(ticker="X", starting_capital=100_000.0, put_delta=0.30,
                       target_dte=11, take_profit_pct=0.60, **kw)


# --- the predicate ----------------------------------------------------------

def test_floor_unset_allows_everything():
    """Default OFF: every pre-existing result must stay byte-identical."""
    cfg = _cfg()
    assert yield_ok(0.10, 1000.0, 11, cfg)[0] is True
    assert yield_ok(0.01, 5.0, 14, cfg)[0] is True


def test_the_owners_case_is_refused():
    """$10 of credit against $100,000 locked for 11 days = 0.33% annualised."""
    cfg = _cfg(min_ann_yield_on_collateral=0.08)
    ok, why = yield_ok(0.10, 1000.0, 11, cfg)
    assert ok is False and why == "yield_below_floor"


def test_a_normal_entry_is_untouched():
    """~$2.00 credit on a $222 strike over 11 days = ~29.9% annualised, which
    is the measured median. The floor must be inert there, or it is not
    removing a tail, it is changing the strategy."""
    cfg = _cfg(min_ann_yield_on_collateral=0.08)
    assert yield_ok(2.00, 222.0, 11, cfg)[0] is True


def test_the_least_generous_real_name_still_passes():
    """XLU measured at 8.4% annualised on 0.30 delta -- the floor sits under
    it deliberately, so no legitimate name is excluded."""
    cfg = _cfg(min_ann_yield_on_collateral=0.08)
    assert yield_ok(0.22, 87.0, 11, cfg)[0] is True     # ~8.4%


def test_boundary_exactly_on_the_floor_is_allowed():
    cfg = _cfg(min_ann_yield_on_collateral=0.08)
    # dte 365 makes the annualisation factor exactly 1.0
    assert yield_ok(8.00, 100.0, 365, cfg)[0] is True   # exactly 8%: allowed
    assert yield_ok(7.99, 100.0, 365, cfg)[0] is False  # under: refused


def test_nonpositive_strike_is_refused_not_divided_by():
    cfg = _cfg(min_ann_yield_on_collateral=0.08)
    ok, why = yield_ok(1.00, 0.0, 11, cfg)
    assert ok is False and why == "bad_strike"


def test_nonpositive_dte_is_refused_not_divided_by():
    cfg = _cfg(min_ann_yield_on_collateral=0.08)
    ok, why = yield_ok(1.00, 100.0, 0, cfg)
    assert ok is False and why == "bad_dte"


# --- wired into the engine --------------------------------------------------

def _chain(credit):
    """One ticker, one expiry, one $1,000 strike -- the owner's case. At
    credit 0.10 that is $10 against $100,000 of collateral."""
    strike, rows = 1000.0, []
    exp = pd.Timestamp("2024-01-16")
    for day in pd.bdate_range("2024-01-02", periods=12):
        rows.append(dict(date=day, expiry=exp, dte=(exp - day).days,
                         strike=strike, right="P", bid=credit,
                         ask=credit + 0.02, mid=credit + 0.01, close=0.0,
                         delta=-0.30, iv=0.3, underlying=strike * 1.02))
    return pd.DataFrame(rows)


def test_engine_opens_when_the_yield_is_ordinary():
    """$8 credit on a $1,000 strike over ~11 days is ~26% annualised."""
    res = run_wheel(_chain(8.00), _cfg(min_ann_yield_on_collateral=0.08))
    assert any(t.action == "SELL_PUT" for t in res.trades)


def test_engine_refuses_the_owners_case():
    res = run_wheel(_chain(0.10), _cfg(min_ann_yield_on_collateral=0.08))
    assert not any(t.action == "SELL_PUT" for t in res.trades)


def test_the_refusal_is_never_silent():
    """A2/A3 precedent: a gated day must be distinguishable from a no-weather
    day, or days_flat attribution silently lies."""
    res = run_wheel(_chain(0.10), _cfg(min_ann_yield_on_collateral=0.08))
    kinds = {w[1] for w in (res.warnings or [])}
    assert "entry_gated_low_yield" in kinds


def test_same_chain_trades_freely_with_the_floor_off():
    """Proves the refusal is the FLOOR, not something else about the fixture."""
    res = run_wheel(_chain(0.10), _cfg())
    assert any(t.action == "SELL_PUT" for t in res.trades)


# --- wired into the PORTFOLIO engine ----------------------------------------
# This is the path the diagnostic grid runs. Wiring the solo wheel alone would
# leave the grid ungated. selector="vol_pctile" with no regime states keeps the
# fixture free of weather setup: an unknown row always allows, so the only
# thing that can refuse an entry here is the yield floor.

def _portfolio(credit, **kw):
    chains = {"X": _chain(credit)}
    return run_portfolio_wheel(chains, _cfg(**kw), {"X": pd.DataFrame()},
                               selector="vol_pctile", n_slots=1,
                               universe=["X"])


def test_portfolio_opens_when_the_yield_is_ordinary():
    res = _portfolio(8.00, min_ann_yield_on_collateral=0.08)
    assert any(t.action == "SELL_PUT" for t in res.trades)


def test_portfolio_refuses_the_owners_case():
    res = _portfolio(0.10, min_ann_yield_on_collateral=0.08)
    assert not any(t.action == "SELL_PUT" for t in res.trades)


def test_portfolio_refusal_is_never_silent_and_is_deduped():
    """The n_slots while-loop revisits gated tickers every iteration, so an
    undeduped warning would emit once per pass, not once per day."""
    res = _portfolio(0.10, min_ann_yield_on_collateral=0.08)
    gated = [w for w in (res.warnings or []) if w[1] == "entry_gated_low_yield"]
    assert gated, "the veto must not be silent"
    assert len(gated) == len(set(gated)), "warnings must be deduped"


def test_portfolio_trades_freely_with_the_floor_off():
    res = _portfolio(0.10)
    assert any(t.action == "SELL_PUT" for t in res.trades)
