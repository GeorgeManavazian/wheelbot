"""Intrinsic filter: refuse a short put whose credit is mostly moneyness.

Found on the 2.53y / $500k / 338-ticker run: 2 of 299 entries had a credit
above 3% of strike -- NOK at 37.2% (a $6.50 put sold for $2.42 with the stock
near $4.15, i.e. ~95% intrinsic) and HL at 9.4%. Those two carried -$47,150 of
the -$64,650 total share-leg loss: 73% of the damage from 0.7% of the trades.

The wheel is paid for taking assignment RISK. When the credit is intrinsic,
assignment is near-certain and already in the price -- it is a stock purchase
booked as "premium collected", which also inflates the premium and win-rate
figures the strategy is judged on.
"""
import pandas as pd
import pytest

from src.engine_v2.options.chain import Contract
from src.engine_v2.options.fills import credit_ok
from src.engine_v2.options.wheel import WheelConfig, run_wheel


def _cfg(**kw):
    return WheelConfig(ticker="X", starting_capital=100_000.0, put_delta=0.20,
                       target_dte=7, take_profit_pct=None, **kw)


# --- the predicate ----------------------------------------------------------

def test_cap_unset_allows_everything():
    """Default OFF: every pre-existing result must stay byte-identical."""
    cfg = _cfg()
    assert credit_ok(2.42, 6.50, cfg)[0] is True      # the NOK trade
    assert credit_ok(0.03, 6.50, cfg)[0] is True


def test_the_nok_trade_is_refused_at_3pct():
    cfg = _cfg(max_credit_pct_of_strike=0.03)
    ok, why = credit_ok(2.42, 6.50, cfg)              # 37.2% of strike
    assert ok is False and why == "credit_is_intrinsic"


def test_the_hl_trade_is_refused_at_3pct():
    cfg = _cfg(max_credit_pct_of_strike=0.03)
    assert credit_ok(0.61, 6.50, cfg)[0] is False     # 9.4% of strike


def test_the_median_entry_is_untouched():
    """Median real entry is 0.52% of strike -- a 3% cap must be inert there,
    or the filter is not removing the tail, it is changing the strategy."""
    cfg = _cfg(max_credit_pct_of_strike=0.03)
    assert credit_ok(0.52, 100.0, cfg)[0] is True
    assert credit_ok(2.99, 100.0, cfg)[0] is True     # just inside


def test_boundary_is_inclusive_of_the_cap():
    cfg = _cfg(max_credit_pct_of_strike=0.03)
    assert credit_ok(3.00, 100.0, cfg)[0] is True     # exactly at cap: allowed
    assert credit_ok(3.01, 100.0, cfg)[0] is False    # over: refused


def test_nonpositive_strike_is_refused_not_divided_by():
    cfg = _cfg(max_credit_pct_of_strike=0.03)
    ok, why = credit_ok(1.0, 0.0, cfg)
    assert ok is False and why == "bad_strike"


# --- wired into the engine --------------------------------------------------

def _chain(credit_pct):
    """One ticker, one expiry, one strike, priced at `credit_pct` of strike."""
    strike, rows = 100.0, []
    for day in pd.bdate_range("2024-01-02", periods=12):
        bid = strike * credit_pct
        rows.append(dict(date=day, expiry=pd.Timestamp("2024-01-16"),
                         dte=(pd.Timestamp("2024-01-16") - day).days,
                         strike=strike, right="P", bid=bid, ask=bid + 0.02,
                         mid=bid + 0.01, close=0.0, delta=-0.20, iv=0.3,
                         underlying=strike * 0.98))
    return pd.DataFrame(rows)


def test_engine_opens_when_credit_is_ordinary():
    res = run_wheel(_chain(0.005), _cfg(max_credit_pct_of_strike=0.03))
    assert any(t.action == "SELL_PUT" for t in res.trades)


def test_engine_refuses_when_credit_is_intrinsic():
    res = run_wheel(_chain(0.20), _cfg(max_credit_pct_of_strike=0.03))
    assert not any(t.action == "SELL_PUT" for t in res.trades)


def test_the_refusal_is_never_silent():
    """A2/A3 precedent: a gated day must be distinguishable from a no-weather
    day, or days_flat attribution silently lies."""
    res = run_wheel(_chain(0.20), _cfg(max_credit_pct_of_strike=0.03))
    kinds = {w[1] for w in (res.warnings or [])}
    assert "entry_gated_intrinsic" in kinds


def test_same_chain_trades_freely_with_the_cap_off():
    """Proves the refusal is the CAP, not something else about the fixture."""
    res = run_wheel(_chain(0.20), _cfg())
    assert any(t.action == "SELL_PUT" for t in res.trades)
