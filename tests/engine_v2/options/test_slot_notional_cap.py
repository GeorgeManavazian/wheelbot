"""slot_notional_cap_mult: refuse entries whose single-contract notional
outruns the slot budget.

Batch 3, H-B2-1 (spec: the 2026-08-18 pre-registration). The grid's 10-worst
campaigns are all one shape: a high-priced name takes a multiple of its slot's
share of the account, then sits marked down for months. The A12 concentration
fallback exists because idleness was measured to cost more than concentration
-- but it is unbounded: the best-ranked name that fits at ANY concentration
wins, up to the whole free cash on one contract. This knob bounds it: an entry
whose strike x multiplier exceeds cap_mult x the equal-split slot budget is
refused under its own name, in BOTH the normal path and the fallback.

The cap references the EQUAL split (available / empty_slots), not the
fallback's concentrated k-split -- bounding concentration with a budget that
grows as concentration deepens would be the cap eating itself.

None = byte-identical (the unbounded fallback stays exactly as A12 shipped it).
"""
import pandas as pd
import pytest

from src.engine_v2.options.portfolio import (PortfolioState,
                                             run_portfolio_wheel,
                                             step_one_day)
from src.engine_v2.options.wheel import WheelConfig

D = pd.Timestamp("2026-07-21")
EXPIRY = D + pd.Timedelta(days=7)
COLS = ["date", "expiry", "dte", "strike", "right", "bid", "ask", "mid",
        "close", "delta", "iv", "underlying",
        "open_interest", "volume", "bid_size", "ask_size"]


def _chain(strike):
    ch = pd.DataFrame([[D, EXPIRY, 7, strike, "P", 0.90, 1.00, 0.95, 0.95,
                        -0.40, 0.2, strike + 1.0, 500.0, 100.0, 10.0, 10.0]],
                      columns=COLS)
    ch["date"] = pd.to_datetime(ch["date"])
    ch["expiry"] = pd.to_datetime(ch["expiry"])
    return ch


class _M:
    """ticker -> (vol_pctile, strike); each name fillable from its own
    one-row chain."""

    def __init__(self, names):
        self._n = names
        self.universe = list(names)
        self._ch = {tk: _chain(s) for tk, (_, s) in names.items()}

    def chain(self, tk, d):
        return self._ch[tk]

    def spot(self, tk, d, fb):
        return self._n[tk][1] + 1.0

    def settle_price(self, tk, expiry):
        return self._n[tk][1] + 1.0

    def regime_row(self, tk, day):
        return {"vol_pctile": self._n[tk][0], "trend": "uptrend",
                "vol": "calm"}

    def eligible(self, tk, day):
        return True


def _cfg(**kw):
    return WheelConfig(ticker="BIG", starting_capital=100_000.0,
                       put_delta=0.40, call_delta=0.50, target_dte=7,
                       take_profit_pct=0.25, call_min_strike="basis", **kw)


def _step(market, cfg, n_slots=5):
    st = PortfolioState(cash=100_000.0, positions=[])
    r = step_one_day(st, market, D, cfg, selector="plain", n_slots=n_slots)
    return st, r


def _sold(r):
    return [t.contract.root for t in r.trades if t.action == "SELL_PUT"]


#: 100k over 5 slots = 20k equal-split budget; cap at 1.25x = 25k notional.
#: BIG's single contract is 30k -- affordable only by concentrating.
BIG_ONLY = {"BIG": (0.90, 300.0)}
BIG_AND_SMALL = {"BIG": (0.90, 300.0), "SMALL": (0.50, 40.0)}


def test_none_keeps_the_unbounded_fallback_byte_identical():
    """Without the cap, A12's fallback concentrates into BIG at k=3
    (33.3k affords one 30k contract). The pre-existing behavior, pinned."""
    _, r = _step(_M(BIG_ONLY), _cfg())
    assert _sold(r) == ["BIG"]


def test_the_cap_bounds_the_concentration_fallback():
    st, r = _step(_M(BIG_ONLY), _cfg(slot_notional_cap_mult=1.25))
    assert _sold(r) == []
    assert any(w[1] == "entry_gated_notional" for w in r.warnings)
    assert st.idle_slot_days == 5


def test_the_cap_refuses_the_oversized_name_and_promotes_the_next():
    """The refusal never idles a slot another name could fill: BIG is refused
    on notional, SMALL (rank-lower) takes the slot -- same promote-the-next
    discipline as every entry gate."""
    _, r = _step(_M(BIG_AND_SMALL), _cfg(slot_notional_cap_mult=1.25))
    assert "BIG" not in _sold(r)
    assert "SMALL" in _sold(r)
    assert any(w[1] == "entry_gated_notional" and w[2] == "BIG"
               for w in r.warnings)


def test_a_name_inside_the_cap_is_untouched():
    """SMALL's 4k contract sits far under the 25k cap: with only SMALL in the
    universe the cap changes nothing -- it is a ceiling, not a sizer."""
    plain = _step(_M({"SMALL": (0.50, 40.0)}), _cfg())[1]
    capped = _step(_M({"SMALL": (0.50, 40.0)}),
                   _cfg(slot_notional_cap_mult=1.25))[1]
    assert _sold(plain) == _sold(capped) == ["SMALL"]


def test_nonpositive_caps_are_refused_before_any_trade():
    for bad in (0.0, -1.25):
        with pytest.raises(ValueError, match="slot_notional_cap_mult"):
            run_portfolio_wheel({}, _cfg(slot_notional_cap_mult=bad), {},
                                universe=[])
