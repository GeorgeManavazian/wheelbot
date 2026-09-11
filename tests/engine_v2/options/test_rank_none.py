"""rank_by="none": the sort key becomes a constant; universe tie-order decides.

Batch 4 (spec: the batch-3 close pre-registration, owner gate opened
2026-08-18). The clean inertness control the batch-3 rank slices could not be:
those confounded ordering with starvation (a 5-name window against an
11.7-deep pool). This DELETES the vol_pctile sort instead -- gate-passers fill
slots in fixed universe order, every candidate stays in the pool, nothing ever
starves. Whatever it measures is ordering signal alone.

Semantics pinned here:
  * "none" is a closed-set value on the existing rank_by knob, refused
    anywhere else (the typo'd-sort-key stance of run_portfolio_wheel).
  * The sort key becomes a constant, so the pool tuple's second element --
    `market.universe.index(tk)`, the FIXED universe order -- decides alone.
    T3: universe order is itself an ordering; it is pinned to the universe
    list exactly as loaded, never to pool-build order or dict order, so it
    is inspectable from the universe file alone.
  * Every gate still applies: this deletes the sort, never a veto.
  * Existing rank_by values are untouched (vol_pctile default asserted here).
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


def _chain():
    ch = pd.DataFrame([[D, EXPIRY, 7, 40.0, "P", 0.90, 1.00, 0.95, 0.95,
                        -0.40, 0.2, 41.0, 500.0, 100.0, 10.0, 10.0]],
                      columns=COLS)
    ch["date"] = pd.to_datetime(ch["date"])
    ch["expiry"] = pd.to_datetime(ch["expiry"])
    return ch


class _M:
    """Multi-ticker fake: ticker -> vol_pctile, every name fillable from the
    same one-row chain (fixture style of test_rank_window)."""

    def __init__(self, pctiles, ineligible=()):
        self._p = pctiles
        self.universe = list(pctiles)
        self._ch = _chain()
        self._out = set(ineligible)

    def chain(self, tk, d):
        return self._ch

    def spot(self, tk, d, fb):
        return 41.0

    def settle_price(self, tk, expiry):
        return 41.0

    def regime_row(self, tk, day):
        return {"vol_pctile": self._p[tk], "trend": "uptrend", "vol": "calm"}

    def eligible(self, tk, day):
        return tk not in self._out


#: Universe order A, B, C, D -- but ranked D > C > B > A by vol_pctile, so the
#: two orderings DISAGREE on every name and the tests cannot pass by accident.
FOUR = {"A": 0.60, "B": 0.70, "C": 0.80, "D": 0.90}


def _cfg(**kw):
    return WheelConfig(ticker="A", starting_capital=100_000.0, put_delta=0.40,
                       call_delta=0.50, target_dte=7, take_profit_pct=0.25,
                       call_min_strike="basis", **kw)


def _step(market, cfg, n_slots=1):
    st = PortfolioState(cash=100_000.0, positions=[])
    r = step_one_day(st, market, D, cfg, selector="plain", n_slots=n_slots)
    return st, r


def _sold(r):
    return [t.contract.root for t in r.trades if t.action == "SELL_PUT"]


def test_the_value_is_accepted_by_run_portfolio_wheel():
    # No trades on an empty universe -- the point is only that validation
    # admits the closed-set value instead of raising.
    run_portfolio_wheel({}, _cfg(rank_by="none"), {}, universe=[])


def test_fills_in_universe_order_not_rank_order():
    _, r = _step(_M(FOUR), _cfg(rank_by="none"), n_slots=2)
    assert _sold(r) == ["A", "B"]      # vol_pctile would say ["D", "C"]


def test_universe_order_among_gate_passers_only():
    """A name a gate removes is skipped, never substituted by rank: with A
    ineligible the first universe survivor (B) fills, not the top-ranked D."""
    _, r = _step(_M(FOUR, ineligible={"A"}), _cfg(rank_by="none"))
    assert _sold(r) == ["B"]


def test_default_vol_pctile_ordering_is_unaffected():
    _, r = _step(_M(FOUR), _cfg(), n_slots=2)
    assert _sold(r) == ["D", "C"]


def test_a_typoed_sort_key_is_still_refused():
    # "None"/"" are the plausible typos for this value; neither may fall back.
    for bad in ("None", "NONE", "", "universe"):
        with pytest.raises(ValueError, match="rank_by"):
            run_portfolio_wheel({}, _cfg(rank_by=bad), {}, universe=[])
