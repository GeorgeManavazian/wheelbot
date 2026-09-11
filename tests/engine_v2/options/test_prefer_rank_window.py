"""prefer_rank_window: ranks [a, b] of the sorted pool move to the FRONT.

Batch 4 (spec: the batch-3 close pre-registration, owner gate opened
2026-08-18). The PREFERENCE form of the batch-3 rank slice, starvation-free
by construction: the batch-3 windows vetoed everything outside their slice
and starved 86-229 days; this re-orders and never removes -- exactly the
prefer_chop_half shape, so any gap it shows against frozen IS ordering
signal, clean of the window_empty confound.

Semantics pinned here:
  * None -> byte-identical to today (load-bearing: frozen must not move).
  * [a, b] is 1-indexed and inclusive over the SORTED pool: ranks a..b move
    to the front of the queue, ordinary rank order INSIDE each tier, and the
    rest of the pool follows in ordinary rank order -- full-pool fallback,
    so a pool shorter than `a` is simply unchanged and a preferred tier that
    runs out degrades to today's order. A preference cannot starve.
  * Re-applied to the re-ranked pool every fill-iteration, like rank_window.
  * Malformed windows are refused before any trade (band-knob stance).
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

    def __init__(self, pctiles):
        self._p = pctiles
        self.universe = list(pctiles)
        self._ch = _chain()

    def chain(self, tk, d):
        return self._ch

    def spot(self, tk, d, fb):
        return 41.0

    def settle_price(self, tk, expiry):
        return 41.0

    def regime_row(self, tk, day):
        return {"vol_pctile": self._p[tk], "trend": "uptrend", "vol": "calm"}

    def eligible(self, tk, day):
        return True


#: Ranked A > B > C > D by vol_pctile (the default sort key).
FOUR = {"A": 0.90, "B": 0.80, "C": 0.70, "D": 0.60}


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


def test_none_is_byte_identical_top_of_list():
    _, r = _step(_M(FOUR), _cfg(), n_slots=2)
    assert _sold(r) == ["A", "B"]


def test_the_preferred_slice_moves_to_the_front_one_indexed():
    _, r = _step(_M(FOUR), _cfg(prefer_rank_window=[2, 3]))
    assert _sold(r) == ["B"]


def test_rank_order_inside_each_tier_full_pool_behind():
    """[2,3] over A>B>C>D re-applied per fill on the rebuilt pool:
    ABCD -> B first; ACD's ranks 2-3 are C,D -> C; AD's rank 2 is D -> D;
    A alone falls below the window, unchanged pool -> A. Every name trades:
    a preference re-orders the whole queue and never starves."""
    _, r = _step(_M(FOUR), _cfg(prefer_rank_window=[2, 3]), n_slots=4)
    assert _sold(r) == ["B", "C", "D", "A"]


def test_a_window_beyond_the_pool_is_a_no_op_never_a_starve():
    st, r = _step(_M(FOUR), _cfg(prefer_rank_window=[6, 10]))
    assert _sold(r) == ["A"]           # unchanged order, slot filled
    assert not any(w[1] == "entry_rank_window_empty" for w in r.warnings)
    assert st.idle_slot_days == 0


def test_the_preferred_tier_clips_at_the_pool_end():
    """[3, 10] over four names prefers ranks 3-4: C then D, then 1-2 behind."""
    _, r = _step(_M(FOUR), _cfg(prefer_rank_window=[3, 10]), n_slots=2)
    assert _sold(r) == ["C", "D"]


def test_composes_with_prefer_chop_half_ordering():
    """Applied AFTER the sort branches: with prefer_chop_half on, the rank
    window re-orders the half-sorted queue, not the raw vol_pctile order.
    Here every name is in the same half, so the combined order equals the
    plain [2,3] preference -- the point is that the combination runs at all
    and stays a pure re-ordering."""
    _, r = _step(_M(FOUR), _cfg(prefer_chop_half="B", prefer_rank_window=[2, 3]))
    assert _sold(r) == ["B"]


def test_malformed_windows_are_refused_before_any_trade():
    for bad in ([3, 1], [0, 5], [2], [1, 2, 3]):
        with pytest.raises(ValueError, match="prefer_rank_window"):
            run_portfolio_wheel({}, _cfg(prefer_rank_window=bad), {},
                                universe=[])
