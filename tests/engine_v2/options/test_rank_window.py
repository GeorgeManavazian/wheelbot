"""rank_window: select entries from ranks [a, b] of the sorted pool.

Batch 3, H-B2-3 (spec: the 2026-08-18 pre-registration). The wheel has always
taken the TOP of the ranked pool; whether that ranking carries any signal has
never been measured. The honest instrument is a SLICE, not a veto: run the
same machine on ranks 6-10 (and 11-15) and compare. If the slice lands inside
the book-noise band of ranks 1-5, the selection axis is inert (the C12
shape); if it is materially worse, ranking has signal.

Semantics pinned here:
  * None -> byte-identical to today (the top of the list).
  * [a, b] is 1-indexed and inclusive, applied to the sorted pool EVERY
    fill-iteration -- the pool is rebuilt without held names each time, so
    the window slides over the CURRENT ranking, exactly as "the 6th-best
    name right now" means.
  * A pool shorter than `a` fills nothing: the slot stays honestly idle and
    the day is counted under `entry_rank_window_empty` -- a window that
    starves is a finding, never a silent fallback to ranks it was told to
    avoid.
  * The affordability fallback searches only the sliced pool: the window is
    a hard scope, or the mechanic silently becomes top-of-list under memory
    pressure, which is the zombie-gate shape.
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
    same one-row chain (fixture style of test_iv_rank_ranking)."""

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
    _, r = _step(_M(FOUR), _cfg())
    assert _sold(r) == ["A"]


def test_window_takes_the_ath_ranked_name_one_indexed():
    _, r = _step(_M(FOUR), _cfg(rank_window=[2, 3]))
    assert _sold(r) == ["B"]


def test_window_slides_over_the_reranked_pool_per_fill():
    """n_slots=2 with window [2,2]: the first fill takes today's 2nd-best (B);
    the pool is then rebuilt without B, whose 2nd-best is C. A window frozen
    on the first ranking would try B twice and fill A-something instead."""
    _, r = _step(_M(FOUR), _cfg(rank_window=[2, 2]), n_slots=2)
    assert _sold(r) == ["B", "C"]


def test_a_window_beyond_the_pool_starves_loudly_never_falls_back():
    st, r = _step(_M(FOUR), _cfg(rank_window=[6, 10]))
    assert _sold(r) == []
    assert any(w[1] == "entry_rank_window_empty" for w in r.warnings)
    assert st.idle_slot_days == 1


def test_the_window_clips_at_the_pool_end():
    """[3, 10] over four names is ranks 3-4, not an error and not a starve."""
    _, r = _step(_M(FOUR), _cfg(rank_window=[3, 10]))
    assert _sold(r) == ["C"]


def test_malformed_windows_are_refused_before_any_trade():
    for bad in ([3, 1], [0, 5], [2], [1, 2, 3]):
        with pytest.raises(ValueError, match="rank_window"):
            run_portfolio_wheel({}, _cfg(rank_window=bad), {}, universe=[])
