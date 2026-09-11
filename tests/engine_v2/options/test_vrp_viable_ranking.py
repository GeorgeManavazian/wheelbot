"""`rank_by="vrp_viable"`: a two-tier reorder, not a new instrument.

BATCH 5 ARM 2 (2026-08-22). The vrp-ranker verdict (rejected 2026-08-17)
found a real, un-falsified statistic -- IV/RV runs 1.620 -> 0.998 monotone
from calmest to jumpiest realized-vol decile -- and died on slot economics:
a pure VRP sort prefers calm names, calm names pay thin credits, and thin
credits cannot clear the collateral-yield floor often enough (campaigns
142 -> 100). This arm holds slot economics fixed and asks only the
residual question: among names whose premium ALREADY clears a real floor,
does VRP reorder them profitably?

THE DESIGN. One pinned constant (`vrp.VRP_VIABLE_MIN_ANN_YIELD = 0.30`,
never a knob, never swept): candidates whose annualized credit yield on
collateral clears it sort by VRP descending, front of the queue; everyone
else follows in ordinary vol_pctile order. A pure reorder -- every gate
above still applies, nothing here can refuse an entry, and an empty tier 1
degrades byte-identically to `rank_by="vol_pctile"`.

ONE-NUMBER DOCTRINE. The yield used to tier a candidate is computed by
`fills.ann_yield_on_collateral`, the SAME function `yield_ok` (the
`min_ann_yield_on_collateral` gate) calls -- a second formula computing the
same number would be a bug even if it agreed today. See
`test_the_tiering_yield_is_the_gates_yield_one_number_doctrine`.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.engine_v2.options.fills import ann_yield_on_collateral, yield_ok
from src.engine_v2.options.vrp import VRP_VIABLE_MIN_ANN_YIELD, is_viable

D = pd.Timestamp("2026-07-21")
COLS = ["date", "expiry", "dte", "strike", "right", "bid", "ask", "mid",
        "close", "delta", "iv", "underlying",
        "open_interest", "volume", "bid_size", "ask_size"]


def _row(strike, dte, bid):
    expiry = D + pd.Timedelta(days=dte)
    return [D, expiry, dte, strike, "P", bid, bid + 0.10, bid + 0.05,
           bid + 0.05, -0.40, 0.2, strike * 1.05, 500.0, 100.0, 10.0, 10.0]


def _chain_for(strike, dte, bid):
    ch = pd.DataFrame([_row(strike, dte, bid)], columns=COLS)
    ch["date"] = pd.to_datetime(ch["date"])
    ch["expiry"] = pd.to_datetime(ch["expiry"])
    return ch


def _states_row(vol_pctile):
    df = pd.DataFrame([["uptrend", "calm", vol_pctile]],
                      columns=["trend", "vol", "vol_pctile"],
                      index=pd.to_datetime([D - pd.Timedelta(days=1)]))
    return df


class M:
    """Multi-ticker market fake: each ticker carries its own (strike, dte,
    bid) -- controlling its own annualized yield independently of the other
    tiering variables -- plus its own vol_pctile and VRP.

    `names` maps ticker -> dict(strike, dte, bid, pctile, vrp). `vrp=None`
    is the unmeasurable case.
    """

    def __init__(self, names, rankable=True):
        self._names = names
        self.universe = list(names)
        self._rankable = rankable

    def chain(self, tk, d):
        n = self._names[tk]
        return _chain_for(n["strike"], n["dte"], n["bid"])

    def spot(self, tk, d, fb):
        return self._names[tk]["strike"] * 1.05

    def settle_price(self, tk, expiry):
        return self._names[tk]["strike"] * 1.05

    def regime_row(self, tk, day):
        return {"vol_pctile": self._names[tk]["pctile"],
                "trend": "uptrend", "vol": "calm"}

    def eligible(self, tk, day):
        return True

    def vrp_rankable(self):
        return self._rankable

    def vrp(self, tk, day, realized_vol):
        return self._names[tk]["vrp"]


def _pcfg(**kw):
    from src.engine_v2.options.wheel import WheelConfig
    kw.setdefault("rank_by", "vrp_viable")
    return WheelConfig(ticker="DOW", starting_capital=1_000_000.0,
                       put_delta=0.40, call_delta=0.50, target_dte=7,
                       take_profit_pct=0.25, call_min_strike="basis", **kw)


def _step(market, cfg, n_slots=1):
    from src.engine_v2.options.portfolio import PortfolioState, step_one_day
    st = PortfolioState(cash=1_000_000.0, positions=[])
    return st, step_one_day(st, market, D, cfg, selector="plain",
                            n_slots=n_slots)


def _sold(r):
    return [t.contract.root for t in r.trades if t.action == "SELL_PUT"]


# -- 1. tier assignment and full ordering -------------------------------

def test_tier_assignment_and_full_ordering():
    """Two viable names (yield >= 0.30) ranked by VRP desc, strictly ahead
    of two non-viable names ranked by vol_pctile desc -- even though the
    non-viable names have far richer vol_pctile than either viable one."""
    names = {
        # yield = (bid/strike)*(365/dte) = (1.0/40)*(365/7) = 1.303 -- viable
        "RICH_VRP": {"strike": 40.0, "dte": 7, "bid": 1.0,
                    "pctile": 0.01, "vrp": 3.0},
        # yield = (1.0/100)*(365/7) = 0.521 -- viable, lower VRP
        "MID_VRP":  {"strike": 100.0, "dte": 7, "bid": 1.0,
                    "pctile": 0.02, "vrp": 1.5},
        # yield = (1.0/300)*(365/7) = 0.174 -- NOT viable, highest pctile
        "HI_PCT":   {"strike": 300.0, "dte": 7, "bid": 1.0,
                    "pctile": 0.90, "vrp": 9.0},
        # yield = (1.0/500)*(365/7) = 0.104 -- NOT viable, lower pctile
        "LO_PCT":   {"strike": 500.0, "dte": 7, "bid": 1.0,
                    "pctile": 0.30, "vrp": 9.0},
    }
    _, r = _step(M(names), _pcfg(), n_slots=4)
    assert _sold(r) == ["RICH_VRP", "MID_VRP", "HI_PCT", "LO_PCT"]


# -- 2. boundary: yield exactly 0.30 is IN tier 1 ------------------------
#
# Tested at the predicate itself, not through the full pipeline: the DTE
# selection band (`select.derived_band`, +-a few days around target_dte) and
# `select_contract`'s nearest-delta search make hand-picking a chain that
# hits an EXACT float boundary through real contract selection fragile and
# tautological -- it would really be testing `select_contract`, not the
# tiering rule. `is_viable` is the single decision point every ranking path
# calls (see the elif branch in portfolio.step_one_day), so pinning it here
# pins the pipeline's behavior too.

def test_yield_exactly_at_the_floor_is_tier_one():
    """>=, not >: the pinned constant means exactly what it says."""
    assert ann_yield_on_collateral(0.30, 1.0, 365) == pytest.approx(0.30)
    assert is_viable(VRP_VIABLE_MIN_ANN_YIELD) is True
    assert is_viable(VRP_VIABLE_MIN_ANN_YIELD - 1e-9) is False


# -- 3. tier-1-empty degrades to exact vol_pctile order ------------------

def test_tier_one_empty_degrades_to_plain_vol_pctile_order():
    """Nobody clears the floor -> the day's order is byte-identical to
    `rank_by="vol_pctile"`: the richer-VRP, lower-pctile name loses."""
    names = {
        "JUMPY": {"strike": 500.0, "dte": 7, "bid": 1.0,
                 "pctile": 0.99, "vrp": 0.1},
        "PAID":  {"strike": 500.0, "dte": 7, "bid": 1.0,
                 "pctile": 0.10, "vrp": 9.0},
    }
    _, r = _step(M(names), _pcfg(), n_slots=2)
    assert _sold(r) == ["JUMPY", "PAID"], \
        "tier-1-empty must reduce to plain vol_pctile order"


# -- 4. unknown VRP in tier 1 sorts NEUTRAL and is counted ---------------

def test_unknown_vrp_in_tier_one_sorts_neutral_and_is_counted():
    names = {
        "NOHIST": {"strike": 40.0, "dte": 7, "bid": 1.0,
                  "pctile": 0.01, "vrp": None},
        "BELOW":  {"strike": 40.0, "dte": 7, "bid": 1.0,
                  "pctile": 0.02, "vrp": 0.5},
        "ABOVE":  {"strike": 40.0, "dte": 7, "bid": 1.0,
                  "pctile": 0.03, "vrp": 2.0},
    }
    _, r = _step(M(names), _pcfg(), n_slots=3)
    assert _sold(r) == ["ABOVE", "NOHIST", "BELOW"], \
        "NEUTRAL (1.0) must outrank a measurably cheap VRP and lose to a rich one"
    assert any(w[1] == "entry_ranked_vrp_unknown" and w[2] == "NOHIST"
              for w in r.warnings)


def test_unknown_is_not_counted_for_a_tier_two_candidate():
    """Tier 2 sorts on vol_pctile alone -- VRP is never even read for it, so
    an unmeasurable VRP on a non-viable name reports nothing."""
    names = {"NOHIST": {"strike": 500.0, "dte": 7, "bid": 1.0,
                        "pctile": 0.50, "vrp": None}}
    _, r = _step(M(names), _pcfg(), n_slots=1)
    assert not any(w[1] == "entry_ranked_vrp_unknown" for w in r.warnings)


# -- 5. the sort never refuses an entry ----------------------------------

def test_ranking_never_refuses_an_entry():
    """A tier-2, low-pctile, sole candidate must still trade."""
    names = {"ONLY": {"strike": 500.0, "dte": 7, "bid": 1.0,
                      "pctile": 0.01, "vrp": 0.1}}
    _, r = _step(M(names), _pcfg(), n_slots=1)
    assert _sold(r) == ["ONLY"], "ranking acted as a filter"


# -- 6. one-number doctrine: the tiering yield IS the gate's yield -------

class _Cfg:
    def __init__(self, floor):
        self.min_ann_yield_on_collateral = floor


def test_the_tiering_yield_is_the_gates_yield_one_number_doctrine():
    for credit, strike, dte in [(1.0, 40.0, 7), (1.0, 300.0, 7),
                                (0.30, 1.0, 365), (2.5, 156.43, 11)]:
        y = ann_yield_on_collateral(credit, strike, dte)
        gate_ok, _ = yield_ok(credit, strike, dte,
                              _Cfg(VRP_VIABLE_MIN_ANN_YIELD))
        assert gate_ok == is_viable(y), (
            "the viable-tier predicate and the collateral-yield gate must "
            "agree at the SAME floor -- a second formula computing the same "
            "number would be a bug even if it agreed here")


# -- 7. fail-loud without an IV history, both drivers --------------------

def test_a_market_with_no_iv_history_refuses_to_rank_on_vrp_viable():
    names = {"ONLY": {"strike": 40.0, "dte": 7, "bid": 1.0,
                      "pctile": 0.50, "vrp": None}}
    with pytest.raises(ValueError,
                       match="rank_by='vrp_viable' needs a market"):
        _step(M(names, rankable=False), _pcfg(), n_slots=1)


def test_the_batch_driver_refuses_vrp_viable_without_a_history_too():
    from src.engine_v2.options.portfolio import run_portfolio_wheel
    with pytest.raises(ValueError, match="rank_by='vrp_viable' needs an IV history"):
        run_portfolio_wheel({}, _pcfg(), {}, selector="chop", n_slots=1,
                            universe=[], iv_history=None)


def test_an_unknown_sort_key_is_still_refused_by_name():
    from src.engine_v2.options.portfolio import run_portfolio_wheel
    with pytest.raises(ValueError, match="rank_by must be"):
        run_portfolio_wheel({}, _pcfg(rank_by="vrp_viablee"), {},
                            selector="chop", n_slots=1, universe=[],
                            iv_history=None)


# -- pinned constant -------------------------------------------------------

def test_the_viable_floor_is_pinned_at_the_measured_median():
    assert VRP_VIABLE_MIN_ANN_YIELD == pytest.approx(0.30)


def test_unmeasurable_yield_is_never_viable():
    assert is_viable(None) is False
