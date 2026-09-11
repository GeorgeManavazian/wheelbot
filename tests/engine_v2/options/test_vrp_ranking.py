"""Ranking by the variance risk premium: implied vol against realized vol.

WHAT THIS MEASURES AND WHY IT IS NOT `iv_rank`. Selling a put earns
`IV - future realized vol`. That spread IS the edge; everything else in the
wheel is the delivery mechanism. The live ranker sorts on realized vol -- the
RISK TAKEN -- and the arm measured on 2026-08-04 sorted on IV rank, the LEVEL
of the price paid. Neither reads the spread, and the level of IV is confounded
with the level of risk: measured over 10,528 gate-passing ticker-days on the
live window, IV/RV falls monotonically across every realized-vol decile, from
1.62 in the calmest to 0.998 in the jumpiest. So `rank_by="vol_pctile"` sorts
to the exact end of the distribution where the edge is gone.

UNKNOWN IS NEUTRAL, NOT LAST -- the same rule `iv_rank` follows, for the same
reason recorded there: sorting unmeasurable names last is a filter wearing a
sort's clothes, and that is what made the IV-rank VETO cost 18 points of return
by halving the campaign count.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.engine_v2.options.iv_rank import IVHistory, NEUTRAL_IV_RANK
from src.engine_v2.options import vrp


def hist(**series) -> IVHistory:
    return IVHistory({t: pd.Series({pd.Timestamp(d): v for d, v in obs.items()})
                      for t, obs in series.items()})


DAY = pd.Timestamp("2025-03-10")


def test_vrp_is_implied_over_realized():
    h = hist(AAA={"2025-03-10": 0.40})
    assert vrp.vrp_of(h, "AAA", DAY, realized_vol=0.20) == pytest.approx(2.0)


def test_a_name_paid_less_than_its_own_risk_scores_below_one():
    """The whole point of the statistic: below 1.0 the option is cheaper than
    the volatility the stock has actually been delivering."""
    h = hist(AAA={"2025-03-10": 0.30})
    assert vrp.vrp_of(h, "AAA", DAY, realized_vol=0.50) == pytest.approx(0.6)


def test_it_reads_no_observation_after_the_day_asked_about():
    """Look-ahead is the failure that makes a backtest meaningless, so the
    slice happens here rather than at the call site, exactly as IVHistory.rank
    does it."""
    h = hist(AAA={"2025-03-10": 0.40, "2025-03-11": 9.99})
    assert vrp.vrp_of(h, "AAA", DAY, realized_vol=0.20) == pytest.approx(2.0)


def test_the_most_recent_observation_at_or_before_the_day_is_used():
    h = hist(AAA={"2025-03-05": 0.20, "2025-03-07": 0.40})
    assert vrp.vrp_of(h, "AAA", DAY, realized_vol=0.20) == pytest.approx(2.0)


@pytest.mark.parametrize("rv", [None, 0.0, float("nan")])
def test_an_unusable_realized_vol_is_unknown_not_infinite(rv):
    """Dividing by zero would score the calmest possible name as the single
    best opportunity in the book -- the most attractive possible answer to the
    least measurable case."""
    h = hist(AAA={"2025-03-10": 0.40})
    assert vrp.vrp_of(h, "AAA", DAY, realized_vol=rv) is None


def test_a_ticker_with_no_history_is_unknown_not_zero():
    assert vrp.vrp_of(hist(), "AAA", DAY, realized_vol=0.20) is None


def test_no_history_object_at_all_is_unknown():
    assert vrp.vrp_of(None, "AAA", DAY, realized_vol=0.20) is None


def test_unknown_sorts_neutral_so_the_sort_never_becomes_a_filter():
    """A veto can wave unknowns through harmlessly; a sort must physically
    place them. NEUTRAL is the midpoint of the measurable range, so an
    unmeasurable name outranks a measurably underpaid one and loses to a
    measurably rich one."""
    assert vrp.sort_key(None) == vrp.NEUTRAL_VRP
    assert vrp.sort_key(2.0) == 2.0
    assert vrp.NEUTRAL_VRP == pytest.approx(1.0)


def test_neutral_is_one_because_one_is_where_the_edge_is_zero():
    """Unlike a percentile, this statistic has a meaningful zero point: at 1.0
    the option is priced exactly at the volatility the stock delivered. That
    makes 1.0 the honest 'no information' placement, and it is NOT a tuning
    knob -- moving it re-introduces the bias neutrality exists to avoid."""
    assert vrp.sort_key(None) < vrp.sort_key(1.01)
    assert vrp.sort_key(None) > vrp.sort_key(0.99)


def test_neutral_vrp_is_not_the_iv_rank_neutral():
    """Different statistic, different midpoint. 0.5 is the middle of a
    percentile; 1.0 is fair value on a ratio. Sharing the constant would be a
    silent unit error."""
    assert vrp.NEUTRAL_VRP != NEUTRAL_IV_RANK


# -- wiring: the sort must not be able to run on nothing ---------------------

class _NoHistoryMarket:
    """Has every method the ranker calls, and no IV behind them -- the exact
    shape that made `iv_rankable` a required declaration rather than a
    method-presence check."""
    universe: list = []
    def chain(self, t, d): return None
    def spot(self, t, d, fb): return 10.0
    def settle_price(self, t, e): return 10.0
    def regime_row(self, t, d): return None
    def eligible(self, t, d): return True
    def vrp_rankable(self): return False
    def vrp(self, t, d, rv): return None


def test_a_market_with_no_iv_history_refuses_to_rank_on_vrp():
    """Not "returns nothing useful" -- REFUSES. With no history every name
    scores NEUTRAL, the tie breaks on universe order, and the run produces a
    full plausible set of trades chosen by nothing at all."""
    from src.engine_v2.options.portfolio import PortfolioState, step_one_day
    from src.engine_v2.options.wheel import WheelConfig
    cfg = WheelConfig(ticker="GDX", put_delta=0.20, call_delta=0.50,
                      target_dte=7, take_profit_pct=0.50,
                      starting_capital=100_000.0, call_min_strike="basis",
                      rank_by="vrp")
    state = PortfolioState(cash=100_000.0, positions=[])
    with pytest.raises(ValueError, match="needs a market that can read implied vol"):
        step_one_day(state, _NoHistoryMarket(), pd.Timestamp("2021-01-15"),
                     cfg, selector="chop", n_slots=1)


def test_the_batch_driver_refuses_vrp_without_a_history_too():
    """The guard lives in both places on purpose: the live bot calls
    step_one_day directly and never touches run_portfolio_wheel."""
    from src.engine_v2.options.portfolio import run_portfolio_wheel
    from src.engine_v2.options.wheel import WheelConfig
    cfg = WheelConfig(ticker="GDX", put_delta=0.20, call_delta=0.50,
                      target_dte=7, take_profit_pct=0.50,
                      starting_capital=100_000.0, call_min_strike="basis",
                      rank_by="vrp")
    with pytest.raises(ValueError, match="rank_by='vrp' needs an IV history"):
        run_portfolio_wheel({}, cfg, {}, selector="chop", n_slots=1,
                            universe=[], iv_history=None)


def test_an_unknown_sort_key_is_still_refused_by_name():
    from src.engine_v2.options.portfolio import run_portfolio_wheel
    from src.engine_v2.options.wheel import WheelConfig
    cfg = WheelConfig(ticker="GDX", put_delta=0.20, call_delta=0.50,
                      target_dte=7, take_profit_pct=0.50,
                      starting_capital=100_000.0, call_min_strike="basis",
                      rank_by="vrpp")
    with pytest.raises(ValueError, match="rank_by must be"):
        run_portfolio_wheel({}, cfg, {}, selector="chop", n_slots=1,
                            universe=[], iv_history=None)
