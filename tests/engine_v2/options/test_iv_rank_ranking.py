"""IV rank as the SORT KEY, not a veto.

Why this exists: the veto was measured on the real portfolio (2026-08-04,
scratchpad/run_iv_rank_ab.py) and REJECTED. It raises P&L per campaign ~14%
and halves campaign count 254 -> 130, costing 18 points of total return:

    arm                 total    Sharpe   campaigns   P&L/campaign
    baseline (off)     +43.0%      1.10         254           $169
    iv_rank >= 0.88    +24.9%      0.84         130           $192
    iv_rank >= 0.90    +21.4%      0.77         121           $177
    iv_rank >= 0.95     +9.5%      0.39          80           $119

The signal is real; the instrument was wrong. A veto trades quality for
volume, and the wheel's return comes from capital turnover -- a refused entry
leaves the slot earning nothing that week.

So: rank, don't filter. The engine currently sorts candidates by
`vol_pctile` (REALIZED vol, highest first) -- it picks maximum risk with no
check on the price paid for taking it, and the weather gate has already capped
realized vol at the 75th percentile, so it chooses inside a 70-75th percentile
sliver on the wrong variable. Replacing that key with IV rank changes WHICH 5
names fill the slots without changing HOW MANY get filled, so it never pays
the turnover cost that killed the veto.

Owner decisions, 2026-08-05:
  1. REPLACE, not blend. Pure IV-rank sort is the honest A/B -- one variable,
     no weight to tune, and a losing result is readable.
  2. Unknown ranks sort at NEUTRAL (0.5), not last. Sorting unknowns last
     silently de-prioritises every thin-history name, which is a filter
     wearing a sort's clothes -- exactly the failure mode above.

Spec: this file + [[2026-08-04 - IV rank as an entry veto, measured and rejected]]
"""
import pandas as pd
import pytest

from src.engine_v2.options.iv_rank import NEUTRAL_IV_RANK, IVHistory
from src.engine_v2.options.market import BatchMarket
from src.engine_v2.options.portfolio import (PortfolioState, run_portfolio_wheel,
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


def _states_row(vol_pctile):
    """One regime row, dated strictly before D so step_one_day reads it."""
    df = pd.DataFrame([["uptrend", "calm", vol_pctile]],
                      columns=["trend", "vol", "vol_pctile"],
                      index=pd.to_datetime([D - pd.Timedelta(days=1)]))
    return df


class _Undeclared:
    """Multi-ticker market fake carrying BOTH ranking variables per ticker,
    but NOT declaring iv_rankable() -- what LiveMarket was, and what every
    market fake in this repo looks like by default.

    `names` maps ticker -> (vol_pctile, iv_rank). An iv_rank of None is the
    unmeasurable case (absent ticker or too few trailing observations).
    """

    def __init__(self, names):
        self._names = names
        self.universe = list(names)
        self._ch = _chain()

    def chain(self, tk, d):
        return self._ch

    def spot(self, tk, d, fb):
        return 41.0

    def settle_price(self, tk, expiry):
        return 41.0

    def regime_row(self, tk, day):
        return {"vol_pctile": self._names[tk][0],
                "trend": "uptrend", "vol": "calm"}

    def eligible(self, tk, day):
        return True

    def iv_rank(self, tk, day):
        return self._names[tk][1]


class M(_Undeclared):
    """The same fake, declaring that it can actually rank on IV."""

    def iv_rankable(self):
        return True


def _pcfg(**kw):
    return WheelConfig(ticker="DOW", starting_capital=100_000.0, put_delta=0.40,
                       call_delta=0.50, target_dte=7, take_profit_pct=0.25,
                       call_min_strike="basis", **kw)


def _step(market, cfg, n_slots=1):
    st = PortfolioState(cash=100_000.0, positions=[])
    return st, step_one_day(st, market, D, cfg, selector="plain",
                            n_slots=n_slots)


# The two names disagree on purpose: whichever variable is driving the sort,
# the OTHER one would have picked the loser. No test here can pass by accident.
OPPOSED = {"JUMPY": (0.99, 0.10),    # max realized vol, cheap IV
           "PAID":  (0.10, 0.95)}    # calm realized vol, rich IV


def _sold(r):
    return [t.contract.root for t in r.trades if t.action == "SELL_PUT"]


# --- the sort key ------------------------------------------------------------

def test_default_still_ranks_on_realized_vol():
    """rank_by unset -> the pre-existing key. Byte-identical default path."""
    _, r = _step(M(OPPOSED), _pcfg())
    assert _sold(r) == ["JUMPY"]


def test_iv_rank_sort_picks_the_best_paid_name():
    """The whole point: choose on price received, not on risk taken."""
    _, r = _step(M(OPPOSED), _pcfg(rank_by="iv_rank"))
    assert _sold(r) == ["PAID"], "sorted on realized vol despite rank_by"


def test_ranking_never_refuses_an_entry():
    """THE regression guard. A sort reorders; it must not veto.

    A bottom-decile IV rank is exactly what the rejected gate refused. Under
    ranking it must still trade when it is the only candidate -- otherwise we
    have rebuilt the instrument that cost 18 points of return, and campaign
    count will quietly fall away from the 254 baseline.
    """
    _, r = _step(M({"CHEAP": (0.50, 0.01)}), _pcfg(rank_by="iv_rank"))
    assert _sold(r) == ["CHEAP"], "ranking acted as a filter"


# --- unknown ranks (owner decision 2: neutral, not last) ---------------------

def test_unknown_rank_outranks_a_measurably_cheap_name():
    """Neutral 0.5 beats 0.10: no history is not evidence of cheap vol."""
    _, r = _step(M({"CHEAP": (0.99, 0.10), "NOHIST": (0.10, None)}),
                 _pcfg(rank_by="iv_rank"))
    assert _sold(r) == ["NOHIST"]


def test_unknown_rank_loses_to_a_measurably_rich_name():
    """...and 0.5 still loses to 0.95: no history is not evidence of rich vol
    either. Neutral means neutral in both directions."""
    _, r = _step(M({"RICH": (0.10, 0.95), "NOHIST": (0.99, None)}),
                 _pcfg(rank_by="iv_rank"))
    assert _sold(r) == ["RICH"]


def test_unknown_rank_sorts_at_the_documented_constant():
    """Pin the value the two tests above straddle, so moving it is deliberate."""
    assert NEUTRAL_IV_RANK == 0.5


def test_unknown_ranks_are_counted_not_shrugged_at():
    """How many names were ranked on a neutral is the diagnostic that says
    whether decision 2 mattered at all. It must be measurable from the run,
    not guessed at afterwards."""
    _, r = _step(M({"NOHIST": (0.50, None)}), _pcfg(rank_by="iv_rank"))
    assert any(w[1] == "entry_ranked_iv_unknown" and w[2] == "NOHIST"
               for w in r.warnings)


def test_unknown_is_not_counted_when_ranking_on_vol_pctile():
    """The default path reads no IV at all, so it has nothing to report."""
    _, r = _step(M({"NOHIST": (0.50, None)}), _pcfg())
    assert not any(w[1] == "entry_ranked_iv_unknown" for w in r.warnings)


def test_unknown_warning_is_deduped_per_day_and_ticker():
    """The n_slots while-loop rebuilds the pool on every iteration, so a name
    that never wins a slot is re-ranked each time (same reason the gate
    warnings are deduped)."""
    st = PortfolioState(cash=100_000.0, positions=[])
    r = step_one_day(st, M({"A": (0.10, 0.95), "B": (0.10, 0.90),
                            "NOHIST": (0.10, None)}),
                     D, _pcfg(rank_by="iv_rank"), selector="plain", n_slots=2)
    assert sum(1 for w in r.warnings if w[1] == "entry_ranked_iv_unknown") == 1


# --- refusing to run inert ---------------------------------------------------

def test_iv_rank_sort_without_a_history_refuses_to_run():
    """Same stance as min_iv_rank / earnings_blackout: a ranker with no history
    would score every name neutral and still print a full set of trades, which
    reads in the log exactly like a working one."""
    with pytest.raises(ValueError, match="needs an IV history"):
        run_portfolio_wheel({"SPY": _chain()}, _pcfg(rank_by="iv_rank"),
                            regime_states={"SPY": pd.DataFrame()})


def test_unknown_rank_by_is_rejected():
    """A typo'd sort key must not silently fall back to the default."""
    with pytest.raises(ValueError, match="rank_by must be"):
        run_portfolio_wheel({"SPY": _chain()}, _pcfg(rank_by="iv_ranks"),
                            regime_states={"SPY": pd.DataFrame()})


# --- the same refusal, from the LIVE entry point -----------------------------
#
# The guard above lives in run_portfolio_wheel. The live bot does not call it:
# live/run_daily.py calls step_one_day directly (paper_step). So the batch path
# went through the door with the alarm on it and the live path went through a
# door with no alarm at all -- a ranker scoring every name NEUTRAL, selection
# falling through to `market.universe.index(tk)`, and ~531 dedup'd
# entry_ranked_iv_unknown warnings a day as the only trace. That is the exact
# zombie-gate class this file exists to catch, authored by accident while
# writing the guard meant to prevent it.
#
# A market must therefore DECLARE that it can rank on IV. Absent declaration is
# not "probably fine": ~20 test fakes and the shipped LiveMarket all lacked the
# capability entirely, and silence read as consent.

def test_iv_rank_sort_refuses_from_step_one_day_when_undeclared():
    with pytest.raises(ValueError, match="rank_by='iv_rank'"):
        _step(_Undeclared(OPPOSED), _pcfg(rank_by="iv_rank"))


# --- and the real BatchMarket must still get THROUGH that guard --------------
#
# A guard that refuses everything is not a guard, it is an outage. This is the
# end-to-end proof that the backtest arm the A/B was measured on still runs:
# real chains -> real IVHistory -> BatchMarket -> step_one_day.

def _ramp_history(last_iv_by_ticker):
    """A real IVHistory: 252 trailing observations per ticker ending on D.

    The first 251 ramp 0.100 -> 0.350, so a final value above the top ranks
    ~1.0 and one below the bottom ranks 0.0. Deliberately over MIN_RANK_OBS
    (150) -- a thinner series would return None and the test would pass for
    the wrong reason.
    """
    days = pd.bdate_range(end=D, periods=252)
    ramp = [0.100 + i / 1000.0 for i in range(251)]
    return IVHistory({tk: pd.Series(ramp + [last], index=days)
                      for tk, last in last_iv_by_ticker.items()})


# vol_pctile says SPY (0.99 vs 0.10); IV rank says GDX (0.40 rich vs 0.05
# cheap). Whichever key drives the sort, the other would have picked the loser.
_OPPOSED_STATES = {
    "SPY": _states_row(0.99),
    "GDX": _states_row(0.10),
}


def test_batch_market_ranks_on_iv_end_to_end():
    port = run_portfolio_wheel(
        {"SPY": _chain(), "GDX": _chain()}, _pcfg(rank_by="iv_rank"),
        _OPPOSED_STATES, universe=["SPY", "GDX"], n_slots=1,
        iv_history=_ramp_history({"SPY": 0.05, "GDX": 0.40}))
    assert [t.contract.root for t in port.trades if t.action == "SELL_PUT"] \
        == ["GDX"], "BatchMarket did not rank on IV through run_portfolio_wheel"


def test_batch_market_without_a_history_is_not_rankable():
    """The declaration must be about the history, not about the class. A
    BatchMarket built with iv_history=None looks identical from the outside --
    it HAS an iv_rank() method, which returns None for every name."""
    m = BatchMarket({"SPY": _chain()}, _OPPOSED_STATES, {}, ["SPY"])
    assert m.iv_rankable() is False
