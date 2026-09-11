"""The capacity counters: idle slots, pool depth, and the starvation count.

These exist to settle an argument the project has been having with itself. The
notes repeatedly explain a fall in campaign count with "a refusal leaves the
slot idle" -- but `step_one_day` rescans the WHOLE universe for every slot it
has to fill, so a refusal promotes the next candidate and cannot idle anything
unless the pool is empty outright. These counters make that a measurement
rather than a claim, and the last test here pins that they change no decision.
"""
import pandas as pd
import pytest

from src.engine_v2.options.chain import Contract
from src.engine_v2.options.wheel import WheelConfig
from src.engine_v2.options.portfolio import PortfolioState, step_one_day


def _cfg():
    return WheelConfig(ticker="GDX", put_delta=0.20, call_delta=0.50,
                       target_dte=7, take_profit_pct=0.50,
                       starting_capital=100_000.0, call_min_strike="basis")


class _EmptyMarket:
    """No universe at all: every slot stays empty for want of a candidate."""
    universe: list = []
    def chain(self, t, d): return None
    def spot(self, t, d, fb): return 10.0
    def settle_price(self, t, e): return 10.0
    def regime_row(self, t, d): return None
    def eligible(self, t, d): return True


class _NoChainMarket(_EmptyMarket):
    """A universe with names in it, but no chain for any of them -- so the pool
    comes back empty for a reason other than an empty universe."""
    universe = ["AAA", "BBB", "CCC"]


def test_empty_pool_counts_every_free_slot_as_idle():
    d = pd.Timestamp("2021-01-15")
    state = PortfolioState(cash=100_000.0, positions=[])
    step_one_day(state, _EmptyMarket(), d, _cfg(), selector="chop", n_slots=5)
    assert state.idle_slot_days == 5
    assert state.pool_depth_sessions == 1
    assert state.pool_depth_sum == 0
    assert state.pool_starved_days == 1


def test_counters_accumulate_across_sessions():
    state = PortfolioState(cash=100_000.0, positions=[])
    cfg, mkt = _cfg(), _NoChainMarket()
    for day in ("2021-01-15", "2021-01-19", "2021-01-20"):
        step_one_day(state, mkt, pd.Timestamp(day), cfg,
                     selector="chop", n_slots=2)
    assert state.idle_slot_days == 6          # 2 slots x 3 sessions
    assert state.pool_depth_sessions == 3
    assert state.pool_starved_days == 3


def test_the_scan_funnel_counts_where_candidates_left():
    """The refusal warnings only start at the liquidity floor, so without this
    the loudest counted gate reads as the biggest one. Here all three names
    have no chain, which is a silent `continue` and logs nothing."""
    d = pd.Timestamp("2021-01-15")
    state = PortfolioState(cash=100_000.0, positions=[])
    step_one_day(state, _NoChainMarket(), d, _cfg(), selector="chop", n_slots=2)
    assert state.scan_names == 3
    assert state.drop_no_chain == 3
    assert state.drop_weather == 0
    assert state.drop_no_contract == 0


def test_the_funnel_counts_the_first_pass_only():
    """The while-loop rescans the universe for every slot it fills. Counting
    each pass would multiply the funnel by the number of free slots and make a
    wide account look like a pickier one."""
    state = PortfolioState(cash=100_000.0, positions=[])
    step_one_day(state, _NoChainMarket(), pd.Timestamp("2021-01-15"), _cfg(),
                 selector="chop", n_slots=5)
    assert state.scan_names == 3        # not 15


def test_a_full_book_is_not_counted_as_a_routing_session():
    """No free slot means no demand, so the session says nothing about supply
    and must not drag the mean pool depth down."""
    d = pd.Timestamp("2021-01-15")
    put = Contract("GDX", pd.Timestamp("2021-02-19"), 30.0, "P")
    pos = {"ticker": "GDX", "shares": 0, "phase": "PUT", "basis": None,
           "premium": 99.0, "campaign": 1, "last_spot": 35.0,
           "short": {"contract": put, "contracts": 1, "credit": 1.0,
                     "last_mid": 1.0}}
    state = PortfolioState(cash=100_000.0, positions=[pos], campaign=1)
    step_one_day(state, _EmptyMarket(), d, _cfg(), selector="chop", n_slots=1)
    assert state.pool_depth_sessions == 0
    assert state.idle_slot_days == 0


def test_the_counters_are_diagnostic_only_and_change_no_decision():
    """Run the same session twice from identical state -- once reading the
    counters, once with them pre-loaded to nonsense -- and the trades, cash and
    positions must be identical. A counter that could steer a decision would be
    a behaviour change wearing a diagnostic's clothes."""
    d = pd.Timestamp("2021-01-15")
    cfg, mkt = _cfg(), _NoChainMarket()
    a = PortfolioState(cash=100_000.0, positions=[])
    b = PortfolioState(cash=100_000.0, positions=[], idle_slot_days=9_999,
                       pool_depth_sum=7, pool_depth_sessions=3,
                       pool_starved_days=2)
    ra = step_one_day(a, mkt, d, cfg, selector="chop", n_slots=3)
    rb = step_one_day(b, mkt, d, cfg, selector="chop", n_slots=3)
    assert [(t.date, t.action) for t in ra.trades] == \
           [(t.date, t.action) for t in rb.trades]
    assert ra.equity == rb.equity
    assert a.cash == b.cash and a.positions == b.positions
    assert b.idle_slot_days == 9_999 + 3      # accumulated, never read back
