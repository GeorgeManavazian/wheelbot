"""Schema 2 additions (gate-campaign batch 1, 2026-08-17).

Two additions and one repair, each with the reason it exists:

  spy_total_return -- the campaign's done condition is "2-yr P&L >= SPY
      buy-hold over the same window" and NO scorecard could answer it: the
      existing `buy_hold` arm has silently been None on every card ever
      written (see test below). Report-only: it joins the arms table but is
      never part of delta(), so no bar can accidentally grade against it.

  calls_written_below_basis / campaigns_with_below_basis_call -- the
      tier2-deep arms' auto-reject is "below-basis call writes on more than a
      third of assigned campaigns". exit-ladder-on's poison was exactly these
      writes and the card could not count them. Reconstructed bench-side from
      the trade log with the engine's own net-basis definition
      (portfolio._net_basis), so the pinned engine stays byte-identical.

  conditions["fill_check"] -- bench/run.py assigned it and then overwrote the
      whole conditions dict two lines later, so no card ever carried the fill
      check it printed. Order fixed; pinned here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bench import scorecard as S
from bench.loaders import market


def curve(vals, start="2025-01-01"):
    idx = pd.bdate_range(start, periods=len(vals))
    return pd.Series(np.array(vals, dtype=float), index=idx)


class FakeResult:
    def __init__(self, eq, trades=(), warnings=()):
        self.equity = eq
        self.trades = list(trades)
        self.warnings = list(warnings)
        self.final_cash = 0.0
        self.final_shares = {}
        self.days_flat = 0
        self.days_shares_uncovered = 0
        self.n_campaigns_opened = len({getattr(t, "campaign_id", 0)
                                       for t in trades}) if trades else 0


class FakeContract:
    def __init__(self, root, strike=None, right=None):
        self.root = root
        self.strike = strike
        self.right = right


class FakeTrade:
    def __init__(self, date, action, cash_after, campaign_id,
                 contract=None, contracts=1):
        self.date = pd.Timestamp(date)
        self.action = action
        self.cash_after = cash_after
        self.campaign_id = campaign_id
        self.contract = contract if contract is not None else FakeContract("AAA")
        self.contracts = contracts


# -- schema -----------------------------------------------------------------

def test_schema_is_2():
    assert S.SCHEMA == 2


# -- SPY buy-hold -----------------------------------------------------------

def test_spy_buy_hold_scales_spy_closes_to_the_capital():
    closes = {"SPY": curve([400.0, 440.0, 480.0]),
              "QQQ": curve([300.0, 300.0, 300.0])}
    eq = market.spy_buy_hold(closes, "2025-01-01", "2025-12-31", 100_000.0)
    assert eq.iloc[0] == pytest.approx(100_000.0)
    assert eq.iloc[-1] == pytest.approx(120_000.0)


def test_spy_buy_hold_is_windowed_not_whole_history():
    idx = pd.to_datetime(["2024-01-02", "2025-01-02", "2025-06-02"])
    closes = {"SPY": pd.Series([200.0, 400.0, 500.0], index=idx)}
    eq = market.spy_buy_hold(closes, "2025-01-01", "2025-12-31", 100_000.0)
    # The 2024 double must not be counted: base is the first close INSIDE the window.
    assert eq.iloc[0] == pytest.approx(100_000.0)
    assert eq.iloc[-1] == pytest.approx(125_000.0)
    assert len(eq) == 2


def test_spy_buy_hold_raises_rather_than_silently_benchmarking_nothing():
    """The old equal-weight buy_hold arm returned None on every run ever made
    and nothing noticed. A benchmark the done-condition depends on fails loud."""
    with pytest.raises(market.PrerequisiteMissing):
        market.spy_buy_hold({"QQQ": curve([1.0, 2.0])},
                            "2025-01-01", "2025-12-31", 100_000.0)
    with pytest.raises(market.PrerequisiteMissing):
        market.spy_buy_hold({"SPY": curve([1.0, 2.0])},
                            "2030-01-01", "2030-12-31", 100_000.0)


def test_spy_buy_hold_metrics_join_the_render_but_never_the_delta():
    c = S.new("v", "frozen", "live", True)
    c.arms["variant"] = S.metrics_from_equity(curve([100, 120]), 100.0)
    c.arms["base"] = S.metrics_from_equity(curve([100, 110]), 100.0)
    c.arms["spy_buy_hold"] = S.metrics_from_equity(curve([100, 115]), 100.0)
    text = S.render(c)
    assert "spy_buy_hold" in text
    # delta stays variant-minus-base: the benchmark is report-only.
    assert c.delta()["total_return"] == pytest.approx(0.10)


# -- below-basis covered-call writes ----------------------------------------

OPENING = 20_000.0
MULT = 100


def _assigned_campaign(cid=1, credit=200.0, strike=100.0, start_cash=OPENING):
    """SELL_PUT banks `credit`, assignment at `strike` for one contract.
    Net basis afterwards = strike - credit/100."""
    cash1 = start_cash + credit
    cash2 = cash1 - strike * MULT
    return [
        FakeTrade("2025-01-01", "SELL_PUT", cash1, cid,
                  FakeContract("AAA", strike, "P")),
        FakeTrade("2025-01-02", "ASSIGNED", cash2, cid,
                  FakeContract("AAA", strike, "P")),
    ], cash2


def test_a_call_written_below_net_basis_is_counted():
    trades, cash = _assigned_campaign()          # net basis = 100 - 2 = 98
    trades.append(FakeTrade("2025-01-03", "SELL_CALL", cash + 150.0, 1,
                            FakeContract("AAA", 97.0, "C")))
    m = S.metrics_from_result(FakeResult(curve([100.0] * 5), trades), OPENING)
    assert m["calls_written_below_basis"] == 1
    assert m["campaigns_with_below_basis_call"] == 1


def test_net_basis_is_strike_minus_premium_banked_not_the_gross_strike():
    """The engine's floor is `basis - premium/shares` (portfolio._net_basis).
    A 99 call against a 100 assignment with $200 banked is ABOVE the 98 net
    basis; counting it would use the gross strike and overstate the counter."""
    trades, cash = _assigned_campaign()          # net basis = 98
    trades.append(FakeTrade("2025-01-03", "SELL_CALL", cash + 150.0, 1,
                            FakeContract("AAA", 99.0, "C")))
    m = S.metrics_from_result(FakeResult(curve([100.0] * 5), trades), OPENING)
    assert m["calls_written_below_basis"] == 0
    assert m["campaigns_with_below_basis_call"] == 0


def test_the_write_does_not_lower_its_own_floor():
    """portfolio.py computes the floor BEFORE booking the call's credit, so the
    reconstruction must test the strike against the pre-write premium. A $300
    credit on the 97.5 call would drag net basis to 95 if it were (wrongly)
    banked first, and the below-basis write would go uncounted."""
    trades, cash = _assigned_campaign()          # net basis before write = 98
    trades.append(FakeTrade("2025-01-03", "SELL_CALL", cash + 300.0, 1,
                            FakeContract("AAA", 97.5, "C")))
    m = S.metrics_from_result(FakeResult(curve([100.0] * 5), trades), OPENING)
    assert m["calls_written_below_basis"] == 1


def test_premium_accumulates_across_the_campaign_before_the_check():
    """A buy-back (negative premium flow) RAISES net basis again: after closing
    the first call at a cost, a later write at the same strike can be below the
    new net basis. The premium ledger must follow cash_after differences, the
    way the engine's pos['premium'] follows fills."""
    trades, cash = _assigned_campaign()          # premium 200, nb = 98
    # write at 98.5 (above 98): not counted; credit +100 -> premium 300, nb 97
    cash += 100.0
    trades.append(FakeTrade("2025-01-03", "SELL_CALL", cash, 1,
                            FakeContract("AAA", 98.5, "C")))
    # buy it back for 250 -> premium 50, nb = 99.5
    cash -= 250.0
    trades.append(FakeTrade("2025-01-04", "CLOSE_CALL", cash, 1,
                            FakeContract("AAA", 98.5, "C")))
    # 98.5 again: now BELOW the 99.5 net basis -> counted
    cash += 80.0
    trades.append(FakeTrade("2025-01-05", "SELL_CALL", cash, 1,
                            FakeContract("AAA", 98.5, "C")))
    m = S.metrics_from_result(FakeResult(curve([100.0] * 6), trades), OPENING)
    assert m["calls_written_below_basis"] == 1
    assert m["campaigns_with_below_basis_call"] == 1


def test_two_below_basis_writes_in_one_campaign_count_once_per_campaign():
    trades, cash = _assigned_campaign()          # nb = 98
    cash += 50.0
    trades.append(FakeTrade("2025-01-03", "SELL_CALL", cash, 1,
                            FakeContract("AAA", 90.0, "C")))
    cash -= 60.0
    trades.append(FakeTrade("2025-01-04", "CLOSE_CALL", cash, 1,
                            FakeContract("AAA", 90.0, "C")))
    cash += 40.0
    trades.append(FakeTrade("2025-01-05", "SELL_CALL", cash, 1,
                            FakeContract("AAA", 90.0, "C")))
    m = S.metrics_from_result(FakeResult(curve([100.0] * 6), trades), OPENING)
    assert m["calls_written_below_basis"] == 2
    assert m["campaigns_with_below_basis_call"] == 1


def test_campaigns_are_tracked_independently():
    t1, cash1 = _assigned_campaign(cid=1)                    # nb = 98
    t1.append(FakeTrade("2025-01-03", "SELL_CALL", cash1 + 50.0, 1,
                        FakeContract("AAA", 90.0, "C")))     # below
    t2, cash2 = _assigned_campaign(cid=2, start_cash=cash1 + 50.0)
    t2.append(FakeTrade("2025-01-06", "SELL_CALL", cash2 + 50.0, 2,
                        FakeContract("BBB", 99.5, "C")))     # above
    m = S.metrics_from_result(FakeResult(curve([100.0] * 7), t1 + t2), OPENING)
    assert m["calls_written_below_basis"] == 1
    assert m["campaigns_with_below_basis_call"] == 1


def test_a_run_with_no_calls_reports_zero_not_none():
    """Zero is the honest answer here (unlike the rate metrics): the counter
    is a census of events, and no events is a real, gradable zero."""
    m = S.metrics_from_result(FakeResult(curve([100, 101])), 100.0)
    assert m["calls_written_below_basis"] == 0
    assert m["campaigns_with_below_basis_call"] == 0
