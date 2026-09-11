"""Scorecards: the schema, the metrics, and the staleness rule that stops a
favourable old run being used as evidence for a config that has since changed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bench import fingerprint, scorecard as S


def curve(vals, start="2025-01-01"):
    idx = pd.bdate_range(start, periods=len(vals))
    return pd.Series(np.array(vals, dtype=float), index=idx)


class FakeResult:
    def __init__(self, eq, trades=(), warnings=(), **caps):
        self.equity = eq
        self.trades = list(trades)
        self.warnings = list(warnings)
        self.final_cash = 0.0
        self.final_shares = {}
        self.days_flat = 3
        self.days_shares_uncovered = 2
        self.n_campaigns_opened = 4
        self.idle_slot_days = caps.get("idle_slot_days", 0)
        self.pool_depth_sum = caps.get("pool_depth_sum", 0)
        self.pool_depth_sessions = caps.get("pool_depth_sessions", 0)
        self.pool_starved_days = caps.get("pool_starved_days", 0)
        self.rank_viable_sum = caps.get("rank_viable_sum", 0)


class FakeContract:
    def __init__(self, root):
        self.root = root


class FakeTrade:
    def __init__(self, date, action, cash_after, campaign_id, ticker="AAA"):
        self.date = pd.Timestamp(date)
        self.action = action
        self.cash_after = cash_after
        self.campaign_id = campaign_id
        self.contract = FakeContract(ticker)


# -- metrics ----------------------------------------------------------------

def test_metrics_are_computed_from_the_equity_curve_alone():
    m = S.metrics_from_equity(curve([100, 110, 121]), 100.0)
    assert m["total_return"] == pytest.approx(0.21)
    assert m["pnl"] == pytest.approx(21.0)
    assert m["n_days"] == 3


def test_a_flat_curve_reports_no_ret_per_drawdown_rather_than_infinity():
    """Zero drawdown divides to infinity, which JSON cannot hold and a table
    cannot rank. None is the honest answer."""
    m = S.metrics_from_equity(curve([100, 100, 100]), 100.0)
    assert m["max_drawdown"] == 0
    assert m["ret_per_maxdd"] is None


def test_worst_year_is_the_minimum_of_the_yearly_returns():
    eq = pd.Series([100.0, 120.0, 90.0, 95.0],
                   index=pd.to_datetime(["2024-01-02", "2024-12-30",
                                         "2025-01-02", "2025-12-30"]))
    m = S.metrics_from_equity(eq, 100.0)
    assert set(m["by_year"]) == {"2024", "2025"}
    assert m["worst_year"] == pytest.approx(min(m["by_year"].values()))


def test_refusal_reasons_are_counted_and_ranked():
    warns = [(1, "entry_gated_illiquid", "SPY"),
             (2, "entry_gated_illiquid", "QQQ"),
             (3, "entry_gated_earnings", "AAPL")]
    m = S.metrics_from_result(FakeResult(curve([100, 101]), warnings=warns), 100.0)
    assert list(m["warning_counts"]) == ["entry_gated_illiquid",
                                         "entry_gated_earnings"]
    assert m["warning_counts"]["entry_gated_illiquid"] == 2


def test_the_slot_jam_counter_survives_into_the_scorecard():
    """`days_uncovered` is the n=1 diagnosis -- without it a jammed slot reads
    as a bad strategy."""
    m = S.metrics_from_result(FakeResult(curve([100, 101])), 100.0)
    assert m["days_uncovered"] == 2
    assert m["days_flat"] == 3
    assert m["campaigns"] == 4


# -- capacity: what the bot's slots were actually doing ---------------------

def test_pool_depth_and_starvation_come_off_the_engine_counters():
    r = FakeResult(curve([100, 101]), pool_depth_sum=90,
                   pool_depth_sessions=10, pool_starved_days=2,
                   idle_slot_days=7)
    m = S.metrics_from_result(r, 100.0, n_slots=5)
    assert m["pool_depth_mean"] == pytest.approx(9.0)
    assert m["pool_starved_pct"] == pytest.approx(0.2)
    assert m["idle_slot_days"] == 7


def test_a_run_with_no_routing_sessions_reports_no_pool_depth():
    """Zero sessions is not a depth of zero -- it is no observation at all, and
    dividing by it would print a confident 0.0 that means nothing."""
    m = S.metrics_from_result(FakeResult(curve([100, 101])), 100.0, n_slots=5)
    assert m["pool_depth_mean"] is None
    assert m["pool_starved_pct"] is None


def test_viable_share_lands_on_the_card_batch5_arm2():
    """rank_by="vrp_viable"'s mechanism clause ("non-trivially populated,
    REPORTED") reads this field -- without it on the card the run is not
    evidence. Ratio of sums across sessions, same style as pool_depth_mean."""
    r = FakeResult(curve([100, 101]), pool_depth_sum=40,
                   pool_depth_sessions=4, rank_viable_sum=10)
    m = S.metrics_from_result(r, 100.0, n_slots=5)
    assert m["rank_viable_share_mean"] == pytest.approx(0.25)


def test_a_run_with_no_routing_sessions_reports_no_viable_share():
    m = S.metrics_from_result(FakeResult(curve([100, 101])), 100.0, n_slots=5)
    assert m["rank_viable_share_mean"] is None


OPENING_CASH = 1_000.0


def _two_campaigns():
    """Campaign 1: clean, opens day 0, closes day 2, nets +$100.
    Campaign 2: assigned, opens day 2, still open at the end, nets -$500.
    `cash_after` runs from OPENING_CASH, the way the engine's does."""
    return [
        FakeTrade("2025-01-01", "SELL_PUT", 1_100.0, 1),
        FakeTrade("2025-01-03", "PUT_EXPIRED", 1_100.0, 1),
        FakeTrade("2025-01-03", "SELL_PUT", 1_200.0, 2),
        FakeTrade("2025-01-06", "ASSIGNED", 700.0, 2),
    ]


def test_slot_time_is_split_by_whether_the_campaign_was_assigned():
    eq = curve([100.0] * 6)          # six business days from 2025-01-01
    m = S.metrics_from_result(FakeResult(eq, trades=_two_campaigns()),
                              OPENING_CASH, n_slots=1)
    # campaign 1 held [day0, day2) = 2 sessions; campaign 2 held day2 -> end = 4
    assert m["slotdays_occupied"] == 6
    assert m["slotdays_assigned"] == 4
    assert m["slotdays_assigned_pct"] == pytest.approx(4 / 6)
    assert m["slot_fill_pct"] == pytest.approx(1.0)


def test_an_unfinished_campaign_is_excluded_from_the_per_slot_day_figures():
    """Its shares are still on the book, so its P&L is not a number yet.
    Counting it as a zero would understate every arm that ends mid-position."""
    eq = curve([100.0] * 6)
    m = S.metrics_from_result(FakeResult(eq, trades=_two_campaigns()),
                              OPENING_CASH, n_slots=1)
    assert m["campaigns_open_at_end"] == 1
    assert m["campaigns_clean"] == 1
    assert m["campaigns_assigned"] == 0
    # +$100 over the two sessions campaign 1 held the slot.
    assert m["pnl_per_slotday_clean"] == pytest.approx(50.0)
    assert m["pnl_per_slotday_assigned"] is None


def test_campaign_cash_is_the_difference_in_cash_after_so_friction_counts():
    """Price x size would miss the commission and fees, which are the whole
    question on an 11-day trade."""
    eq = curve([100.0] * 4)
    trades = [FakeTrade("2025-01-01", "SELL_PUT", 1_000.0, 1),
              FakeTrade("2025-01-02", "CLOSE_PUT", 940.0, 1),
              FakeTrade("2025-01-03", "PUT_EXPIRED", 940.0, 1)]
    m = S.metrics_from_result(FakeResult(eq, trades=trades),
                              OPENING_CASH, n_slots=1)
    # -60 over the two sessions the slot was held: the buy-back and its costs.
    assert m["pnl_per_slotday_clean"] == pytest.approx(-30.0)


def test_slot_metrics_are_absent_rather_than_wrong_without_a_slot_count():
    m = S.metrics_from_result(FakeResult(curve([100.0] * 6),
                                         trades=_two_campaigns()), OPENING_CASH)
    assert "slot_fill_pct" not in m
    assert m["slotdays_occupied"] == 6      # still knowable without n_slots


# -- the card ---------------------------------------------------------------

def make_card(tier="live", vhash="aaaa", bhash="bbbb", dirty=False):
    c = S.new("v", "frozen", tier, S and tier in ("live", "full"))
    c.arms["variant"] = S.metrics_from_equity(curve([100, 120]), 100.0)
    c.arms["base"] = S.metrics_from_equity(curve([100, 110]), 100.0)
    c.provenance = {"engine": {"sha": "abc123", "dirty": dirty},
                    "config_hash": {"variant": vhash, "base": bhash},
                    "data": {"hash": "d00d"}}
    return c


def test_delta_is_variant_minus_base():
    c = make_card()
    assert c.delta()["total_return"] == pytest.approx(0.10)


def test_a_scorecard_round_trips_through_disk(tmp_path):
    c = make_card()
    p = c.save(tmp_path)
    again = S.load(p)
    assert again.arms == c.arms
    assert again.config_hash == "aaaa"
    assert again.delta() == c.delta()


def test_scorecards_for_a_variant_come_back_newest_first(tmp_path):
    a = make_card(tier="live", vhash="a1")
    a.created = "2026-08-01T00:00:00+00:00"
    a.save(tmp_path)
    b = make_card(tier="full", vhash="b1")
    b.created = "2026-08-10T00:00:00+00:00"
    b.save(tmp_path)
    got = S.for_variant("v", tmp_path)
    assert [c.config_hash for c in got] == ["b1", "a1"]


# -- staleness: the anti-cherry-pick rule -----------------------------------

def test_a_scorecard_is_stale_once_the_variant_changes():
    c = make_card(vhash="aaaa")
    assert c.is_stale_for("aaaa") is None
    why = c.is_stale_for("zzzz")
    assert why and "the variant changed" in why


def test_a_scorecard_with_no_config_hash_is_stale_not_fresh():
    """Salvaged results from the old `results/` tree have no hash, because the
    harnesses never recorded one. Reading that as "matches" would make the
    least verifiable numbers in the project the easiest to promote on."""
    c = make_card(vhash="")
    why = c.is_stale_for("anything")
    assert why and "no config hash" in why


def test_a_scorecard_is_stale_once_the_BASE_changes():
    """A comparison against a master config that no longer exists is not a
    comparison. This is the case that bites after someone else promotes."""
    c = make_card(vhash="aaaa", bhash="bbbb")
    why = c.is_stale_for("aaaa", "cccc")
    assert why and "BASE changed" in why


def test_dirty_is_carried_on_the_card_not_inferred_later():
    assert make_card(dirty=True).dirty is True
    assert make_card(dirty=False).dirty is False


# -- rendering --------------------------------------------------------------

def test_render_labels_a_non_evidence_tier_as_such():
    c = S.new("v", "frozen", "smoke", False)
    c.arms["variant"] = S.metrics_from_equity(curve([100, 101]), 100.0)
    c.arms["base"] = S.metrics_from_equity(curve([100, 100.5]), 100.0)
    text = S.render(c)
    assert "not evidence" in text
    assert "RANKING, not a forecast" in text


def test_render_shouts_about_a_dirty_tree():
    assert "DIRTY TREE" in S.render(make_card(dirty=True))


def test_render_lists_declared_divergences():
    c = make_card()
    c.divergences = ["open_interest is present on 61% of chain rows"]
    assert "DECLARED DIVERGENCES" in S.render(c)
    assert "61%" in S.render(c)


# -- fingerprints -----------------------------------------------------------

def test_config_hash_ignores_key_order_but_not_values():
    a = fingerprint.config_hash({"x": 1, "y": 2})
    b = fingerprint.config_hash({"y": 2, "x": 1})
    c = fingerprint.config_hash({"x": 1, "y": 3})
    assert a == b and a != c


def test_config_hash_distinguishes_none_from_absent():
    """`liq_max_rel_spread = None` is a deliberate off with 25 lines of
    reasoning; omitting the knob entirely is a different config."""
    assert fingerprint.config_hash({"x": None}) != fingerprint.config_hash({})


def test_data_fingerprint_never_raises_on_a_relative_store(tmp_path):
    """The stores are repo-relative by default. An early version called
    `relative_to` on them against an absolute root and threw."""
    fp = fingerprint.data_fingerprint(["SPY"], "2025-01-01", "2025-02-01")
    assert set(fp) >= {"hash", "n_files", "bytes", "stores"}


def test_git_provenance_reports_a_sha_and_a_branch():
    g = fingerprint.git_provenance()
    assert len(g["sha"]) == 12
    assert isinstance(g["dirty"], bool)


def test_share_sales_are_counted_so_a_deep_stop_can_be_told_from_a_common_one():
    """The whole case for the assigned-stock exit ladder is that its last rung
    fires deep and rarely. `put_stop_mult` was falsified at every threshold and
    the -40% stop fired three times and did not survive its neighbours -- so
    "how often did it cut" has to be on the card, not inferred from n_trades."""
    c = FakeContract("AAA")
    trades = [FakeTrade("2025-01-01", "SELL_PUT", 1_000.0, 1),
              FakeTrade("2025-01-02", "ASSIGNED", 500.0, 1),
              FakeTrade("2025-01-03", "SOLD_SHARES", 900.0, 1),
              FakeTrade("2025-01-06", "SELL_PUT", 950.0, 2),
              FakeTrade("2025-01-07", "ASSIGNED", 400.0, 2)]
    m = S.metrics_from_result(FakeResult(curve([100.0] * 6), trades=trades),
                              OPENING_CASH, n_slots=1)
    assert m["assignments"] == 2
    assert m["shares_sold"] == 1
    assert m["sold_per_assignment"] == pytest.approx(0.5)


def test_no_assignments_reports_no_sale_rate_rather_than_zero():
    """Zero assignments is no observation, not a rate of zero -- the same rule
    the pool-depth metrics follow."""
    m = S.metrics_from_result(FakeResult(curve([100, 101])), 100.0)
    assert m["shares_sold"] == 0
    assert m["sold_per_assignment"] is None
