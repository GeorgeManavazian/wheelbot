"""The fill-realism check: does a trade log describe fills a human could get?

The failure this guards against is a backtest filling at a stored quote on a
contract nobody traded. The check cannot prove a fill WAS possible -- EOD data
carries no size at the touch -- so every test here is about whether the four
implausibility measures are computed honestly and reported without hiding the
tail in a mean.
"""
from __future__ import annotations

import pandas as pd
import pytest

from bench import realism


class C:
    def __init__(self, root, expiry, strike, right="P"):
        self.root, self.strike, self.right = root, strike, right
        self.expiry = pd.Timestamp(expiry)


class T:
    def __init__(self, date, action, contract, contracts, price=1.0, cid=1):
        self.date = pd.Timestamp(date)
        self.action, self.contract, self.contracts = action, contract, contracts
        self.price_per_contract, self.campaign_id = price, cid


def chain(rows):
    return pd.DataFrame(rows)


def row(date, expiry, strike, bid, ask, oi, vol, right="P"):
    return {"date": pd.Timestamp(date), "expiry": pd.Timestamp(expiry),
            "strike": strike, "right": right, "bid": bid, "ask": ask,
            "open_interest": oi, "volume": vol}


CH = {"AAA": chain([row("2025-01-02", "2025-01-17", 50.0, 1.00, 1.10, 400, 40),
                    row("2025-01-09", "2025-01-17", 50.0, 0.40, 0.44, 380, 12),
                    row("2025-01-02", "2025-01-17", 40.0, 0.20, 0.60, 8, 0)])}


def test_a_fill_is_joined_to_the_market_it_filled_into():
    t = T("2025-01-02", "SELL_PUT", C("AAA", "2025-01-17", 50.0), 4)
    df = realism.check_fills([t], CH)
    r = df.iloc[0]
    assert r["row_found"] and r["open_interest"] == 400 and r["volume"] == 40
    assert r["pct_of_volume"] == pytest.approx(4 / 40)
    assert r["pct_of_oi"] == pytest.approx(4 / 400)
    assert r["rel_spread"] == pytest.approx(0.10 / 1.05)
    assert r["dte"] == 15


def test_settlement_events_are_not_checked_as_fills():
    """Nobody has to be on the other side of an expiry or an assignment.
    Counting them would dilute the statistic this module exists to compute.

    A-4 (2026-08-17) changed the EXPORT contract -- settlements now appear as
    rows, tagged leg="settlement", so per-stop attribution reads from the log
    instead of being inferred -- but the CHECK contract is unchanged: no
    market columns are looked up for them and summarise() excludes them from
    every fill statistic."""
    c = C("AAA", "2025-01-17", 50.0)
    trades = [T("2025-01-02", "SELL_PUT", c, 1),
              T("2025-01-17", "PUT_EXPIRED", c, 1),
              T("2025-01-17", "ASSIGNED", c, 1)]
    df = realism.check_fills(trades, CH)
    assert list(df[df["leg"] == "option"]["action"]) == ["SELL_PUT"]
    assert list(df[df["leg"] == "settlement"]["action"]) == \
        ["PUT_EXPIRED", "ASSIGNED"]
    assert df[df["leg"] == "settlement"]["bid"].isna().all()
    assert realism.summarise(df)["fills"] == 1


def test_a_strike_that_did_not_trade_is_flagged_not_averaged_away():
    t = T("2025-01-02", "SELL_PUT", C("AAA", "2025-01-17", 40.0), 1)
    df = realism.check_fills([t], CH)
    assert bool(df.iloc[0]["zero_volume"]) is True
    # A zero denominator must not silently become a ratio of zero, which would
    # read as "took 0% of the day's volume" -- the most flattering possible
    # answer to the least plausible fill in the log.
    assert df.iloc[0]["pct_of_volume"] is None
    assert realism.summarise(df)["zero_volume_fills"] == 1


def test_a_fill_whose_chain_row_is_missing_is_a_finding_not_a_blank():
    t = T("2025-01-03", "SELL_PUT", C("AAA", "2025-01-17", 50.0), 1)
    s = realism.summarise(realism.check_fills([t], CH))
    assert s["rows_not_found"] == 1


def test_entries_and_exits_are_summarised_apart():
    """A position you can open and cannot close is the failure that matters,
    and it lives entirely in the exit rows."""
    c = C("AAA", "2025-01-17", 50.0)
    trades = [T("2025-01-02", "SELL_PUT", c, 4),
              T("2025-01-09", "CLOSE_PUT", c, 4)]
    s = realism.summarise(realism.check_fills(trades, CH))
    assert s["entries"] == 1 and s["exits"] == 1
    # the exit is into a thinner day: 4 of 12 traded, against 4 of 40 on entry
    assert s["exit_pct_of_volume_median"] == pytest.approx(4 / 12)
    assert s["entry_pct_of_volume_median"] == pytest.approx(4 / 40)


def test_the_summary_reports_the_tail_not_just_the_middle():
    """One 700-contract order among 300 sane ones is the exact shape being
    looked for, and a mean would bury it."""
    c = C("AAA", "2025-01-17", 50.0)
    trades = [T("2025-01-02", "SELL_PUT", c, 1)] * 9 + \
             [T("2025-01-02", "SELL_PUT", c, 40)]
    s = realism.summarise(realism.check_fills(trades, CH))
    assert s["entry_pct_of_volume_median"] == pytest.approx(1 / 40)
    assert s["entry_pct_of_volume_max"] == pytest.approx(1.0)
    assert s["over_volume_cap"] == 1


def test_render_survives_an_empty_log():
    assert "no fills" in realism.render(realism.summarise(pd.DataFrame()))
