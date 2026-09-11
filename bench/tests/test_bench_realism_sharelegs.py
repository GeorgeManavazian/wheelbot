"""A-4 (batch 2 pre-work): share legs in the fills export.

Batch 1's per-stop attribution (which fire realized what, against which
assignment) had to be INFERRED from option rows and cash arithmetic; the
batch-2 grid grades its section-2 predictions from the logs directly. So
settlements (ASSIGNED / CALLED_AWAY / PUT_EXPIRED / CALL_EXPIRED) and stock
fills (SOLD_SHARES) become rows in the export, tagged by `leg`, with
`cash_after` carried on every row.

The realism STATISTICS must not move: settlements are not fills -- nobody has
to be on the other side of an expiry -- so `summarise` keeps operating on the
option legs only (the module's own founding rule).
"""
from __future__ import annotations

import pandas as pd
import pytest

from bench import realism


class FakeContract:
    def __init__(self, root, strike=100.0, right="P", expiry="2025-02-21"):
        self.root = root
        self.strike = strike
        self.right = right
        self.expiry = pd.Timestamp(expiry)


class FakeTrade:
    def __init__(self, date, action, contract, contracts=1, price=1.0,
                 cash_after=10_000.0, campaign_id=1):
        self.date = pd.Timestamp(date)
        self.action = action
        self.contract = contract
        self.contracts = contracts
        self.price_per_contract = price
        self.cash_after = cash_after
        self.campaign_id = campaign_id


def chain_for(c, date="2025-01-06", bid=1.0, ask=1.2, oi=500, vol=50):
    return {c.root: pd.DataFrame({
        "date": [pd.Timestamp(date)], "expiry": [c.expiry],
        "strike": [c.strike], "right": [c.right],
        "bid": [bid], "ask": [ask], "open_interest": [oi], "volume": [vol]})}


def mixed_trades():
    c = FakeContract("AAA")
    cc = FakeContract("AAA", right="C")
    return [
        FakeTrade("2025-01-06", "SELL_PUT", c, cash_after=10_200.0),
        FakeTrade("2025-02-21", "ASSIGNED", c, cash_after=200.0),
        FakeTrade("2025-03-03", "SELL_CALL", cc, cash_after=350.0),
        FakeTrade("2025-03-21", "CALLED_AWAY", cc, cash_after=10_350.0),
        FakeTrade("2025-04-04", "SOLD_SHARES", "BBB", contracts=100,
                  price=42.0, cash_after=14_550.0, campaign_id=2),
        FakeTrade("2025-04-21", "PUT_EXPIRED", c, cash_after=14_550.0,
                  campaign_id=3),
    ], chain_for(c)


def test_settlements_and_stock_sales_become_rows():
    trades, chains = mixed_trades()
    df = realism.check_fills(trades, chains)
    got = df.set_index("action")["leg"].to_dict()
    assert got["ASSIGNED"] == "settlement"
    assert got["CALLED_AWAY"] == "settlement"
    assert got["PUT_EXPIRED"] == "settlement"
    assert got["SOLD_SHARES"] == "stock"
    assert got["SELL_PUT"] == "option"
    assert got["SELL_CALL"] == "option"


def test_a_stock_sale_row_carries_the_bare_ticker_size_and_price():
    trades, chains = mixed_trades()
    df = realism.check_fills(trades, chains)
    r = df[df["action"] == "SOLD_SHARES"].iloc[0]
    assert r["ticker"] == "BBB"
    assert r["contracts"] == 100          # share count for a stock leg
    assert r["fill_price"] == pytest.approx(42.0)
    assert r["campaign"] == 2


def test_cash_after_is_on_every_row_for_attribution():
    """The whole point of A-4: batch 1's per-stop attribution had to be
    inferred; cash_after per row makes it arithmetic."""
    trades, chains = mixed_trades()
    df = realism.check_fills(trades, chains)
    assert "cash_after" in df.columns
    assert df["cash_after"].notna().all()
    assert float(df[df["action"] == "ASSIGNED"]["cash_after"].iloc[0]) == 200.0


def test_settlement_rows_never_dilute_the_realism_statistics():
    """Settlements are not fills. The founding rule of this module: checking
    them would dilute the statistic it exists to compute."""
    trades, chains = mixed_trades()
    df = realism.check_fills(trades, chains)
    s = realism.summarise(df)
    assert s["fills"] == 2                # SELL_PUT + SELL_CALL only
    assert s["settlements"] == 3
    assert s["stock_fills"] == 1
    # rows_not_found counts option legs only: the SELL_CALL's chain row does
    # not exist in the fixture, the settlement rows must not add to it.
    assert s["rows_not_found"] == 1


def test_option_only_logs_are_byte_identical_in_the_summary():
    c = FakeContract("AAA")
    trades = [FakeTrade("2025-01-06", "SELL_PUT", c, cash_after=10_200.0)]
    s = realism.summarise(realism.check_fills(trades, chain_for(c)))
    assert s["fills"] == 1
    assert s["settlements"] == 0
    assert s["stock_fills"] == 0
