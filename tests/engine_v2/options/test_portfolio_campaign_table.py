import pandas as pd
from src.engine_v2.options.chain import Contract
from src.engine_v2.options.wheel import Trade, WheelConfig
from src.engine_v2.options.portfolio import PortfolioResult
from src.engine_v2.options.report import portfolio_campaign_table

MULT, COMM = 100, 0.65


def _cfg():
    return WheelConfig(ticker="SPY", put_delta=0.20, call_delta=0.50,
                       target_dte=7, take_profit_pct=0.50, call_min_strike="basis")


def _result():
    d = pd.Timestamp("2021-01-04")
    put1 = Contract("SPY", pd.Timestamp("2021-01-15"), 400.0, "P")
    put2 = Contract("GDX", pd.Timestamp("2021-01-15"), 30.0, "P")
    trades = [
        # campaign 1: SPY put sold for 2.00, expires worthless -> closed winner
        Trade(d, "SELL_PUT", put1, 1, 2.00, 100_000.0, campaign_id=1),
        Trade(pd.Timestamp("2021-01-15"), "PUT_EXPIRED", put1, 1, 0.0, 100_000.0, campaign_id=1),
        # campaign 2: GDX put sold for 1.00, assigned at 30, still holding -> open
        Trade(d, "SELL_PUT", put2, 1, 1.00, 100_000.0, campaign_id=2),
        Trade(pd.Timestamp("2021-01-15"), "ASSIGNED", put2, 1, 30.0, 97_000.0, campaign_id=2),
    ]
    return PortfolioResult(pd.Series({d: 100_000.0}), trades, 97_000.0,
                           {"GDX": 100}, n_campaigns_opened=2)


def test_two_campaigns_one_closed_one_open():
    ct = portfolio_campaign_table(_result(), _cfg(), {"SPY": 405.0, "GDX": 25.0})
    assert list(ct["campaign_id"]) == [1, 2]
    c1 = ct[ct.campaign_id == 1].iloc[0]
    c2 = ct[ct.campaign_id == 2].iloc[0]
    # campaign 1: premium in, expired worthless -> realized = 2*100 - 0.65
    assert abs(c1["pnl_realized"] - (2.00 * MULT - COMM)) < 1e-9
    assert c1["open_at_end"] == False
    assert c1["shares_held"] == 0
    assert abs(c1["collateral"] - 400.0 * MULT) < 1e-9
    # campaign 2: premium 1*100-0.65, then assigned -3000, holding 100 shares
    assert abs(c2["pnl_realized"] - (1.00 * MULT - COMM - 30.0 * MULT)) < 1e-9
    assert c2["open_at_end"] == True
    assert c2["shares_held"] == 100
    # pnl_mtm marks the 100 GDX shares at last_spot 25
    assert abs(c2["pnl_mtm"] - (c2["pnl_realized"] + 100 * 25.0)) < 1e-9


def test_open_short_put_at_end_is_open_and_marked():
    d = pd.Timestamp("2021-01-04")
    put = Contract("SPY", pd.Timestamp("2021-01-15"), 400.0, "P")
    trades = [
        # single campaign: SELL_PUT only, never closed -> still open short at end
        Trade(d, "SELL_PUT", put, 1, 2.00, 100_000.0, campaign_id=1),
    ]
    result = PortfolioResult(pd.Series({d: 100_000.0}), trades, 100_000.0,
                             {}, n_campaigns_opened=1)
    ct = portfolio_campaign_table(result, _cfg(), {"SPY": 380.0})
    row = ct[ct.campaign_id == 1].iloc[0]
    assert row["open_at_end"] == True
    expected_pnl_mtm = (2.00 * MULT - COMM) - max(400.0 - 380.0, 0.0) * MULT * 1
    assert abs(row["pnl_mtm"] - expected_pnl_mtm) < 1e-9
    assert abs(row["pnl_mtm"] - (-1800.65)) < 1e-9
