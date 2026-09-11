import pandas as pd
from src.engine_v2.options.chain import Mark
from src.engine_v2.options.wheel import (
    WheelConfig, Trade, underlying_series, sell_proceeds, buy_cost)

def test_config_defaults():
    c = WheelConfig()
    assert c.starting_capital == 100_000.0 and c.commission_per_contract == 0.65
    assert c.contract_multiplier == 100 and c.take_profit_pct == 0.50
    assert c.put_delta == 0.20 and c.call_delta == 0.20
    assert c.target_dte == 7 and c.cash_yield == 0.0 and c.ticker == "SPY"

def test_underlying_series_one_per_date():
    cols = ["date","expiry","dte","strike","right","bid","ask","mid","close","delta","iv","underlying"]
    rows = [
        ["2024-01-02","2024-02-16",45,470,"P",1,1.1,1.05,1.05,-0.3,0.1,472.0],
        ["2024-01-02","2024-02-16",45,475,"P",2,2.1,2.05,2.05,-0.4,0.1,472.0],
        ["2024-01-03","2024-02-16",44,470,"P",1,1.1,1.05,1.05,-0.3,0.1,473.5],
    ]
    ch = pd.DataFrame(rows, columns=cols)
    ch["date"] = pd.to_datetime(ch["date"]); ch["expiry"] = pd.to_datetime(ch["expiry"])
    u = underlying_series(ch)
    assert u[pd.Timestamp("2024-01-02")] == 472.0
    assert u[pd.Timestamp("2024-01-03")] == 473.5

def test_fills_cross_spread_and_commission():
    cfg = WheelConfig()
    m = Mark(bid=2.00, ask=2.10, mid=2.05)
    # sell 3 contracts: +2.00*100*3 - 0.65*3
    assert sell_proceeds(m, 3, cfg) == 2.00*100*3 - 0.65*3
    # buy-to-close 3: 2.10*100*3 + 0.65*3
    assert buy_cost(m, 3, cfg) == 2.10*100*3 + 0.65*3
