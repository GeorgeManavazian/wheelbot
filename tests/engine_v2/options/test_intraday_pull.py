import os, pandas as pd, pytest
from src.engine_v2.options.intraday import held_contracts, intraday_marks
from src.engine_v2.options.wheel import WheelResult
from src.engine_v2.options.chain import Contract

def _trade(date, action, strike, right):
    from src.engine_v2.options.wheel import Trade
    return Trade(pd.Timestamp(date), action, Contract("SPY", pd.Timestamp("2024-02-16"), strike, right), 1, 1.0, 0.0)

def test_held_contracts_from_trades():
    trades = [_trade("2024-01-05","SELL_PUT",470,"P"),
              _trade("2024-01-12","CLOSE_PUT",470,"P"),
              _trade("2024-01-12","SELL_PUT",475,"P"),
              _trade("2024-01-19","PUT_EXPIRED",475,"P")]
    res = WheelResult(pd.Series(dtype=float), trades, 0.0, 0)
    held = held_contracts(res)
    assert (pd.Timestamp("2024-02-16"), 470.0, "P", pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12")) in held
    assert any(h[1] == 475.0 and h[4] == pd.Timestamp("2024-01-19") for h in held)

def test_intraday_marks_keys():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-16 09:30","2024-01-16 10:30"]),
        "expiry": pd.to_datetime(["2024-01-19","2024-01-19"]),
        "strike": [470.0,470.0], "right":["P","P"], "close":[2.0,1.5],
        "high":[2.1,1.6],"low":[1.9,1.4],"volume":[10,20]})
    m = intraday_marks(df)
    k = (pd.Timestamp("2024-01-19"), 470.0, "P")
    assert k in m and list(m[k]["close"]) == [2.0, 1.5]

def test_held_contracts_understands_rolls_and_stops():
    import pandas as pd
    from src.engine_v2.options.wheel import WheelConfig, WheelResult, Trade
    from src.engine_v2.options.chain import Contract
    from src.engine_v2.options.intraday import held_contracts
    def t(date, action, strike, expiry):
        return Trade(pd.Timestamp(date), action,
                     Contract("SPY", pd.Timestamp(expiry), strike, "P"), 1, 1.0, 0.0)
    trades = [
        t("2024-01-02","SELL_PUT",470,"2024-01-09"),
        t("2024-01-04","ROLL_CLOSE",470,"2024-01-09"),
        t("2024-01-04","ROLL_OPEN",470,"2024-01-11"),
        t("2024-01-05","STOP_CLOSE",470,"2024-01-11"),
        t("2024-01-08","SELL_PUT",460,"2024-01-16"),
        t("2024-01-16","PUT_EXPIRED",460,"2024-01-16"),
    ]
    held = held_contracts(WheelResult(pd.Series(dtype=float), trades, 0.0, 0))
    assert len(held) == 3
    # rolled-away leg closes on the roll date, not later
    assert held[0][4] == pd.Timestamp("2024-01-04")
    # rolled-in leg exists and closes at the stop
    assert held[1][0] == pd.Timestamp("2024-01-11") and held[1][4] == pd.Timestamp("2024-01-05")
