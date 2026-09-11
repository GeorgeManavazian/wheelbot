# tests/engine_v2/options/test_wheel_residual.py
import pandas as pd
import pytest
from src.engine_v2.options.wheel import WheelConfig, run_wheel

COLS = ["date","expiry","dte","strike","right","bid","ask","mid","close","delta","iv","underlying"]

def _chain(rows):
    ch = pd.DataFrame(rows, columns=COLS)
    ch["date"] = pd.to_datetime(ch["date"]); ch["expiry"] = pd.to_datetime(ch["expiry"])
    return ch

def test_residual_short_settled_at_window_end():
    # sell a put on d0; window ends d1 with the put STILL open (expiry is later, no TP)
    rows = [
        ["2024-01-02","2024-02-16",45,470,"P",2.00,2.10,2.05,2.05,-0.30,0.1,472.0],
        ["2024-01-03","2024-02-16",44,470,"P",1.50,1.60,1.55,1.55,-0.25,0.1,473.0],
    ]
    cfg = WheelConfig(starting_capital=50_000.0, put_delta=0.30, target_dte=45,
                      take_profit_pct=None, commission_per_contract=0.0)
    res = run_wheel(_chain(rows), cfg)
    assert res.residual_settled is True
    # settled to the last day's mid (1.55): cash = 50000 +200 (credit) -155 (buy back @mid) = 50045
    assert res.final_cash == pytest.approx(50_000 + 200 - 155)
    # final_cash + shares*spot reconciles to the last equity point
    assert res.equity.iloc[-1] == pytest.approx(res.final_cash + res.final_shares * 473.0)

def test_no_residual_when_flat_at_end():
    rows = [
        ["2024-01-02","2024-01-09",7,470,"P",2.00,2.10,2.05,2.05,-0.30,0.1,472.0],
        ["2024-01-09","2024-01-09",0,470,"P",0.00,0.05,0.02,0.02,-0.01,0.1,475.0],
    ]
    cfg = WheelConfig(starting_capital=50_000.0, put_delta=0.30, target_dte=7,
                      take_profit_pct=None, commission_per_contract=0.0)
    res = run_wheel(_chain(rows), cfg)
    assert res.residual_settled is False
    assert res.final_cash == pytest.approx(50_200.0)
