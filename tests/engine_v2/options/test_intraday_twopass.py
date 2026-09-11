import pandas as pd, pytest
from src.engine_v2.options.wheel import WheelConfig, run_wheel
from src.engine_v2.options.intraday import run_wheel_intraday

COLS = ["date","expiry","dte","strike","right","bid","ask","mid","close","delta","iv","underlying"]
def _chain(rows):
    ch = pd.DataFrame(rows, columns=COLS)
    ch["date"]=pd.to_datetime(ch["date"]); ch["expiry"]=pd.to_datetime(ch["expiry"]); return ch

def test_two_pass_intraday_changes_tp():
    rows = [
        ["2024-01-02","2024-01-19",17,470,"P",2.00,2.10,2.05,2.05,-0.30,0.1,472.0],
        ["2024-01-03","2024-01-19",16,470,"P",1.75,1.80,1.775,1.775,-0.25,0.1,473.0],
    ]
    ch = _chain(rows)
    intraday_df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-03 11:30","2024-01-03 12:30"]),
        "expiry": pd.to_datetime(["2024-01-19","2024-01-19"]),
        "strike":[470.0, 470.0], "right":["P","P"],
        "close":[0.90, 1.10], "high":[1.0, 1.2], "low":[0.85, 1.05], "volume":[10, 10]})
    cfg = WheelConfig(starting_capital=50_000.0, put_delta=0.30, target_dte=17,
                      take_profit_pct=0.50, commission_per_contract=0.0)
    eod = run_wheel(ch, cfg)
    intr = run_wheel_intraday(ch, cfg, intraday_df)
    # EOD never TP's here (ask 1.80 > 1.00); intraday does: the 11:30 bar crosses
    # (0.90 <= 1.00) and fills at the NEXT bar's close (12:30 @ 1.10)
    assert "CLOSE_PUT" not in [t.action for t in eod.trades]
    assert "CLOSE_PUT" in [t.action for t in intr.trades]
