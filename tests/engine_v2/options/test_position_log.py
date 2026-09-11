import os, pandas as pd, pytest
from src.engine_v2.options.wheel import WheelConfig, run_wheel, Trade
from src.engine_v2.options.chain import Contract
from src.engine_v2.options.report import position_log

def _t(date, action, strike, right, n, price):
    return Trade(pd.Timestamp(date), action, Contract("SPY", pd.Timestamp("2024-02-16"), strike, right), n, price, 0.0)

def test_blotter_outcomes_and_pnl():
    from src.engine_v2.options.wheel import WheelResult
    trades = [
        _t("2024-01-02","SELL_PUT",470,"P",1,2.00),   # credit 200
        _t("2024-01-05","CLOSE_PUT",470,"P",1,0.90),   # cost 90 -> pnl 110, Took profit
        _t("2024-01-08","SELL_PUT",460,"P",1,1.50),    # credit 150
        _t("2024-02-16","ASSIGNED",460,"P",1,460),     # pnl 150 kept, Assigned; shares in @460
        _t("2024-02-16","SELL_CALL",470,"C",1,3.00),   # credit 300
        _t("2024-03-15","CALLED_AWAY",470,"C",1,470),  # call pnl 300; shares sold @470 -> share pnl (470-460)*100 = 1000
    ]
    res = WheelResult(pd.Series(dtype=float), trades, 0.0, 0)
    df = position_log(res, WheelConfig(commission_per_contract=0.0))
    puts = df[df.instrument=="PUT"]
    assert (puts.outcome.tolist()) == ["Took profit","Assigned"]
    assert puts.iloc[0].realized_pnl == pytest.approx(110)
    assert puts.iloc[1].realized_pnl == pytest.approx(150)
    call = df[df.instrument=="CALL"].iloc[0]
    assert call.outcome == "Called away" and call.realized_pnl == pytest.approx(300)
    sh = df[df.instrument=="SHARES"].iloc[0]
    assert sh.realized_pnl == pytest.approx(1000) and sh.outcome == "Called away"

def test_blotter_rolled_outcome():
    from src.engine_v2.options.wheel import WheelResult
    trades = [
        _t("2024-01-02","SELL_PUT",470,"P",1,2.00),    # credit 200
        _t("2024-02-16","ROLL_CLOSE",470,"P",1,5.10),  # cost 510 -> pnl -310, Rolled
    ]
    res = WheelResult(pd.Series(dtype=float), trades, 0.0, 0)
    df = position_log(res, WheelConfig(commission_per_contract=0.0))
    row = df.iloc[0]
    assert row.outcome == "Rolled"
    assert row.realized_pnl == pytest.approx(200 - 510)

def test_blotter_liquidated_outcome():
    from src.engine_v2.options.wheel import WheelResult
    trades = [
        _t("2024-01-08","SELL_PUT",460,"P",1,1.50),    # credit 150
        _t("2024-02-16","ASSIGNED",460,"P",1,460),     # shares in @460
        _t("2024-02-16","LIQUIDATE",460,"P",1,452.0),  # dumped at spot 452
    ]
    res = WheelResult(pd.Series(dtype=float), trades, 0.0, 0)
    df = position_log(res, WheelConfig(commission_per_contract=0.0))
    put = df[df.instrument == "PUT"].iloc[0]
    assert put.outcome == "Assigned" and put.realized_pnl == pytest.approx(150)
    sh = df[df.instrument == "SHARES"].iloc[0]
    assert sh.outcome == "Liquidated"
    assert sh.realized_pnl == pytest.approx((452.0 - 460) * 100 * 1)
    assert sh.closed == pd.Timestamp("2024-02-16")

def test_blotter_stopped_outcome():
    from src.engine_v2.options.wheel import WheelResult
    trades = [
        _t("2024-01-02","SELL_PUT",470,"P",1,2.00),    # credit 200
        _t("2024-01-10","STOP_CLOSE",470,"P",1,6.30),  # cost 630 -> pnl -430, Stopped
    ]
    res = WheelResult(pd.Series(dtype=float), trades, 0.0, 0)
    df = position_log(res, WheelConfig(commission_per_contract=0.0))
    row = df.iloc[0]
    assert row.outcome == "Stopped"
    assert row.realized_pnl == pytest.approx(200 - 630)

def test_blotter_reconciles_to_total():
    FIX = "fixtures/spy_wheel_cycle.parquet"
    if not os.path.exists(FIX): pytest.skip("fixture missing")
    ch = pd.read_parquet(FIX); cfg = WheelConfig(target_dte=40, put_delta=0.30, call_delta=0.30)
    res = run_wheel(ch, cfg); df = position_log(res, cfg)
    total = res.final_cash - cfg.starting_capital
    assert df["realized_pnl"].sum() == pytest.approx(total, abs=0.01)

def test_blotter_reconciles_on_truncated_window():
    FIX = "fixtures/spy_wheel_cycle.parquet"
    if not os.path.exists(FIX):
        pytest.skip("fixture missing")
    ch = pd.read_parquet(FIX)
    # truncate so the run ends with a short still open (residual-settled)
    cut = sorted(ch["date"].unique())[20]
    ch2 = ch[ch["date"] <= cut]
    cfg = WheelConfig(target_dte=40, put_delta=0.30, call_delta=0.30)
    res = run_wheel(ch2, cfg)
    df = position_log(res, cfg)
    total = res.final_cash - cfg.starting_capital
    assert df["realized_pnl"].dropna().sum() == pytest.approx(total, abs=0.01)

# ---- repair pass: roll rows + campaign ids in the blotter ----

def test_roll_open_starts_its_own_row():
    trades = [
        _t("2024-01-02","SELL_PUT",470,"P",1,2.00),
        _t("2024-01-04","ROLL_CLOSE",470,"P",1,3.10),
        _t("2024-01-04","ROLL_OPEN",460,"P",1,3.20),
        _t("2024-01-11","PUT_EXPIRED",460,"P",1,0.0),
    ]
    from src.engine_v2.options.wheel import WheelResult
    res = WheelResult(pd.Series(dtype=float), trades, 0.0, 0)
    log = position_log(res, WheelConfig(commission_per_contract=0.0))
    opts = log[log["instrument"] == "PUT"]
    assert list(opts["outcome"]) == ["Rolled", "Expired worthless"]
    assert "campaign_id" in log.columns
