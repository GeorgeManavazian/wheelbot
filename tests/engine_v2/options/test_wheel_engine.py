import os
import pandas as pd
import pytest
from src.engine_v2.options.wheel import WheelConfig, run_wheel

COLS = ["date","expiry","dte","strike","right","bid","ask","mid","close","delta","iv","underlying"]

CYCLE_FIX = "fixtures/spy_wheel_cycle.parquet"

def _chain(rows):
    ch = pd.DataFrame(rows, columns=COLS)
    ch["date"] = pd.to_datetime(ch["date"]); ch["expiry"] = pd.to_datetime(ch["expiry"])
    return ch

def _cycle_chain():
    if not os.path.exists(CYCLE_FIX):
        pytest.skip("wheel cycle fixture not built")
    return pd.read_parquet(CYCLE_FIX)

# small cfg: 1 contract, no take-profit unless a test wants it.
# Deltas pinned to 0.30 (the old default) so nearest-delta selection keeps
# picking the same synthetic rows these tests were written around.
def _cfg(**kw):
    base = dict(starting_capital=50_000.0, put_delta=0.30, call_delta=0.30,
                take_profit_pct=None, commission_per_contract=0.0)
    base.update(kw); return WheelConfig(**base)

def test_put_expires_otm_keeps_credit():
    # sell 7-DTE put strike 470 for 2.00 on d0; at expiry spot 475 > 470 -> OTM
    rows = [
        ["2024-01-02","2024-01-09",7,470,"P",2.00,2.10,2.05,2.05,-0.30,0.1,472.0],
        ["2024-01-09","2024-01-09",0,470,"P",0.00,0.05,0.02,0.02,-0.01,0.1,475.0],
    ]
    res = run_wheel(_chain(rows), _cfg())
    acts = [t.action for t in res.trades]
    assert "SELL_PUT" in acts and "PUT_EXPIRED" in acts and "ASSIGNED" not in acts
    # 1 contract, credit 2.00*100 = 200 kept; final cash = 50000 + 200
    assert res.final_cash == pytest.approx(50_200.0)
    assert res.final_shares == 0

def test_put_itm_assigned_then_call_called_away():
    # put strike 470 ITM at expiry (spot 465) -> assigned 100 sh @470
    # then covered call strike 475 sold, ITM at its expiry (spot 480) -> called away @475
    # A9 (2026-08-02): the original fixture sold the call ON assignment day --
    # that encoded the defect (notice arrives after the close; shares settle
    # next session). The call now sells on the NEXT session's row (01-10) at
    # that session's price; cash walk re-pinned accordingly.
    rows = [
        ["2024-01-02","2024-01-09",7,470,"P",2.00,2.10,2.05,2.05,-0.30,0.1,472.0],
        ["2024-01-09","2024-01-09",0,470,"P",5.00,5.10,5.05,5.05,-0.99,0.1,465.0],
        ["2024-01-09","2024-01-16",7,475,"C",3.00,3.10,3.05,3.05, 0.30,0.1,465.0],
        ["2024-01-10","2024-01-16",6,475,"C",2.90,3.00,2.95,2.95, 0.30,0.1,466.0],
        ["2024-01-16","2024-01-16",0,475,"C",5.00,5.10,5.05,5.05, 0.99,0.1,480.0],
    ]
    res = run_wheel(_chain(rows), _cfg())
    acts = [t.action for t in res.trades]
    assert acts == ["SELL_PUT","ASSIGNED","SELL_CALL","CALLED_AWAY"]
    assert [t for t in res.trades if t.action == "SELL_CALL"][0].date == \
        pd.Timestamp("2024-01-10")
    # cash walk (1 contract, mult 100, commission 0):
    # start 50000; +200 put credit; -47000 assigned; +290 call credit (D+1
    # session's bid 2.90, NOT assignment day's 3.00); +47500 called away
    assert res.final_cash == pytest.approx(50_000 + 200 - 47_000 + 290 + 47_500)
    assert res.final_shares == 0

def test_take_profit_closes_short_put():
    # sell put for 2.00; next day ask drops to 0.90 (< 50% of 2.00 -> 1.00) -> buy to close
    rows = [
        ["2024-01-02","2024-01-19",17,470,"P",2.00,2.10,2.05,2.05,-0.30,0.1,472.0],
        ["2024-01-03","2024-01-19",16,470,"P",0.80,0.90,0.85,0.85,-0.15,0.1,476.0],
    ]
    res = run_wheel(_chain(rows), _cfg(take_profit_pct=0.50, target_dte=17))
    acts = [t.action for t in res.trades]
    assert acts[:2] == ["SELL_PUT","CLOSE_PUT"]
    # only the 470/01-19 put exists on 01-03 == the contract just closed, so NO
    # same-day re-entry (anti-churn guard): +200 credit, -90 to close => +110
    assert res.final_cash == pytest.approx(50_000 + 200 - 90)
    assert acts.count("SELL_PUT") == 1

def test_take_profit_same_day_reentry_into_different_strike():
    # sell 470 put; next day it hits take-profit AND a fresh ~0.30-delta 475 put is
    # available -> re-enter SAME DAY into the different strike (owner chose policy B).
    rows = [
        ["2024-01-02","2024-01-19",17,470,"P",2.00,2.10,2.05,2.05,-0.30,0.1,472.0],
        ["2024-01-03","2024-01-19",16,470,"P",0.80,0.90,0.85,0.85,-0.15,0.1,478.0],
        ["2024-01-03","2024-01-19",16,475,"P",1.50,1.60,1.55,1.55,-0.30,0.1,478.0],
    ]
    res = run_wheel(_chain(rows), _cfg(take_profit_pct=0.50, target_dte=17))
    acts = [t.action for t in res.trades]
    assert acts == ["SELL_PUT","CLOSE_PUT","SELL_PUT"]      # re-entered same day
    reentry = [t for t in res.trades if t.action == "SELL_PUT"][1]
    assert reentry.contract.strike == 475.0                 # different strike, not 470
    # 50000 +200 (sell 470) -90 (close 470) +150 (sell 475) -155 (settle 475 @mid 1.55) = 50105
    assert res.final_cash == pytest.approx(50_000 + 200 - 90 + 150 - 155)

def test_sizing_multiple_contracts_and_commission():
    # cash 50000, strike 100 -> floor(50000/10000)=5 contracts
    rows = [
        ["2024-01-02","2024-01-09",7,100,"P",1.00,1.10,1.05,1.05,-0.30,0.1,102.0],
        ["2024-01-09","2024-01-09",0,100,"P",0.00,0.05,0.02,0.02,-0.01,0.1,105.0],
    ]
    res = run_wheel(_chain(rows), _cfg(starting_capital=50_000.0, commission_per_contract=0.65))
    sell = [t for t in res.trades if t.action == "SELL_PUT"][0]
    assert sell.contracts == 5   # floor(50000/(100*100)) = 5
    # credit 1.00*100*5 - 0.65*5 = 500 - 3.25
    assert res.final_cash == pytest.approx(50_000 + 500 - 3.25)

def test_equity_identity_reconciles():
    rows = [
        ["2024-01-02","2024-01-19",17,470,"P",2.00,2.10,2.05,2.05,-0.30,0.1,472.0],
        ["2024-01-03","2024-01-19",16,470,"P",1.50,1.60,1.55,1.55,-0.25,0.1,473.0],
    ]
    res = run_wheel(_chain(rows), _cfg(target_dte=17))
    # on d0: cash 50200, no shares, short liability = mid 2.05*100 = 205 -> equity 49995
    assert res.equity.iloc[0] == pytest.approx(50_000 + 200 - 205)

# ---- new-config behavior (target_dte / cash_yield / ticker / days_flat) ----

def test_config_has_no_dte_window():
    import dataclasses
    names = {f.name for f in dataclasses.fields(WheelConfig)}
    assert "target_dte" in names and "cash_yield" in names and "ticker" in names
    assert "dte_min" not in names and "dte_max" not in names

def test_contracts_carry_ticker():
    ch = _cycle_chain()
    res = run_wheel(ch, WheelConfig(target_dte=30, ticker="GDX"))
    sells = [t for t in res.trades if t.action.startswith("SELL")]
    assert sells and all(t.contract.root == "GDX" for t in sells)

def test_cash_yield_accrues_on_idle_cash():
    ch = _cycle_chain()
    flat = run_wheel(ch, WheelConfig(target_dte=30, put_delta=0.0001))  # never trades
    paid = run_wheel(ch, WheelConfig(target_dte=30, put_delta=0.0001, cash_yield=0.05))
    days = (ch["date"].max() - ch["date"].min()).days
    expect = 100_000.0 * ((1 + 0.05 / 365) ** days - 1)
    assert paid.equity.iloc[-1] - flat.equity.iloc[-1] == pytest.approx(expect, rel=0.02)

def test_days_flat_counted():
    ch = _cycle_chain()
    res = run_wheel(ch, WheelConfig(target_dte=7))
    assert res.days_flat >= 0
    never = run_wheel(ch[ch["dte"] > 200], WheelConfig(target_dte=7))
    assert never.days_flat == len(never.equity)

def test_hold_to_expiry_at_tp_100():
    ch = _cycle_chain()
    res = run_wheel(ch, WheelConfig(target_dte=30, take_profit_pct=1.0))
    assert not [t for t in res.trades if t.action.startswith("CLOSE")]
