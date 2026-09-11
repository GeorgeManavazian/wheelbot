"""Campaign accounting: every trade carries the id of the wheel saga it belongs
to (entry -> defenses -> assignment -> calls -> exit = one campaign)."""
import pandas as pd
import pytest
from src.engine_v2.options.wheel import WheelConfig, run_wheel

COLS = ["date","expiry","dte","strike","right","bid","ask","mid","close","delta","iv","underlying"]

def _chain(rows):
    ch = pd.DataFrame(rows, columns=COLS)
    ch["date"] = pd.to_datetime(ch["date"]); ch["expiry"] = pd.to_datetime(ch["expiry"])
    return ch

def _cfg(**kw):
    base = dict(starting_capital=50_000.0, put_delta=0.30, call_delta=0.30,
                take_profit_pct=None, commission_per_contract=0.0)
    base.update(kw); return WheelConfig(**base)

# saga: put sold, assigned, call sold, called away -> ONE campaign; the next
# put entry -> campaign 2. A9 (2026-08-02): the call row moved off assignment
# day (01-09 -> 01-10) -- the same-day sale encoded the defect.
_SAGA = [
    ["2024-01-02","2024-01-09",7,470,"P",2.00,2.10,2.05,2.05,-0.30,0.1,472.0],
    ["2024-01-09","2024-01-09",0,470,"P",5.00,5.10,5.05,5.05,-0.99,0.1,465.0],
    ["2024-01-10","2024-01-16",6,470,"C",2.00,2.10,2.05,2.05,0.30,0.1,465.0],
    ["2024-01-16","2024-01-16",0,470,"C",5.00,5.10,5.05,5.05,0.99,0.1,476.0],
    ["2024-01-16","2024-01-23",7,472,"P",2.00,2.10,2.05,2.05,-0.30,0.1,476.0],
    ["2024-01-23","2024-01-23",0,472,"P",0.05,0.10,0.075,0.075,-0.01,0.1,480.0],
]

def test_full_saga_is_one_campaign_next_entry_is_new():
    res = run_wheel(_chain(_SAGA), _cfg())
    per = [(t.action, t.campaign_id) for t in res.trades]
    assert ("ASSIGNED", 1) in per
    assert ("SELL_CALL", 1) in per
    assert ("CALLED_AWAY", 1) in per
    puts = [t for t in res.trades if t.action == "SELL_PUT"]
    assert puts[0].campaign_id == 1
    assert puts[-1].campaign_id == 2
    expired = [t for t in res.trades if t.action == "PUT_EXPIRED"]
    assert expired and expired[-1].campaign_id == 2

def test_warnings_field_exists_and_empty_on_clean_run():
    res = run_wheel(_chain(_SAGA), _cfg())
    assert res.warnings == []

def test_days_shares_uncovered_counts_naked_share_days():
    # assignment day has no call to sell; two more days with only unrelated put
    # rows -> shares sit naked on those two days (assignment day itself also
    # counts: no call was writable that day).
    naked = _SAGA[:2] + [
        ["2024-01-10","2024-01-17",7,470,"P",2.00,2.10,2.05,2.05,-0.30,0.1,465.0],
        ["2024-01-11","2024-01-18",7,470,"P",2.00,2.10,2.05,2.05,-0.30,0.1,465.0],
    ]
    res = run_wheel(_chain(naked), _cfg())
    assert res.days_shares_uncovered == 3   # Jan 9 (post-assignment), 10, 11
    covered = run_wheel(_chain(_SAGA), _cfg())
    # A9: the assignment day itself is now always an uncovered day -- the
    # deferral makes it structural (owner default 2026-08-02: it counts; it
    # is a real day the shares sat unrented).
    assert covered.days_shares_uncovered == 1
