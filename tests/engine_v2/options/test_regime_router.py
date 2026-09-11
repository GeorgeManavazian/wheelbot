"""Regime router (spec 2026-07-14-regime-router-design): uptrend->hold shares,
chop->wheel+basis, downtrend+stressed->wheel, downtrend+quiet->cash,
unknown->wheel. Approach A borders. Zero knobs, EOD, strictly-prior-day state."""
import pandas as pd
import pytest
from src.engine_v2.options.wheel import WheelConfig, run_wheel
from src.engine_v2.options.regime_router import run_regime_router

COLS = ["date","expiry","dte","strike","right","bid","ask","mid","close","delta","iv","underlying"]

def _chain(rows):
    ch = pd.DataFrame(rows, columns=COLS)
    ch["date"] = pd.to_datetime(ch["date"]); ch["expiry"] = pd.to_datetime(ch["expiry"])
    return ch

def _cfg(**kw):
    base = dict(starting_capital=50_000.0, put_delta=0.30, call_delta=0.30,
                take_profit_pct=None, commission_per_contract=0.0,
                call_min_strike="basis")
    base.update(kw); return WheelConfig(**base)

def _states(rows):
    """rows: (date, trend, vol, vol_pctile)"""
    df = pd.DataFrame(rows, columns=["date","trend","vol","vol_pctile"]).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df

PUT_DAY = [["2024-01-02","2024-01-09",7,470,"P",2.00,2.10,2.05,2.05,-0.30,0.1,472.0]]
UP = [("2024-01-01","uptrend","calm",0.2)]
CHOP = [("2024-01-01","chop","normal",0.5)]

# ---- anchor: WHEEL posture == solo wheel+basis, byte for byte ----

# ---- validation ----

def test_missing_states_raises():
    with pytest.raises(ValueError):
        run_regime_router(_chain(PUT_DAY), _cfg(), None)

def test_solo_only_flags_raise():
    st = _states(CHOP)
    for kw in ({"roll_tested_puts": True}, {"put_stop_mult": 3.0},
               {"regime_entry_gate": True}, {"liquidate_assignment": True,
                                             "call_min_strike": None}):
        with pytest.raises(ValueError):
            run_regime_router(_chain(PUT_DAY), _cfg(**kw), st)

def test_unknown_state_routes_to_wheel_and_warns():
    st = _states([("2023-06-01","chop","normal",0.5)])   # stale >14d
    res = run_regime_router(_chain(PUT_DAY), _cfg(), st)
    assert [t for t in res.trades if t.action == "SELL_PUT"]
    assert any(w[1] == "route_state_unknown" for w in res.warnings)

# ---- TREND + CASH cells ----

def test_uptrend_buys_100_lots_at_eod_spot():
    res = run_regime_router(_chain(PUT_DAY), _cfg(), _states(UP))
    buys = [t for t in res.trades if t.action == "BUY_SHARES"]
    assert buys and buys[0].price_per_contract == 472.0
    lots = int(50_000 // (472.0 * 100))
    assert buys[0].contracts == lots
    assert res.final_cash == pytest.approx(50_000 - lots * 100 * 472.0)
    assert res.days_in_posture["TREND"] == 1

def test_uptrend_holds_no_calls_written():
    rows = PUT_DAY + [["2024-01-03","2024-01-10",7,470,"C",2.0,2.1,2.05,2.05,0.30,0.1,474.0]]
    res = run_regime_router(_chain(rows), _cfg(), _states(UP))
    assert not [t for t in res.trades if t.action == "SELL_CALL"]
    assert not [t for t in res.trades if t.action == "SELL_SHARES"]

def test_cash_cell_no_entries_days_counted():
    st = _states([("2024-01-01","downtrend","normal",0.5)])
    res = run_regime_router(_chain(PUT_DAY), _cfg(), st)
    assert not [t for t in res.trades if t.action in ("SELL_PUT","BUY_SHARES")]
    assert res.days_in_posture["CASH"] == 1

def test_panic_cell_wheels():
    st = _states([("2024-01-01","downtrend","stressed",0.9)])
    res = run_regime_router(_chain(PUT_DAY), _cfg(), st)
    assert [t for t in res.trades if t.action == "SELL_PUT"]

# ---- border transitions (approach A) ----

def test_uptrend_to_downtrend_sells_next_close():
    rows = [PUT_DAY[0],
            ["2024-01-03","2024-01-10",7,470,"P",2.0,2.1,2.05,2.05,-0.30,0.1,468.0]]
    st = _states([("2024-01-01","uptrend","calm",0.2),
                  ("2024-01-02","downtrend","normal",0.5)])
    res = run_regime_router(_chain(rows), _cfg(), st)
    assert res.trades[0].action == "BUY_SHARES"
    sells = [t for t in res.trades if t.action == "SELL_SHARES"]
    assert sells and pd.Timestamp(sells[0].date) == pd.Timestamp("2024-01-03")
    assert sells[0].price_per_contract == 468.0
    assert res.whipsaw_pairs == 1   # sold 1 day after buying

def test_uptrend_to_chop_keeps_shares_starts_covered_calls_at_purchase_basis():
    rows = [PUT_DAY[0],
            ["2024-01-03","2024-01-10",7,465,"C",3.0,3.1,3.05,3.05,0.30,0.1,470.0],
            ["2024-01-03","2024-01-10",7,475,"C",0.5,0.6,0.55,0.55,0.05,0.1,470.0]]
    st = _states([("2024-01-01","uptrend","calm",0.2),
                  ("2024-01-02","chop","normal",0.5)])
    res = run_regime_router(_chain(rows), _cfg(), st)
    assert not [t for t in res.trades if t.action == "SELL_SHARES"]
    calls = [t for t in res.trades if t.action == "SELL_CALL"]
    assert calls and calls[0].contract.strike >= 472.0   # basis = purchase px

def test_chop_wheel_campaign_survives_flip_to_expiry():
    rows = [PUT_DAY[0],
            ["2024-01-03","2024-01-09",6,470,"P",2.0,2.1,2.05,2.05,-0.30,0.1,472.0],
            ["2024-01-09","2024-01-09",0,470,"P",0.05,0.10,0.07,0.07,-0.01,0.1,475.0]]
    st = _states([("2024-01-01","chop","normal",0.5),
                  ("2024-01-02","uptrend","calm",0.2)])
    res = run_regime_router(_chain(rows), _cfg(), st)
    acts = [t.action for t in res.trades]
    assert "SELL_PUT" in acts and "PUT_EXPIRED" in acts

def test_panic_assignment_then_uptrend_flip_keeps_shares_no_calls():
    rows = [
        ["2024-01-02","2024-01-09",7,470,"P",2.0,2.1,2.05,2.05,-0.30,0.1,472.0],
        ["2024-01-09","2024-01-09",0,470,"P",5.0,5.1,5.05,5.05,-0.99,0.1,465.0],
        ["2024-01-10","2024-01-17",7,475,"C",1.0,1.1,1.05,1.05,0.30,0.1,466.0]]
    st = _states([("2024-01-01","downtrend","stressed",0.9),
                  ("2024-01-09","uptrend","calm",0.2)])
    res = run_regime_router(_chain(rows), _cfg(), st)
    assert [t for t in res.trades if t.action == "ASSIGNED"]
    assert not [t for t in res.trades if t.action in ("SELL_CALL","SELL_SHARES")]
    assert res.final_shares == 100

def test_same_day_state_flip_ignored():
    st = _states([("2024-01-01","uptrend","calm",0.2),
                  ("2024-01-02","downtrend","normal",0.5)])
    res = run_regime_router(_chain(PUT_DAY), _cfg(), st)
    assert [t for t in res.trades if t.action == "BUY_SHARES"]

def test_future_state_rows_do_not_change_decisions():
    a = run_regime_router(_chain(PUT_DAY), _cfg(), _states(UP))
    b = run_regime_router(_chain(PUT_DAY), _cfg(),
                          _states(UP + [("2024-06-01","downtrend","normal",0.5)]))
    assert [(t.date, t.action) for t in a.trades] == [(t.date, t.action) for t in b.trades]

def test_unknown_days_logged_even_while_holding():
    # state goes stale while a short is open: every routed unknown day must be
    # logged, not just entry days (review finding 2026-07-14).
    rows = [
        ["2024-01-02","2024-01-09",7,470,"P",2.00,2.10,2.05,2.05,-0.30,0.1,472.0],
        ["2024-01-03","2024-01-09",6,470,"P",2.00,2.10,2.05,2.05,-0.30,0.1,472.0],
        ["2024-01-09","2024-01-09",0,470,"P",0.05,0.10,0.07,0.07,-0.01,0.1,475.0],
    ]
    st = _states([("2023-06-01","chop","normal",0.5)])   # stale for every day
    res = run_regime_router(_chain(rows), _cfg(), st)
    unknown_days = {pd.Timestamp(w[0]) for w in res.warnings
                    if w[1] == "route_state_unknown"}
    assert len(unknown_days) == 3   # all three chain days routed blind

def test_uncovered_days_not_counted_in_trend_cell():
    # wheel-assigned shares held through an uptrend: covered calls are
    # FORBIDDEN there, so those days must not count as failed coverage.
    rows = [
        ["2024-01-02","2024-01-09",7,470,"P",2.0,2.1,2.05,2.05,-0.30,0.1,472.0],
        ["2024-01-09","2024-01-09",0,470,"P",5.0,5.1,5.05,5.05,-0.99,0.1,465.0],
        ["2024-01-10","2024-01-17",7,475,"C",1.0,1.1,1.05,1.05,0.30,0.1,466.0],
        ["2024-01-11","2024-01-17",6,475,"C",1.0,1.1,1.05,1.05,0.30,0.1,467.0],
    ]
    st = _states([("2024-01-01","downtrend","stressed",0.9),
                  ("2024-01-09","uptrend","calm",0.2)])
    res = run_regime_router(_chain(rows), _cfg(), st)
    assert [t for t in res.trades if t.action == "ASSIGNED"]
    # only the assignment day itself (still a WHEEL cell via prior-day state)
    # counts as failed coverage; the two uptrend days are exempt.
    assert res.days_shares_uncovered == 1

def test_trend_sale_can_redeploy_same_day_into_panic():
    # uptrend -> downtrend+stressed: shares sold at close AND a put may be
    # sold the same close (spec sequencing rule).
    rows = [PUT_DAY[0],
            ["2024-01-03","2024-01-10",7,460,"P",3.0,3.1,3.05,3.05,-0.30,0.1,468.0]]
    st = _states([("2024-01-01","uptrend","calm",0.2),
                  ("2024-01-02","downtrend","stressed",0.9)])
    res = run_regime_router(_chain(rows), _cfg(), st)
    d3 = [t.action for t in res.trades if pd.Timestamp(t.date) == pd.Timestamp("2024-01-03")]
    assert "SELL_SHARES" in d3 and "SELL_PUT" in d3

# ---- conviction trim (spec 2026-07-15) ----

def _states_trim(rows):  # alias for readability
    return _states(rows)

def test_stressed_uptrend_entry_is_half_size():
    st = _states([("2024-01-01","uptrend","stressed",0.9)])
    full = run_regime_router(_chain(PUT_DAY), _cfg(starting_capital=500_000.0), st)
    trim = run_regime_router(_chain(PUT_DAY),
                             _cfg(starting_capital=500_000.0, conviction_trim=True), st)
    f = [t for t in full.trades if t.action=="BUY_SHARES"][0]  # 10 lots
    g = [t for t in trim.trades if t.action=="BUY_SHARES"][0]  # 5 lots
    assert g.contracts == f.contracts // 2 and g.contracts > 0
    assert trim.n_trimmed_entries == 1 and trim.days_half_size >= 1

def test_calm_uptrend_entry_is_full_size():
    st = _states([("2024-01-01","uptrend","calm",0.2)])
    full = run_regime_router(_chain(PUT_DAY), _cfg(), st)
    trim = run_regime_router(_chain(PUT_DAY), _cfg(conviction_trim=True), st)
    assert [t for t in trim.trades if t.action=="BUY_SHARES"][0].contracts == \
           [t for t in full.trades if t.action=="BUY_SHARES"][0].contracts
    assert trim.n_trimmed_entries == 0 and trim.days_half_size == 0

def test_trim_only_touches_trend_not_wheel():
    st = _states([("2024-01-01","downtrend","stressed",0.9)])
    a = run_regime_router(_chain(PUT_DAY), _cfg(), st)
    b = run_regime_router(_chain(PUT_DAY), _cfg(conviction_trim=True), st)
    assert [t.contracts for t in a.trades if t.action=="SELL_PUT"] == \
           [t.contracts for t in b.trades if t.action=="SELL_PUT"]
    assert b.n_trimmed_entries == 0

def test_single_lot_trim_buys_nothing():
    # spot so high that full = 1 lot; stressed -> 1//2 = 0 bought
    rows = [["2024-01-02","2024-01-09",7,470,"P",2.0,2.1,2.05,2.05,-0.30,0.1,49000.0]]
    st = _states([("2024-01-01","uptrend","stressed",0.9)])
    res = run_regime_router(_chain(rows), _cfg(starting_capital=50_000.0,
                            conviction_trim=True), st)
    assert not [t for t in res.trades if t.action=="BUY_SHARES"]

def test_no_lookahead_trim():
    st = _states([("2024-01-01","uptrend","calm",0.2),
                  ("2024-01-02","uptrend","stressed",0.9)])
    res = run_regime_router(_chain(PUT_DAY), _cfg(conviction_trim=True), st)
    assert res.n_trimmed_entries == 0   # prior day (calm) rules the entry

def test_trimmed_hold_converts_to_wheel_at_half():
    rows = [PUT_DAY[0],
            ["2024-01-03","2024-01-10",7,475,"C",1.0,1.1,1.05,1.05,0.30,0.1,470.0]]
    st = _states([("2024-01-01","uptrend","stressed",0.9),
                  ("2024-01-02","chop","normal",0.5)])
    res = run_regime_router(_chain(rows),
                            _cfg(starting_capital=500_000.0, conviction_trim=True), st)
    buy = [t for t in res.trades if t.action=="BUY_SHARES"][0]
    calls = [t for t in res.trades if t.action=="SELL_CALL"]
    # the wheel now rents exactly the half-size share count
    assert calls and calls[0].contracts == buy.contracts

def test_conviction_trim_rejected_on_solo_wheel():
    # router-only mechanic must not be silently ignored by the solo engine
    ch = _chain(PUT_DAY)
    with pytest.raises(ValueError):
        run_wheel(ch, _cfg(conviction_trim=True))

def test_trim_fraction_constant_is_live():
    # editing TRIM_FRACTION must actually change sizing (guards the dead-const bug)
    import src.engine_v2.options.regime_router as rr
    st = _states([("2024-01-01","uptrend","stressed",0.9)])
    orig = rr.TRIM_FRACTION
    try:
        rr.TRIM_FRACTION = 0.25
        q = run_regime_router(_chain(PUT_DAY),
                              _cfg(starting_capital=500_000.0, conviction_trim=True), st)
        full = run_regime_router(_chain(PUT_DAY), _cfg(starting_capital=500_000.0), st)
        f = [t for t in full.trades if t.action=="BUY_SHARES"][0].contracts
        g = [t for t in q.trades if t.action=="BUY_SHARES"][0].contracts
        assert g == int(f * 0.25)
    finally:
        rr.TRIM_FRACTION = orig
