"""Regime gates (macro phase 2, spec 2026-07-14): entry/roll denied in unpaid
decline, stop suppressed in stressed vol. Ticker state, strictly-prior-day,
zero new knobs. All default-off — plain behavior byte-identical."""
import pandas as pd
import pytest
from src.engine_v2.options.wheel import (WheelConfig, run_wheel,
                                         is_unpaid_decline, _state_before,
                                         GATE_STALENESS_DAYS)
from src.engine_v2.options.report import wheel_report, format_report

COLS = ["date","expiry","dte","strike","right","bid","ask","mid","close","delta","iv","underlying"]

def _chain(rows):
    ch = pd.DataFrame(rows, columns=COLS)
    ch["date"] = pd.to_datetime(ch["date"]); ch["expiry"] = pd.to_datetime(ch["expiry"])
    return ch

def _cfg(**kw):
    base = dict(starting_capital=50_000.0, put_delta=0.30, call_delta=0.30,
                take_profit_pct=None, commission_per_contract=0.0)
    base.update(kw); return WheelConfig(**base)

def _states(rows):
    """rows: list of (date, trend, vol) — minimal regime_series stand-in."""
    df = pd.DataFrame(rows, columns=["date","trend","vol"]).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df

PUT_DAY = [["2024-01-02","2024-01-09",7,470,"P",2.00,2.10,2.05,2.05,-0.30,0.1,472.0]]

# ---- constants + helper ----

def test_is_unpaid_decline_cells():
    assert is_unpaid_decline("downtrend", "calm")
    assert is_unpaid_decline("downtrend", "normal")
    assert not is_unpaid_decline("downtrend", "stressed")
    assert not is_unpaid_decline("uptrend", "calm")
    assert not is_unpaid_decline("chop", "normal")
    assert not is_unpaid_decline("unknown", "unknown")

def test_staleness_constant_matches_autopsy():
    # hand-synced by design (no shared module, to keep options/ -> regime/
    # import direction); this test IS the sync enforcement.
    from src.engine_v2.regime.autopsy import MAX_STALENESS_DAYS
    assert GATE_STALENESS_DAYS == MAX_STALENESS_DAYS

def test_state_before_is_strictly_prior():
    st = _states([("2024-01-01","uptrend","calm"), ("2024-01-02","downtrend","calm")])
    assert _state_before(st, pd.Timestamp("2024-01-02")) == ("uptrend", "calm")
    assert _state_before(st, pd.Timestamp("2024-01-03")) == ("downtrend", "calm")

def test_state_before_staleness_bound():
    st = _states([("2024-01-01","downtrend","calm")])
    ok = pd.Timestamp("2024-01-01") + pd.Timedelta(days=GATE_STALENESS_DAYS)
    assert _state_before(st, ok) == ("downtrend", "calm")
    assert _state_before(st, ok + pd.Timedelta(days=1)) == ("unknown", "unknown")

def test_state_before_empty_or_future_only():
    st = _states([("2024-06-01","downtrend","calm")])
    assert _state_before(st, pd.Timestamp("2024-01-02")) == ("unknown", "unknown")

# ---- validation ----

def test_gate_flag_without_states_raises():
    for flag in ("regime_entry_gate", "regime_roll_gate"):
        with pytest.raises(ValueError):
            run_wheel(_chain(PUT_DAY), _cfg(**{flag: True}))

def test_stop_gate_without_stop_raises():
    st = _states([("2024-01-01","uptrend","calm")])
    with pytest.raises(ValueError):
        run_wheel(_chain(PUT_DAY), _cfg(regime_stop_gate=True), regime_states=st)

# ---- entry gate ----

def test_entry_gate_blocks_new_campaign_in_unpaid_decline():
    st = _states([("2024-01-01","downtrend","calm")])
    res = run_wheel(_chain(PUT_DAY), _cfg(regime_entry_gate=True), regime_states=st)
    assert not [t for t in res.trades if t.action == "SELL_PUT"]
    assert res.days_entry_gated == 1
    assert (pd.Timestamp("2024-01-02"), "entry_gated", None) in res.gate_events

def test_entry_gate_allows_entry_outside_unpaid_decline():
    for trend, vol in [("uptrend","calm"), ("downtrend","stressed"), ("chop","normal")]:
        st = _states([("2024-01-01",trend,vol)])
        res = run_wheel(_chain(PUT_DAY), _cfg(regime_entry_gate=True), regime_states=st)
        assert [t for t in res.trades if t.action == "SELL_PUT"], (trend, vol)
        assert res.days_entry_gated == 0

def test_entry_gate_unknown_state_allows_and_warns():
    st = _states([("2023-06-01","downtrend","calm")])   # stale > 14d
    res = run_wheel(_chain(PUT_DAY), _cfg(regime_entry_gate=True), regime_states=st)
    assert [t for t in res.trades if t.action == "SELL_PUT"]
    assert (pd.Timestamp("2024-01-02"), "gate_state_unknown", "entry") in res.warnings

def test_entry_gate_does_not_gate_call_phase():
    # assigned shares; the call phase continues the OLD campaign, so the call
    # is written even though the state has turned unpaid-decline by then.
    # A9 (2026-08-02): the fixture's only call row sat on assignment day,
    # which merely happened to encode the same-session defect -- the intent
    # here (call phase is not entry-gated) is unrelated. A next-session row
    # carries the call now; the gate assertion is unchanged.
    rows = [
        ["2024-01-02","2024-01-09",7,470,"P",2.00,2.10,2.05,2.05,-0.30,0.1,472.0],
        ["2024-01-09","2024-01-09",0,470,"P",5.00,5.10,5.05,5.05,-0.99,0.1,465.0],
        ["2024-01-10","2024-01-16",6,475,"C",1.00,1.10,1.05,1.05,0.30,0.1,465.0],
    ]
    st = _states([("2024-01-01","uptrend","calm"), ("2024-01-08","downtrend","calm")])
    res = run_wheel(_chain(rows), _cfg(regime_entry_gate=True), regime_states=st)
    assert [t.action for t in res.trades] == ["SELL_PUT", "ASSIGNED", "SELL_CALL"]

def test_entry_gate_off_ignores_states_entirely():
    st = _states([("2024-01-01","downtrend","calm")])
    res = run_wheel(_chain(PUT_DAY), _cfg(), regime_states=st)
    assert [t for t in res.trades if t.action == "SELL_PUT"]
    assert res.days_entry_gated == 0 and res.gate_events == []

def test_entry_gate_not_counted_when_no_viable_entry():
    # unpaid decline BUT no contract near the target delta -> baseline could
    # not have entered either; the gate must not claim the block.
    rows = [["2024-01-02","2024-01-09",7,470,"P",2.00,2.10,2.05,2.05,-0.02,0.1,472.0]]
    # delta 0.02 vs target 0.30: select_contract still returns nearest-delta,
    # so starve by cash instead: strike 470 x 100 > tiny capital -> n = 0.
    st = _states([("2024-01-01","downtrend","calm")])
    res = run_wheel(_chain(PUT_DAY), _cfg(starting_capital=1_000.0,
                                          regime_entry_gate=True), regime_states=st)
    assert res.days_entry_gated == 0 and res.gate_events == []

# ---- roll gate ----
# tested put mid-life (spot 468 <= 470), same-strike out-roll available for
# credit; both potential legs settle OTM so the run ends clean.
ROLL_ROWS = [
    ["2024-01-02","2024-01-09",7,470,"P",2.00,2.10,2.05,2.05,-0.30,0.1,472.0],
    ["2024-01-03","2024-01-09",6,470,"P",3.00,3.10,3.05,3.05,-0.55,0.1,468.0],
    ["2024-01-03","2024-01-16",13,470,"P",5.00,5.10,5.05,5.05,-0.50,0.1,468.0],
    ["2024-01-09","2024-01-09",0,470,"P",0.05,0.10,0.07,0.07,-0.01,0.1,475.0],
    ["2024-01-16","2024-01-16",0,470,"P",0.05,0.10,0.07,0.07,-0.01,0.1,475.0],
]

def test_roll_gate_denies_roll_in_unpaid_decline():
    # entry gate is OFF here — only the roll is gated
    st = _states([("2024-01-01","downtrend","normal")])
    cfg = _cfg(roll_tested_puts=True, regime_roll_gate=True)
    res = run_wheel(_chain(ROLL_ROWS), cfg, regime_states=st)
    assert not [t for t in res.trades if t.action == "ROLL_CLOSE"]
    ev = [e for e in res.gate_events if e[1] == "roll_denied_by_gate"]
    assert ev and ev[0][0] == pd.Timestamp("2024-01-03")

def test_roll_gate_allows_roll_in_panic():
    st = _states([("2024-01-01","downtrend","stressed")])
    cfg = _cfg(roll_tested_puts=True, regime_roll_gate=True)
    res = run_wheel(_chain(ROLL_ROWS), cfg, regime_states=st)
    assert [t for t in res.trades if t.action == "ROLL_CLOSE"]
    assert not [e for e in res.gate_events if e[1] == "roll_denied_by_gate"]

def test_roll_gate_unknown_state_allows_and_warns():
    st = _states([("2023-06-01","downtrend","normal")])
    cfg = _cfg(roll_tested_puts=True, regime_roll_gate=True)
    res = run_wheel(_chain(ROLL_ROWS), cfg, regime_states=st)
    assert [t for t in res.trades if t.action == "ROLL_CLOSE"]
    assert (pd.Timestamp("2024-01-03"), "gate_state_unknown", "roll") in res.warnings

def test_roll_denial_not_logged_when_roll_could_not_execute():
    # unpaid decline, put tested, but NO destination expiry exists -> the
    # un-gated engine could not have rolled; no denial event may be logged.
    rows = [
        ["2024-01-02","2024-01-09",7,470,"P",2.00,2.10,2.05,2.05,-0.30,0.1,472.0],
        ["2024-01-03","2024-01-09",6,470,"P",3.00,3.10,3.05,3.05,-0.55,0.1,468.0],
        ["2024-01-09","2024-01-09",0,470,"P",0.05,0.10,0.07,0.07,-0.01,0.1,475.0],
    ]
    st = _states([("2024-01-01","downtrend","normal")])
    cfg = _cfg(roll_tested_puts=True, regime_roll_gate=True)
    res = run_wheel(_chain(rows), cfg, regime_states=st)
    assert not [e for e in res.gate_events if e[1] == "roll_denied_by_gate"]

def test_roll_gate_does_not_swallow_no_mark_warning():
    # gated state AND missing mark on the tested day: the data-gap warning must
    # still be recorded (the gate check sits after the mark check).
    rows = [
        ["2024-01-02","2024-01-09",7,470,"P",2.00,2.10,2.05,2.05,-0.30,0.1,472.0],
        # tested day exists in the chain via another strike, but 470P has no row
        ["2024-01-03","2024-01-09",6,460,"P",1.00,1.10,1.05,1.05,-0.20,0.1,468.0],
        ["2024-01-09","2024-01-09",0,470,"P",0.05,0.10,0.07,0.07,-0.01,0.1,475.0],
    ]
    st = _states([("2024-01-01","downtrend","normal")])
    cfg = _cfg(roll_tested_puts=True, regime_roll_gate=True)
    res = run_wheel(_chain(rows), cfg, regime_states=st)
    assert any(w[1] == "roll_check_no_mark" for w in res.warnings)
    assert not [e for e in res.gate_events if e[1] == "roll_denied_by_gate"]

def test_denied_roll_does_not_consume_stop_check():
    # same day: roll denied by gate AND stop threshold crossed -> stop still
    # fires (an executed roll consumes the stop; a DENIED roll must not).
    rows = [
        ["2024-01-02","2024-01-09",7,470,"P",2.00,2.10,2.05,2.05,-0.30,0.1,472.0],
        ["2024-01-03","2024-01-09",6,470,"P",6.90,7.00,6.95,6.95,-0.80,0.1,463.0],
        ["2024-01-03","2024-01-16",13,470,"P",9.00,9.10,9.05,9.05,-0.75,0.1,463.0],
        ["2024-01-09","2024-01-09",0,470,"P",0.05,0.10,0.07,0.07,-0.01,0.1,475.0],
        ["2024-01-16","2024-01-16",0,470,"P",0.05,0.10,0.07,0.07,-0.01,0.1,475.0],
    ]
    st = _states([("2024-01-01","downtrend","normal")])
    cfg = _cfg(roll_tested_puts=True, regime_roll_gate=True, put_stop_mult=3.0)
    res = run_wheel(_chain(rows), cfg, regime_states=st)
    assert not [t for t in res.trades if t.action == "ROLL_CLOSE"]
    assert [t for t in res.trades if t.action == "STOP_CLOSE"]

# ---- stop gate ----
STOP_ROWS = [
    ["2024-01-02","2024-01-09",7,470,"P",2.00,2.10,2.05,2.05,-0.30,0.1,472.0],
    ["2024-01-03","2024-01-09",6,470,"P",6.90,7.00,6.95,6.95,-0.80,0.1,463.0],
    ["2024-01-09","2024-01-09",0,470,"P",0.05,0.10,0.07,0.07,-0.01,0.1,475.0],
]

def test_stop_gate_suppresses_stop_in_stressed_vol():
    st = _states([("2024-01-01","downtrend","stressed")])
    cfg = _cfg(put_stop_mult=3.0, regime_stop_gate=True)
    res = run_wheel(_chain(STOP_ROWS), cfg, regime_states=st)
    assert not [t for t in res.trades if t.action == "STOP_CLOSE"]
    ev = [e for e in res.gate_events if e[1] == "stop_suppressed_by_gate"]
    assert ev and ev[0][0] == pd.Timestamp("2024-01-03")

def test_stop_gate_lets_stop_fire_when_not_stressed():
    for vol in ("calm", "normal"):
        st = _states([("2024-01-01","downtrend",vol)])
        cfg = _cfg(put_stop_mult=3.0, regime_stop_gate=True)
        res = run_wheel(_chain(STOP_ROWS), cfg, regime_states=st)
        assert [t for t in res.trades if t.action == "STOP_CLOSE"], vol

def test_stop_gate_unknown_state_allows_stop_and_warns():
    st = _states([("2023-06-01","uptrend","stressed")])
    cfg = _cfg(put_stop_mult=3.0, regime_stop_gate=True)
    res = run_wheel(_chain(STOP_ROWS), cfg, regime_states=st)
    assert [t for t in res.trades if t.action == "STOP_CLOSE"]
    assert (pd.Timestamp("2024-01-03"), "gate_state_unknown", "stop") in res.warnings

# ---- report ----

def test_report_gates_block_present_when_flag_on_even_if_never_fired():
    st = _states([("2024-01-01","uptrend","calm")])
    ch = _chain(PUT_DAY); cfg = _cfg(regime_entry_gate=True)
    rep = wheel_report(run_wheel(ch, cfg, regime_states=st), ch, cfg)
    assert rep.gates == {"days_entry_gated": 0, "n_rolls_denied": 0,
                         "n_stops_suppressed": 0, "n_state_unknown": 0}
    assert "Regime gates" in format_report(rep)

def test_report_gates_none_on_plain_run():
    ch = _chain(PUT_DAY); cfg = _cfg()
    rep = wheel_report(run_wheel(ch, cfg), ch, cfg)
    assert rep.gates is None and "Regime gates" not in format_report(rep)

def test_report_gates_counts_fired_events():
    st = _states([("2024-01-01","downtrend","calm")])
    ch = _chain(PUT_DAY); cfg = _cfg(regime_entry_gate=True)
    rep = wheel_report(run_wheel(ch, cfg, regime_states=st), ch, cfg)
    assert rep.gates["days_entry_gated"] == 1

# ---- invariance + no-look-ahead ----

def test_states_passed_flags_off_is_byte_identical():
    st = _states([("2024-01-01","downtrend","calm")])
    a = run_wheel(_chain(PUT_DAY), _cfg())
    b = run_wheel(_chain(PUT_DAY), _cfg(), regime_states=st)
    assert [(t.date, t.action, t.cash_after) for t in a.trades] == \
           [(t.date, t.action, t.cash_after) for t in b.trades]
    assert a.equity.equals(b.equity) and a.final_cash == b.final_cash

def test_no_lookahead_future_states_do_not_change_decisions():
    # identical states up to the chain window; extra FUTURE rows must not matter
    base = [("2024-01-01","downtrend","calm")]
    future = base + [("2024-06-01","uptrend","calm")]
    cfg = _cfg(regime_entry_gate=True)
    a = run_wheel(_chain(PUT_DAY), cfg, regime_states=_states(base))
    b = run_wheel(_chain(PUT_DAY), cfg, regime_states=_states(future))
    assert [(t.date, t.action) for t in a.trades] == [(t.date, t.action) for t in b.trades]
    assert a.days_entry_gated == b.days_entry_gated == 1

def test_same_day_state_is_not_used():
    # state flips to unpaid decline ON the entry day; strictly-prior rule must
    # use the previous (benign) day -> entry allowed.
    st = _states([("2024-01-01","uptrend","calm"), ("2024-01-02","downtrend","calm")])
    res = run_wheel(_chain(PUT_DAY), _cfg(regime_entry_gate=True), regime_states=st)
    assert [t for t in res.trades if t.action == "SELL_PUT"]

def test_stop_suppression_only_logged_when_stop_would_fire():
    # stressed state but the ask never reaches 3x credit -> no suppression event
    rows = [
        ["2024-01-02","2024-01-09",7,470,"P",2.00,2.10,2.05,2.05,-0.30,0.1,472.0],
        ["2024-01-03","2024-01-09",6,470,"P",3.00,3.10,3.05,3.05,-0.55,0.1,468.0],
        ["2024-01-09","2024-01-09",0,470,"P",0.05,0.10,0.07,0.07,-0.01,0.1,475.0],
    ]
    st = _states([("2024-01-01","downtrend","stressed")])
    cfg = _cfg(put_stop_mult=3.0, regime_stop_gate=True)
    res = run_wheel(_chain(rows), cfg, regime_states=st)
    assert not [e for e in res.gate_events if e[1] == "stop_suppressed_by_gate"]
