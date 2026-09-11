import os
import pandas as pd
import pytest
from src.engine_v2.options.wheel import WheelConfig, run_wheel
from src.engine_v2.options.report import buy_hold_curve, spy_curve, wheel_stats, wheel_report

FIX = "fixtures/spy_wheel_cycle.parquet"
pytestmark = pytest.mark.skipif(not os.path.exists(FIX), reason="wheel cycle fixture not built")

def _run():
    ch = pd.read_parquet(FIX)
    cfg = WheelConfig(target_dte=40, put_delta=0.30, call_delta=0.30)
    return run_wheel(ch, cfg), ch, cfg

def test_buy_hold_curve_shape():
    _, ch, _ = _run()
    bh = buy_hold_curve(ch, 100_000)
    assert bh.iloc[0] == pytest.approx(100_000)          # starts at capital
    assert (bh > 0).all()
    assert bh.name == "buy_hold"

def test_spy_curve_none_when_file_missing():
    _, ch, _ = _run()
    idx = ch.groupby("date")["underlying"].first().index
    assert spy_curve(100_000, idx, path="nonexistent.parquet") is None

def test_wheel_stats_on_real_fixture():
    res, _, cfg = _run()
    s = wheel_stats(res, cfg)
    assert s["n_puts_sold"] == 2 and s["n_take_profits"] == 2
    assert s["n_assignments"] == 0 and s["n_called_away"] == 0
    # 4 option legs x 2 contracts x $0.65
    assert s["commission_paid"] == pytest.approx(4 * 2 * 0.65)
    assert s["premium_collected"] > s["premium_paid_to_close"] > 0
    assert s["assignment_rate"] == 0.0

def test_wheel_report_metrics_finite_and_benchmarked():
    res, ch, cfg = _run()
    rep = wheel_report(res, ch, cfg, spy_path="nonexistent.parquet")
    for k in ("total_return", "cagr", "sharpe", "max_drawdown"):
        assert k in rep.metrics and pd.notna(rep.metrics[k])
    assert set(rep.benchmark_underlying) >= {"cagr", "sharpe", "max_drawdown"}
    assert 2024 in rep.yearly_return.index
    assert rep.periods_per_year == pytest.approx(252, abs=8)   # daily wheel equity
    # recent_start (2021) predates the fixture's 2024 span, so it clamps to the
    # fixture start and recent == full history — the collapse flag must say so.
    assert set(rep.benchmark_underlying_recent.keys()) == set(rep.benchmark_underlying.keys())
    assert rep.recent_is_full is True

def test_benchmark_underlying_is_this_chain():
    res, ch, cfg = _run()
    rep = wheel_report(res, ch, cfg, spy_path="nonexistent.parquet")
    assert rep.benchmark_spy is None
    assert rep.benchmark_underlying["total_return"] == pytest.approx(
        float(ch.groupby("date")["underlying"].first().iloc[-1]
              / ch.groupby("date")["underlying"].first().iloc[0] - 1), rel=1e-6)

def test_spy_benchmark_loads_real_spy(tmp_path):
    res, ch, cfg = _run()
    spy = ch.copy(); spy["underlying"] = spy["underlying"] * 2  # distinct series
    p = tmp_path / "spy.parquet"; spy.to_parquet(p)
    rep = wheel_report(res, ch, cfg, spy_path=str(p))
    assert rep.benchmark_spy is not None

def test_stats_have_dte_distribution_and_flat_days():
    res, ch, cfg = _run()
    rep = wheel_report(res, ch, cfg, spy_path="nonexistent.parquet")
    assert rep.stats["n_days_flat"] == res.days_flat
    assert 0.0 <= rep.stats["pct_days_flat"] <= 1.0
    assert rep.stats["realized_dte"].sum() == rep.stats["n_puts_sold"] + rep.stats["n_calls_sold"]

# ---- repair pass: campaign reporting + roll counterfactuals ----

from src.engine_v2.options.report import campaign_table, roll_counterfactuals

_COLS = ["date","expiry","dte","strike","right","bid","ask","mid","close","delta","iv","underlying"]

def _rows_chain(rows):
    ch = pd.DataFrame(rows, columns=_COLS)
    ch["date"] = pd.to_datetime(ch["date"]); ch["expiry"] = pd.to_datetime(ch["expiry"])
    return ch

def _rows_cfg(**kw):
    base = dict(starting_capital=50_000.0, put_delta=0.30, call_delta=0.30,
                take_profit_pct=None, commission_per_contract=0.0)
    base.update(kw); return WheelConfig(**base)

_ROLL_SAGA = [
    ["2024-01-02","2024-01-09",7,470,"P",2.00,2.10,2.05,2.05,-0.30,0.1,472.0],
    ["2024-01-04","2024-01-09",5,470,"P",3.00,3.10,3.05,3.05,-0.55,0.1,468.0],
    # same strike, out in time (amendment 2026-07-13b destination, ext 7)
    ["2024-01-04","2024-01-16",12,470,"P",3.20,3.30,3.25,3.25,-0.52,0.1,468.0],
    # expiry of the ORIGINAL leg (for the counterfactual): spot 466 -> intrinsic 4.00
    ["2024-01-09","2024-01-09",0,470,"P",4.00,4.10,4.05,4.05,-0.99,0.1,466.0],
    # rolled leg expires worthless (spot back above the strike)
    ["2024-01-16","2024-01-16",0,470,"P",0.05,0.10,0.075,0.075,-0.01,0.1,471.0],
]

def _roll_result():
    ch = _rows_chain(_ROLL_SAGA)
    cfg = _rows_cfg(roll_tested_puts=True)
    return run_wheel(ch, cfg), ch, cfg

def test_campaign_table_one_campaign_with_roll():
    res, ch, cfg = _roll_result()
    tbl = campaign_table(res, cfg)
    assert len(tbl) == 1
    row = tbl.iloc[0]
    assert row["campaign_id"] == 1 and row["n_rolls"] == 1
    # cash flow: +2.00 (entry) -3.10 (roll close) +3.20 (roll open) = +2.10/share
    assert row["pnl"] == pytest.approx(210.0)
    assert not row["open_at_end"]

def test_roll_counterfactual_short_leg():
    res, ch, cfg = _roll_result()
    cf = roll_counterfactuals(res, ch, cfg)
    assert len(cf) == 1
    r = cf.iloc[0]
    # actual: sold 2.00, closed 3.10 -> -1.10/share = -110
    assert r["actual_close_pnl"] == pytest.approx(-110.0)
    # held to expiry: 2.00 credit - 4.00 intrinsic -> -200
    assert r["held_to_expiry_pnl"] == pytest.approx(-200.0)
    assert r["roll_advantage"] == pytest.approx(90.0)

def test_wheel_stats_defense_keys():
    res, ch, cfg = _roll_result()
    s = wheel_stats(res, cfg)
    assert s["n_rolls"] == 1 and s["n_stops"] == 0 and s["n_campaigns"] == 1
    # net premium: +2.00 entry -3.10 roll close +3.20 roll open = 2.10/share
    assert s["net_premium"] == pytest.approx(210.0)

def test_defense_block_in_report_and_format():
    from src.engine_v2.options.report import wheel_report, format_report
    res, ch, cfg = _roll_result()
    rep = wheel_report(res, ch, cfg)
    assert rep.defense is not None
    assert rep.defense["n_campaigns"] == 1
    txt = format_report(rep)
    assert "Defense stats" in txt

def test_plain_run_has_no_defense_block():
    res, ch, cfg = _run()
    rep = wheel_report(res, ch, cfg)
    assert rep.defense is None
    from src.engine_v2.options.report import format_report
    assert "Defense stats" not in format_report(rep)

# ---- TP count, defense gate, interest-free campaign pnl ----

def test_n_take_profits_excludes_rolls_and_stops():
    res, ch, cfg = _roll_result()
    s = wheel_stats(res, cfg)
    assert s["n_take_profits"] == 0        # only a roll happened, no TP
    assert s["n_rolls"] == 1

def test_defense_block_present_when_enabled_but_never_fired():
    from src.engine_v2.options.report import wheel_report, format_report
    ch = _rows_chain([
        ["2024-01-02","2024-01-09",7,470,"P",2.00,2.10,2.05,2.05,-0.30,0.1,472.0],
        ["2024-01-09","2024-01-09",0,470,"P",0.02,0.05,0.035,0.035,-0.01,0.1,475.0],
    ])
    cfg = _rows_cfg(put_stop_mult=3.0)
    res = run_wheel(ch, cfg)
    rep = wheel_report(res, ch, cfg)
    assert rep.defense is not None and rep.defense["n_stops"] == 0
    assert "Defense stats" in format_report(rep)

def test_campaign_pnl_excludes_cash_yield_interest():
    from src.engine_v2.options.report import campaign_table
    # 5% cash yield, one week position: interest accrues on ~50k but campaign
    # pnl must reflect only the option economics (+2.00 credit kept = +200).
    ch = _rows_chain([
        ["2024-01-02","2024-01-09",7,470,"P",2.00,2.10,2.05,2.05,-0.30,0.1,472.0],
        ["2024-01-09","2024-01-09",0,470,"P",0.02,0.05,0.035,0.035,-0.01,0.1,475.0],
    ])
    cfg = _rows_cfg(cash_yield=0.05)
    res = run_wheel(ch, cfg)
    ct = campaign_table(res, cfg)
    assert ct.iloc[0]["pnl"] == pytest.approx(200.0)

def test_counterfactual_settles_at_last_close_before_gap_expiry():
    from src.engine_v2.options.report import roll_counterfactuals
    # original leg expiry Jan-09 has no chain rows; settle uses the LAST close
    # at/before expiry (Jan-04, 466 -> intrinsic 4.00), matching the engine's
    # late-resolution rule -> held pnl -200. Nothing skipped.
    saga = [
        ["2024-01-02","2024-01-09",7,470,"P",2.00,2.10,2.05,2.05,-0.30,0.1,472.0],
        ["2024-01-04","2024-01-09",5,470,"P",3.00,3.10,3.05,3.05,-0.55,0.1,466.0],
        ["2024-01-04","2024-01-16",12,470,"P",3.20,3.30,3.25,3.25,-0.52,0.1,466.0],
        ["2024-01-10","2024-01-16",6,470,"P",3.90,4.10,4.00,4.00,-0.80,0.1,469.0],
        ["2024-01-16","2024-01-16",0,470,"P",0.05,0.10,0.075,0.075,-0.01,0.1,471.0],
    ]
    ch = _rows_chain(saga)
    cfg = _rows_cfg(roll_tested_puts=True)
    res = run_wheel(ch, cfg)
    cf = roll_counterfactuals(res, ch, cfg)
    assert len(cf) == 1
    assert cf.iloc[0]["held_to_expiry_pnl"] == pytest.approx(-200.0)
    assert cf.attrs["n_skipped"] == 0
