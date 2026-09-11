import os
import pandas as pd
import pytest
from src.engine_v2.options.wheel import WheelConfig, run_wheel
from src.engine_v2.options.report import wheel_report, format_report

FIX = "fixtures/spy_wheel_cycle.parquet"
pytestmark = pytest.mark.skipif(not os.path.exists(FIX), reason="wheel cycle fixture not built")

def _report(spy_path="nonexistent.parquet", ticker="SPY"):
    ch = pd.read_parquet(FIX)
    cfg = WheelConfig(target_dte=40, put_delta=0.30, call_delta=0.30, ticker=ticker)
    return wheel_report(run_wheel(ch, cfg), ch, cfg, spy_path=spy_path)

def test_format_report_has_sections():
    txt = format_report(_report())
    assert isinstance(txt, str)
    for token in ("CAGR", "Sharpe", "Max drawdown", "buy-hold", "Year", "assignment"):
        assert token.lower() in txt.lower()
    assert "2024" in txt          # year-by-year row present

def test_format_report_titles_benchmark_with_ticker():
    txt = format_report(_report(ticker="GDX"))
    assert "Benchmark — buy-hold GDX" in txt
    # no SPY file -> no SPY benchmark block
    assert "Benchmark — buy-hold SPY" not in txt

def test_format_report_spy_block_when_present(tmp_path):
    ch = pd.read_parquet(FIX)
    spy = ch.copy(); spy["underlying"] = spy["underlying"] * 2
    p = tmp_path / "spy.parquet"; spy.to_parquet(p)
    txt = format_report(_report(spy_path=str(p), ticker="GDX"))
    assert "Benchmark — buy-hold GDX" in txt
    assert "Benchmark — buy-hold SPY" in txt

def test_format_report_flat_days_and_realized_dte():
    rep = _report()
    txt = format_report(rep)
    assert f"flat {rep.stats['pct_days_flat']:.0%} of days" in txt
    assert "realized DTE" in txt
    dtes = rep.stats["realized_dte"]
    assert f"{dtes.index.min()}..{dtes.index.max()}" in txt
