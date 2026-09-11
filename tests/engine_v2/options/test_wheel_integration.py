import os
import pandas as pd
import pytest
from src.engine_v2.options.wheel import WheelConfig, run_wheel

FIX = "fixtures/spy_wheel_cycle.parquet"


@pytest.mark.skipif(not os.path.exists(FIX), reason="wheel cycle fixture not built")
def test_wheel_runs_on_real_cycle():
    ch = pd.read_parquet(FIX)
    # target_dte=40: the fixture's first date only shows the 42-DTE expiry, and
    # band(40) = (38, 43) accepts it -> the bot trades from the first bar.
    res = run_wheel(ch, WheelConfig(target_dte=40, put_delta=0.30, call_delta=0.30))
    assert len(res.equity) > 5
    assert (res.equity > 0).all()               # never blows up
    assert any(t.action == "SELL_PUT" for t in res.trades)
    # equity starts near capital (first bar carries only a small short liability)
    assert abs(res.equity.iloc[0] - 100_000) < 5_000
    # golden: pinned to the committed fixture so a logic-changing refactor breaks it
    assert len(res.trades) == 4
    assert [t.action for t in res.trades] == ["SELL_PUT", "CLOSE_PUT"] * 2
    assert res.final_cash == pytest.approx(100_838.80, abs=0.01)
    assert res.final_shares == 0
