"""Plain-path regression: with all defense flags off, the wheel engine's output
is byte-identical to the pre-repair engine (main @ a39407d). Digest excludes
fields the repair pass adds (campaign_id, warnings, days_shares_uncovered)."""
import hashlib
import os
import pandas as pd
import pytest
from src.engine_v2.options.wheel import WheelConfig, run_wheel

FIX = "fixtures/spy_wheel_cycle.parquet"

def _digest(res) -> str:
    h = hashlib.sha256()
    for ts, v in res.equity.items():
        h.update(f"{ts.isoformat()}:{v:.6f};".encode())
    for t in res.trades:
        h.update(f"{t.date}|{t.action}|{t.contract}|{t.contracts}"
                 f"|{t.price_per_contract:.6f}|{t.cash_after:.6f};".encode())
    h.update(f"{res.final_cash:.6f}|{res.final_shares}|{res.days_flat}".encode())
    return h.hexdigest()

GOLDEN = {
    "tp50":  "8d2d1b8a541e174cfe640b11c15eb4244cace3c402284f4ae8a3badd3db27f2a",
    "hold":  "86f68c8e6f9811b9b094baf4b6daebcac9537af285221db5faab45f42928f340",
}

@pytest.mark.skipif(not os.path.exists(FIX), reason="wheel cycle fixture not built")
@pytest.mark.parametrize("name,cfg", [
    ("tp50", WheelConfig(target_dte=40, put_delta=0.30, call_delta=0.30)),
    ("hold", WheelConfig(target_dte=40, put_delta=0.30, call_delta=0.30,
                         take_profit_pct=None)),
])
def test_plain_path_is_byte_identical(name, cfg):
    res = run_wheel(pd.read_parquet(FIX), cfg)
    assert _digest(res) == GOLDEN[name]
