import numpy as np
import pandas as pd
from src.engine_v2.regime.base_rates import base_rate_table, THIN_N, CAVEAT

def _steady_up():
    # 700 days of smooth 0.05%/day growth with tiny alternating noise:
    # after warmup everything is uptrend, forward returns known exactly.
    n = 700
    noise = np.array([1e-4 * (-1) ** i for i in range(n)]).cumsum()
    px = 100 * np.exp(np.arange(n) * 5e-4 + noise)
    return pd.Series(px, index=pd.bdate_range("2020-01-01", periods=n))

def test_uptrend_cells_match_known_growth():
    tbl = base_rate_table(_steady_up(), horizon=21)
    up = tbl[tbl["trend"] == "uptrend"]
    assert len(up) >= 1 and set(tbl["trend"]) == {"uptrend"}
    assert (up["win_rate"] == 1.0).all()
    expected = np.exp(21 * 5e-4) - 1
    assert (up["median_fwd"] - expected).abs().max() < 0.005
    assert up["n"].sum() > 300          # vol pctile splits the trend into cells

def test_thin_cells_flagged():
    tbl = base_rate_table(_steady_up(), horizon=21)
    assert ((tbl["n"] < THIN_N) == tbl["thin"]).all()

def test_caveat_attached():
    tbl = base_rate_table(_steady_up())
    assert "walk-forward" in tbl.attrs["caveat"]
