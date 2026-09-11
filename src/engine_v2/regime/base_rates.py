"""Historical base rates per regime cell: what the underlying did over the
next `horizon` days from each state. Descriptive context for a human."""
from __future__ import annotations
import pandas as pd
from .state import regime_series

THIN_N = 30
CAVEAT = ("Full-history descriptive statistics — context for a human, not a "
          "signal. Any backtested decision using these numbers must recompute "
          "them walk-forward (only data before each decision date). Overlapping "
          "forward windows: N counts days, not independent samples.")

def base_rate_table(closes: pd.Series, horizon: int = 21) -> pd.DataFrame:
    states = regime_series(closes)
    c = closes.dropna().astype(float)
    fwd = c.shift(-horizon) / c - 1
    df = states.join(fwd.rename("fwd")).dropna(subset=["fwd"])
    rows = []
    for (trend, vol), g in df.groupby(["trend", "vol"]):
        rows.append(dict(trend=trend, vol=vol, n=len(g),
                         win_rate=float((g["fwd"] > 0).mean()),
                         median_fwd=float(g["fwd"].median()),
                         mean_fwd=float(g["fwd"].mean()),
                         p5_fwd=float(g["fwd"].quantile(0.05)),
                         thin=len(g) < THIN_N))
    out = pd.DataFrame(rows, columns=["trend","vol","n","win_rate","median_fwd",
                                      "mean_fwd","p5_fwd","thin"])
    out.attrs["caveat"] = CAVEAT
    return out
