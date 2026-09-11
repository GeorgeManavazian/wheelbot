"""Wheel-campaign autopsy by regime: the campaign ledger grouped by the market
state (and the traded ticker's state) each campaign opened into."""
from __future__ import annotations
import pandas as pd
from ..options.report import campaign_table
from .state import regime_series

MAX_STALENESS_DAYS = 14   # calendar days; older last-known state -> "unknown"

def _tag(dates, states: pd.DataFrame, col: str) -> list:
    """State as of STRICTLY BEFORE the open date (a trade placed on day d
    cannot know day d's close — same rule as the engine), and never more than
    MAX_STALENESS_DAYS stale (a closes series that ends before the campaign
    opened must say 'unknown', not silently reuse a months-old state)."""
    return [row[col] for row in _tag_rows(dates, states)]

_UNKNOWN = {"trend": "unknown", "vol": "unknown"}

def _tag_rows(dates, states: pd.DataFrame) -> list:
    """One as-of lookup per date (searchsorted, strictly-prior + staleness),
    shared by every column — not one O(n) slice per column per campaign."""
    idx = states.index
    out = []
    for d in dates:
        d = pd.Timestamp(d)
        pos = idx.searchsorted(d) - 1          # last state STRICTLY before d
        if pos < 0 or (d - idx[pos]).days > MAX_STALENESS_DAYS:
            out.append(_UNKNOWN)
        else:
            out.append(states.iloc[pos])
    return out

def campaign_regimes(result, cfg, market_closes: pd.Series,
                     ticker_closes: pd.Series) -> pd.DataFrame:
    ct = campaign_table(result, cfg)
    mkt, tkr = regime_series(market_closes), regime_series(ticker_closes)
    ct = ct.copy()
    for prefix, states in (("market", mkt), ("ticker", tkr)):
        rows = _tag_rows(ct["opened"], states)
        ct[f"{prefix}_trend"] = [r["trend"] for r in rows]
        ct[f"{prefix}_vol"] = [r["vol"] for r in rows]
    return ct

def autopsy_table(campaigns: pd.DataFrame, by: str = "market") -> pd.DataFrame:
    closed = campaigns[~campaigns["open_at_end"]]
    rows = []
    for (trend, vol), g in closed.groupby([f"{by}_trend", f"{by}_vol"]):
        rows.append(dict(trend=trend, vol=vol, n_campaigns=len(g),
                         win_rate=float((g["pnl"] > 0).mean()),
                         total_pnl=float(g["pnl"].sum()),
                         mean_pnl=float(g["pnl"].mean()),
                         n_rolls=int(g["n_rolls"].sum())))
    out = pd.DataFrame(rows, columns=["trend","vol","n_campaigns","win_rate",
                                      "total_pnl","mean_pnl","n_rolls"])
    out.attrs["n_open_excluded"] = int(campaigns["open_at_end"].sum())
    return out
