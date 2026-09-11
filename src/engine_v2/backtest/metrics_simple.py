"""Frequency-aware, gate-free diagnostics computed from a per-bar equity curve.
Annualization derives from bar spacing so the same code is correct for daily
bars now and intraday bars later. No pass/fail — diagnostics only."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..data.regime import tag_regime, REGIME_COLS

def infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return 252.0
    deltas = np.diff(index.values).astype("timedelta64[s]").astype(float)
    med_s = float(np.median(deltas))
    if med_s <= 0:
        return 252.0
    year_s = 365.25 * 24 * 3600
    if med_s >= 20 * 3600:          # daily-or-coarser bars: count trading days
        return 252.0
    trading_seconds_per_year = 252 * 6.5 * 3600   # 6.5h session
    return trading_seconds_per_year / med_s

def cagr(equity: pd.Series, periods_per_year: float) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return float("nan")
    total = equity.iloc[-1] / equity.iloc[0]
    years = len(equity) / periods_per_year
    if years <= 0 or total <= 0:
        return float("nan")
    return float(total ** (1 / years) - 1)

def sharpe(returns: pd.Series, periods_per_year: float) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    sd = float(r.std())
    if sd < 1e-10 or np.isnan(sd):
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods_per_year))

def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0

def max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return float("nan")
    return float(drawdown_series(equity).min())

def yearly_returns(equity: pd.Series) -> pd.Series:
    if len(equity) == 0:
        return pd.Series(dtype=float)
    by_year = equity.groupby(equity.index.year)
    first, last = by_year.first(), by_year.last()
    return (last / first - 1.0).rename("return")

def yearly_sharpe(returns: pd.Series, periods_per_year: float) -> pd.Series:
    """Annualized Sharpe per calendar year — the decay curve the standing
    methodology requires. One value per year present in the index."""
    r = returns.dropna()
    if len(r) == 0:
        return pd.Series(dtype=float)
    out = {y: sharpe(grp, periods_per_year) for y, grp in r.groupby(r.index.year)}
    return pd.Series(out, name="sharpe")

def buy_hold_equity(bars: pd.DataFrame, weights: dict, starting_equity: float) -> pd.Series:
    eq = None
    for tkr, w in weights.items():
        close = bars[tkr]["Close"]
        norm = close / close.iloc[0]
        leg = w * starting_equity * norm
        eq = leg if eq is None else eq + leg
    return eq.rename("benchmark")

def regime_breakdown(returns: pd.Series, bars: pd.DataFrame,
                     periods_per_year: float) -> pd.DataFrame:
    labels = tag_regime(bars).reindex(returns.index).ffill()
    rows = []
    for col in REGIME_COLS:
        for val, grp in returns.groupby(labels[col]):
            rows.append({"dimension": col, "regime": val,
                         "sharpe": sharpe(grp, periods_per_year),
                         "mean_ret": float(grp.mean()), "bars": int(len(grp))})
    return pd.DataFrame(rows)

def monthly_returns(equity: pd.Series) -> pd.DataFrame:
    """Month-by-month fractional returns as a year x month grid (columns 1..12).

    The first month's return is measured off the opening equity, so a backtest
    that starts mid-month still reports that month. NaN where no data.

    `equity` is assumed to be chronologically sorted: the first-month return
    is measured positionally off `equity.iloc[0]`, so an unsorted input would
    yield a wrong first-month return.
    """
    if equity.empty:
        return pd.DataFrame()
    month_end = equity.resample("ME").last()
    # A calendar month with zero observations resamples to NaN. Compute prev
    # from the last non-NaN month-end so a gap doesn't poison the return of
    # the month that follows it -- the gap month itself is still NaN because
    # its own month_end is NaN, but a real month after the gap must not be.
    prev = month_end.ffill().shift(1)
    prev.iloc[0] = equity.iloc[0]  # first month: measure off opening equity
    rets = month_end / prev - 1.0
    grid = pd.DataFrame({
        "year": rets.index.year,
        "month": rets.index.month,
        "ret": rets.to_numpy(),
    })
    out = grid.pivot(index="year", columns="month", values="ret")
    return out.reindex(columns=range(1, 13))
