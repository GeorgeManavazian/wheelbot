"""Backtest loop + trial expansion. Enforces no-look-ahead via slice([:asof])
and produces append-only per-date log."""
from __future__ import annotations
import hashlib
from itertools import product
import pandas as pd
from ..sizing.carver import size_position

STARTING_EQUITY = 100_000.0

def _trial_id(cls_name: str, params: dict, fold_id: int) -> str:
    key = f"{cls_name}|" + ",".join(f"{k}={v}" for k, v in sorted(params.items())) + f"|fold={fold_id}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]

def expand_trials(strategy_cls, parameter_grid: dict, fold_ids: list[int]) -> list[dict]:
    if not parameter_grid:
        combos = [{}]
    else:
        keys = list(parameter_grid.keys())
        combos = [dict(zip(keys, vals)) for vals in product(*[parameter_grid[k] for k in keys])]
    trials = []
    for params in combos:
        for fold_id in fold_ids:
            trials.append({
                "trial_id": _trial_id(strategy_cls.__name__, params, fold_id),
                "strategy": strategy_cls.__name__,
                "params": params,
                "fold_id": fold_id,
            })
    return trials

def run_trial(strategy_cls, params: dict, bars: pd.DataFrame,
              fold_slice: slice, equity: float = STARTING_EQUITY) -> pd.DataFrame:
    strat = strategy_cls(**params)
    view = bars.loc[fold_slice]
    rows = []
    for asof in view.index:
        visible = bars.loc[:asof]  # no look-ahead: inclusive of asof only
        fc = strat.forecast(visible, asof)
        # single-ticker naive fill at close (execution sim wired in Task 16 integ)
        for tkr, f in fc.items():
            price = float(bars[tkr]["Close"].loc[asof])
            past = bars[tkr]["Close"].loc[:asof].pct_change().dropna().tail(60)
            sigma = float(past.std() * (252 ** 0.5)) if len(past) >= 20 else 0.20
            notional = size_position(float(f), equity, sigma)
            rows.append({
                "asof": asof,
                "forecast": float(f),
                "notional": notional,
                "fill_price": price,
                "equity": equity,
            })
    return pd.DataFrame(rows)
