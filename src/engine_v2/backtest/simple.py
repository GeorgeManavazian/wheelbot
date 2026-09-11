"""Clean backtest seam: one straight pass over the full date range, no CPCV,
no gate, no verdict. Returns rich diagnostics. This is the workbench entrypoint."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from ..strategy.protocol import validate_plugin
from .orchestrator import position_history, BacktestConfig
from . import metrics_simple as m

@dataclass
class Result:
    equity: pd.Series
    returns: pd.Series
    trades: int
    cagr: float
    sharpe: float
    max_drawdown: float
    yearly: pd.Series
    yearly_sharpe: pd.Series
    recent: dict
    regime: pd.DataFrame
    benchmarks: dict
    periods_per_year: float

def run_simple(strategy_cls, bars: pd.DataFrame,
               config: BacktestConfig | None = None,
               params: dict | None = None,
               periods_per_year: float | None = None,
               recent_start: str = "2021-07-01",
               benchmark_bars: pd.DataFrame | None = None) -> Result:
    validate_plugin(strategy_cls)
    cfg = config or BacktestConfig()
    ppy = periods_per_year or m.infer_periods_per_year(bars.index)

    hist = position_history(strategy_cls, params or {}, bars, bars.index, cfg, ppy)
    equity = hist["equity"]
    returns = equity.pct_change().fillna(0.0)

    # Standing methodology: recent window is the HEADLINE; keep full history too.
    eq_recent = equity[equity.index >= recent_start]
    ret_recent = returns[returns.index >= recent_start]
    recent = {
        "start": recent_start,
        "cagr": m.cagr(eq_recent, ppy),
        "sharpe": m.sharpe(ret_recent, ppy),
        "max_drawdown": m.max_drawdown(eq_recent),
    }

    bench = benchmark_bars if benchmark_bars is not None else bars
    bench_tickers = set(bench.columns.get_level_values(0))
    benchmarks = {}
    if "SPY" in bench_tickers:
        benchmarks["SPY"] = m.buy_hold_equity(bench, {"SPY": 1.0}, cfg.starting_equity)
    if "SPY" in bench_tickers and "TLT" in bench_tickers:
        benchmarks["60_40"] = m.buy_hold_equity(
            bench, {"SPY": 0.6, "TLT": 0.4}, cfg.starting_equity)

    # regime tagging needs SPY as the market benchmark; prefer the strategy's own
    # bars (original behavior) and fall back to the benchmark override, but never
    # crash if SPY is absent from both (e.g. a non-SPY universe with no override).
    strategy_tickers = set(bars.columns.get_level_values(0))
    if "SPY" in strategy_tickers:
        regime = m.regime_breakdown(returns, bars, ppy)
    elif "SPY" in bench_tickers:
        regime = m.regime_breakdown(returns, bench, ppy)
    else:
        regime = pd.DataFrame(columns=["dimension", "regime", "sharpe", "mean_ret", "bars"])

    return Result(
        equity=equity,
        returns=returns,
        trades=int((hist["traded"] > 0).sum()),
        cagr=m.cagr(equity, ppy),
        sharpe=m.sharpe(returns, ppy),
        max_drawdown=m.max_drawdown(equity),
        yearly=m.yearly_returns(equity),
        yearly_sharpe=m.yearly_sharpe(returns, ppy),
        recent=recent,
        regime=regime,
        benchmarks=benchmarks,
        periods_per_year=ppy,
    )
