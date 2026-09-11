"""Backtest core: per-bar simulation loop + execution + sizing, shared by the
workbench (`backtest/simple.py`) and by direct diagnostic tests. The CPCV/gate
honesty pipeline that used to sit on top of this (`run_backtest`) is parked in
`archive/` (2026-07-10) -- the workbench calls `_simulate`/`position_history`
directly and never runs it."""
from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd
from ..sizing.carver import size_position, DEFAULT_TARGET_RISK
from ..execution.sim import Order, fill_order, bid_ask
from ..execution.shorting import HTBRegistry, apply_short_guard, daily_borrow_fee

STARTING_EQUITY = 100_000.0


# Default half-spread per side, in bps. 1bp suits large liquid ETFs (SPY/TLT/GLD:
# penny-wide quotes on $100+ prices). Override per ticker for thinner names.
DEFAULT_HALF_SPREAD_BPS = 1.0


@dataclass
class BacktestConfig:
    target_risk: float = DEFAULT_TARGET_RISK
    n_folds: int = 10
    cpcv_k: int = 2
    starting_equity: float = STARTING_EQUITY
    htb: HTBRegistry = field(default_factory=HTBRegistry)
    spread_bps_per_side: float = DEFAULT_HALF_SPREAD_BPS
    spread_bps_by_ticker: dict = field(default_factory=dict)
    # Annualized stock-borrow rate for short positions. 50bps ~ general-collateral
    # liquid ETF. Override per ticker for hard-to-borrow names.
    borrow_bps_annual: float = 50.0
    borrow_bps_by_ticker: dict = field(default_factory=dict)

    def half_spread_bps(self, ticker: str) -> float:
        return self.spread_bps_by_ticker.get(ticker, self.spread_bps_per_side)

    def borrow_bps(self, ticker: str) -> float:
        return self.borrow_bps_by_ticker.get(ticker, self.borrow_bps_annual)


def _fill_at_close_via_sim(ticker: str, side: str, qty: float,
                           close: float, adv: float, half_spread_bps: float,
                           date: pd.Timestamp, htb: HTBRegistry):
    """Route through execution/sim, crossing a real bid-ask spread around the close."""
    o = Order(ticker=ticker, side=side, kind="market", qty=qty)
    o = apply_short_guard(o, date, htb)
    if o is None:
        return None
    bid, ask = bid_ask(close, half_spread_bps)
    return fill_order(o, bid=bid, ask=ask, adv=adv)


def _instrument_sigma(bars, tkr, asof, periods_per_year: float = 252.0) -> float:
    past = bars[tkr]["Close"].loc[:asof].pct_change().dropna().tail(60)
    return float(past.std() * (periods_per_year ** 0.5)) if len(past) >= 20 else 0.20


def _simulate(strategy_cls, params, bars, test_index, cfg: BacktestConfig,
              periods_per_year: float = 252.0):
    """Run one CPCV fold, carrying position state across bars.

    Convention: decide and trade at the close of bar t, then hold to the close of
    bar t+1. So P&L on bar t is qty_held * (close_t - close_{t-1}), booked BEFORE
    the bar's rebalance. Costs are charged only on the change in position.

    Yields one row per bar: equity after mark-to-market and trading, the resulting
    position, and shares traded.
    """
    strat = strategy_cls(**params)
    cap = getattr(strategy_cls, "holding_period_cap", None) or 10 ** 9
    tickers = set(bars.columns.get_level_values(0))
    adv = 1e9  # fixture proxy; real ingest supplies per-ticker ADV

    equity = cfg.starting_equity
    qty: dict[str, float] = {}
    bars_held: dict[str, int] = {}
    prev_close: dict[str, float] = {}
    rows = []

    for asof in test_index:
        # 1) mark existing positions to market against the previous close, and
        #    charge one day of borrow on any short still open at this bar's close
        for tkr, q in qty.items():
            if q == 0.0 or tkr not in prev_close:
                continue
            close_now = float(bars[tkr]["Close"].loc[asof])
            equity += q * (close_now - prev_close[tkr])
            if q < 0.0:
                equity -= daily_borrow_fee(abs(q) * close_now, cfg.borrow_bps(tkr))

        # 2) rebalance at this bar's close
        visible = bars.loc[:asof]  # no look-ahead: inclusive of asof only
        fc = strat.forecast(visible, asof)
        traded = 0.0
        for tkr, f in fc.items():
            if tkr not in tickers:
                continue
            close = float(bars[tkr]["Close"].loc[asof])
            held = qty.get(tkr, 0.0)

            if held != 0.0 and bars_held.get(tkr, 0) >= cap:
                target_notional = 0.0  # time exit
            else:
                target_notional = size_position(
                    float(f), equity,
                    _instrument_sigma(bars, tkr, asof, periods_per_year),
                    target_risk=cfg.target_risk,
                )
            target_qty = target_notional / close if close > 0 else 0.0
            delta = target_qty - held

            if abs(delta) * close > 1e-9:
                side = "buy" if delta > 0 else "sell"
                fill = _fill_at_close_via_sim(tkr, side, abs(delta), close, adv,
                                              cfg.half_spread_bps(tkr), asof, cfg.htb)
                if fill is not None:
                    # fill.price is offset from the mid (close) by spread + slippage
                    equity -= abs(delta) * abs(fill.price - close)
                    traded += abs(delta)
                    new_qty = held + delta
                    qty[tkr] = new_qty
                    bars_held[tkr] = 0 if new_qty == 0.0 else bars_held.get(tkr, 0)

            if qty.get(tkr, 0.0) != 0.0:
                bars_held[tkr] = bars_held.get(tkr, 0) + 1

        for tkr in tickers:
            prev_close[tkr] = float(bars[tkr]["Close"].loc[asof])

        rows.append({"asof": asof, "equity": equity, "traded": traded,
                     "qty": sum(qty.values())})

    return pd.DataFrame(rows).set_index("asof")


def _fold_trial_returns(strategy_cls, params, bars, test_index, cfg: BacktestConfig):
    """Per-date return series for one CPCV fold."""
    df = _simulate(strategy_cls, params, bars, test_index, cfg)
    return df["equity"].pct_change().fillna(0.0).rename("ret")


def position_history(strategy_cls, params, bars, test_index, cfg: BacktestConfig,
                     periods_per_year: float = 252.0):
    """Per-bar equity, net position and shares traded. For tests and diagnostics."""
    return _simulate(strategy_cls, params, bars, test_index, cfg, periods_per_year)
