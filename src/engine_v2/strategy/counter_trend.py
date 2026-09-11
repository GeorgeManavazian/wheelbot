"""Counter-trend dip buy. Faithful replication of Clenow, *Trading Evolved*, ch.17.

Long-only. Buys volatility-standardised pullbacks inside bull markets. Clenow's own
caveats travel with it: the settings are "middle of the road", chosen for symmetry
with his trend model rather than for performance, and are explicitly not a production
model. His backtest ran a ~41-market futures portfolio; a single-ETF version will be
far lumpier and the -3 sigma entry fires rarely.
"""
from __future__ import annotations

import pandas as pd

FORECAST_ON = 10.0  # = sizing.carver.FORECAST_MEAN_MAG; Clenow's entry is binary
FLAT = 0.0


class CounterTrendDipBuy:
    display_name = "Counter-Trend Dip Buy · 3σ · 20d"

    mechanism = (
        "Trend-following funds run near-identical rules, so their stop-loss orders "
        "cluster at similar prices. When price reaches that band the stops trigger "
        "together and the cascade pushes price below fair value -- a liquidation "
        "stampede, not new information. Once the stop-out wave exhausts, price snaps "
        "back. We buy the standardised dip inside an established uptrend and hold for "
        "the snap-back. Counterparty: forced sellers with no discretion over timing. "
        "Clenow states this rationale is unverifiable; it is falsifiable, not proven."
    )

    parameter_grid = {"dip_buy": [-2.5, -3.0, -3.5]}

    holding_period_cap = 20  # Clenow's time exit; the loop enforces it as a backstop

    def __init__(self, dip_buy: float = -3.0, vola_window: int = 40,
                 high_window: int = 20, fast_ma: int = 40, slow_ma: int = 80):
        self.dip_buy = dip_buy
        self.vola_window = vola_window
        self.high_window = high_window
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self._bars_held: dict[str, int] = {}

    @property
    def _warmup(self) -> int:
        return max(self.slow_ma, self.vola_window, self.high_window) + 1

    def _signal(self, close: pd.Series) -> tuple[bool, float]:
        """(bull, pullback in standard deviations). Clenow standardises by the std of
        daily price CHANGES, not returns -- the same figure his position sizer uses."""
        bull = (close.ewm(span=self.fast_ma).mean().iloc[-1]
                > close.ewm(span=self.slow_ma).mean().iloc[-1])
        std = close.diff().tail(self.vola_window).std(ddof=1)
        if not std > 0:
            return bull, 0.0
        pullback = (close.iloc[-1] - close.tail(self.high_window).max()) / std
        return bull, float(pullback)

    def forecast(self, bars: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
        out: dict[str, float] = {}
        for tkr in bars.columns.get_level_values(0).unique():
            close = bars[tkr]["Close"].dropna()
            if len(close) < self._warmup:
                out[tkr] = FLAT
                continue

            bull, pullback = self._signal(close)
            held = self._bars_held.get(tkr, 0)

            if held > 0:
                # exit on trend flip, or once the time cap is reached
                if not bull or held >= self.holding_period_cap:
                    self._bars_held[tkr] = 0
                    out[tkr] = FLAT
                else:
                    self._bars_held[tkr] = held + 1
                    out[tkr] = FORECAST_ON
            elif bull and pullback < self.dip_buy:
                self._bars_held[tkr] = 1
                out[tkr] = FORECAST_ON
            else:
                out[tkr] = FLAT

        return pd.Series(out, dtype=float)
