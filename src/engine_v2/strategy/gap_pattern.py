"""Gap Pattern - Type A. Faithful implementation of the Oxford Capital Strategies
review (oxfordstrat.com/trading-strategies/gap-pattern/, rated B).

A full price gap beyond the prior bar's range, taken in the direction of a trend
filter, held for a fixed horizon with a pattern-reversal exit and an ATR stop.
Long AND short. Oxford tested it on 42 US futures 1980-2015; a "B" means a robust
region of positive performance across the parameter grid, not one lucky setting.

Mechanism: an opening gap that clears the prior range and confirms the prevailing
trend signals a liquidity/information shock that tends to continue over the next
days. Counterparty: whoever faded the gap into the trend. Not a proven edge --
Oxford's own verdict is "acceptable, not exceptional", and the grade was earned on
futures, so transfer to equity ETFs is an open question.

Daily-bar modelling limits: entry is booked at the signal bar's CLOSE (no intrabar
fills), and the ATR / pattern stops are evaluated against the daily close, so they
are coarser than the intrabar stop orders Oxford used.
"""
from __future__ import annotations

import pandas as pd

FORECAST_ON = 10.0
FLAT = 0.0


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> float:
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()],
                   axis=1).max(axis=1)
    return float(tr.tail(length).mean())


class _Pos:
    __slots__ = ("dir", "entry", "stop", "gap_level", "held")

    def __init__(self, direction, entry, stop, gap_level):
        self.dir = direction        # +1 long, -1 short
        self.entry = entry
        self.stop = stop            # ATR stop level
        self.gap_level = gap_level  # pre-gap boundary; pattern exit if price closes back through
        self.held = 1


class GapPatternTypeA:
    display_name = "Gap Pattern Type A · 6xATR · trend-filtered"

    mechanism = (
        "A full price gap beyond the prior bar's range, in the direction of the "
        "prevailing trend, marks a liquidity/information shock that tends to continue "
        "over the following days. We enter with the gap and hold for a fixed horizon, "
        "exiting if price closes back through the gap (the move was noise) or hits a "
        "6xATR stop. Counterparty: faders of the gap. Oxford rates the pattern B on "
        "futures -- acceptable, not exceptional; transfer to equity ETFs is untested."
    )

    parameter_grid = {"time_index": [10, 20, 30], "filter_lookback": [20, 50]}

    holding_period_cap = 60  # engine backstop; the plugin's own time_index exits first

    def __init__(self, time_index: int = 20, filter_lookback: int = 50,
                 atr_length: int = 20, atr_mult: float = 6.0):
        self.time_index = time_index
        self.filter_lookback = filter_lookback
        self.atr_length = atr_length
        self.atr_mult = atr_mult
        self._pos: dict[str, _Pos] = {}

    @property
    def _warmup(self) -> int:
        return max(self.filter_lookback, self.atr_length) + 2

    def _exit_signal(self, p: _Pos, close_now: float) -> bool:
        if p.held >= self.time_index:
            return True                                   # time exit
        if p.dir > 0:
            return close_now <= p.stop or close_now < p.gap_level
        return close_now >= p.stop or close_now > p.gap_level

    def _entry(self, high, low, close) -> _Pos | None:
        hi, lo, cl = high.iloc[-1], low.iloc[-1], close.iloc[-1]
        prev_hi, prev_lo = high.iloc[-2], low.iloc[-2]
        # channel over the `filter_lookback` bars ending at the prior bar
        upper = high.iloc[-(self.filter_lookback + 1):-1].max()
        lower = low.iloc[-(self.filter_lookback + 1):-1].min()
        atr = _atr(high, low, close, self.atr_length)

        if lo > prev_hi and hi > upper:                   # gap up + uptrend
            return _Pos(+1, cl, cl - self.atr_mult * atr, prev_hi)
        if hi < prev_lo and lo < lower:                   # gap down + downtrend
            return _Pos(-1, cl, cl + self.atr_mult * atr, prev_lo)
        return None

    def forecast(self, bars: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
        out: dict[str, float] = {}
        for tkr in bars.columns.get_level_values(0).unique():
            high = bars[tkr]["High"].dropna()
            low = bars[tkr]["Low"].dropna()
            close = bars[tkr]["Close"].dropna()
            if len(close) < self._warmup:
                out[tkr] = FLAT
                continue

            pos = self._pos.get(tkr)
            if pos is not None:
                if self._exit_signal(pos, float(close.iloc[-1])):
                    del self._pos[tkr]
                    out[tkr] = FLAT
                else:
                    pos.held += 1
                    out[tkr] = pos.dir * FORECAST_ON
                continue

            new = self._entry(high, low, close)
            if new is None:
                out[tkr] = FLAT
            else:
                self._pos[tkr] = new
                out[tkr] = new.dir * FORECAST_ON
        return pd.Series(out, dtype=float)
