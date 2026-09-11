"""Oxford Gap Pattern - Type A (oxfordstrat.com, rated B).

Rules (daily bars, long AND short):
  LONG setup : Low[i] > High[i-1]         (today's range gaps fully above yesterday's)
  SHORT setup: High[i] < Low[i-1]
  TREND FILTER: long only if High[i] > highest-high over the prior `filter_lookback`
               bars; short only if Low[i] < lowest-low over that window.
  EXITS (first to fire): time exit after `time_index` bars; pattern exit if price
               closes back through the gap; stop at 6*ATR(atr_length) from entry.

Engine models entry at the signal bar's CLOSE (daily bars, no intrabar fills) and
approximates the intrabar stops with the daily close -- documented limitations.
"""
import numpy as np
import pandas as pd
import pytest

from src.engine_v2.strategy.protocol import validate_plugin, FORECAST_MIN, FORECAST_MAX
from src.engine_v2.strategy.gap_pattern import GapPatternTypeA

TICKER = "SPY"


def _bars(o, h, l, c):
    n = len(c)
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    cols = pd.MultiIndex.from_product([[TICKER], ["Open", "High", "Low", "Close", "Volume"]])
    df = pd.DataFrame(index=idx, columns=cols, dtype=float)
    df[(TICKER, "Open")] = o
    df[(TICKER, "High")] = h
    df[(TICKER, "Low")] = l
    df[(TICKER, "Close")] = c
    df[(TICKER, "Volume")] = 1e8
    return df


def _uptrend_then(gap_high, gap_low, gap_close, n=10, base=100.0, step=1.0):
    """n rising bars whose ranges OVERLAP (half-range 0.6 > half-step 0.5, so no
    internal gaps), then one appended bar with the given H/L/C."""
    highs = [base + step * i + 0.6 for i in range(n)]
    lows = [base + step * i - 0.6 for i in range(n)]
    closes = [base + step * i for i in range(n)]
    opens = closes[:]
    highs.append(gap_high); lows.append(gap_low); closes.append(gap_close); opens.append(gap_close)
    return opens, highs, lows, closes


def _fc(strat, bars, i):
    return float(strat.forecast(bars.iloc[: i + 1], bars.index[i])[TICKER])


def test_plugin_satisfies_contract():
    validate_plugin(GapPatternTypeA)
    assert GapPatternTypeA.mechanism.strip()
    assert GapPatternTypeA.holding_period_cap >= 1


def test_long_entry_on_gap_up_in_uptrend():
    # last normal bar (index 9): high ~ 100+9+0.4 = 109.4. Gap bar: low 111 > 109.4,
    # and high 113 > channel max (highest high of bars 5..9 ~ 109.4).
    o, h, l, c = _uptrend_then(gap_high=113.0, gap_low=111.0, gap_close=112.0)
    bars = _bars(o, h, l, c)
    strat = GapPatternTypeA(filter_lookback=5, atr_length=5, time_index=3)
    f = _fc(strat, bars, len(c) - 1)
    assert f == pytest.approx(10.0), "did not go long on a filtered gap-up"


def test_no_entry_when_gap_up_fails_the_trend_filter():
    # A real gap up (low 110.0 > prior high 109.6), but a prior spike lifts the
    # 5-bar channel high above the gap's high -> trend filter rejects.
    o, h, l, c = _uptrend_then(gap_high=110.5, gap_low=110.0, gap_close=110.2)
    h[6] = 120.0  # spike inside the filter window, above the gap's high
    bars = _bars(o, h, l, c)
    strat = GapPatternTypeA(filter_lookback=5, atr_length=5, time_index=3)
    assert _fc(strat, bars, len(c) - 1) == 0.0


def test_no_entry_without_a_gap():
    o, h, l, c = _uptrend_then(gap_high=110.0, gap_low=108.5, gap_close=109.5)
    # low 108.5 < yesterday high 109.4 -> ranges overlap, no gap
    bars = _bars(o, h, l, c)
    strat = GapPatternTypeA(filter_lookback=5, atr_length=5, time_index=3)
    assert _fc(strat, bars, len(c) - 1) == 0.0


def test_short_entry_on_gap_down_in_downtrend():
    highs = [120 - i + 0.6 for i in range(10)]
    lows = [120 - i - 0.6 for i in range(10)]
    closes = [120 - i for i in range(10)]
    opens = closes[:]
    # gap down: high 108 < yesterday low (120-9-0.6=110.4); low below channel low
    highs.append(108.0); lows.append(106.0); closes.append(107.0); opens.append(107.0)
    bars = _bars(opens, highs, lows, closes)
    strat = GapPatternTypeA(filter_lookback=5, atr_length=5, time_index=3)
    assert _fc(strat, bars, len(closes) - 1) == pytest.approx(-10.0)


def test_forecast_stays_in_bounds():
    o, h, l, c = _uptrend_then(gap_high=113.0, gap_low=111.0, gap_close=112.0)
    bars = _bars(o, h, l, c)
    strat = GapPatternTypeA(filter_lookback=5, atr_length=5, time_index=3)
    for i in range(len(c)):
        f = _fc(strat, bars, i)
        assert FORECAST_MIN <= f <= FORECAST_MAX


def test_time_exit_after_time_index_bars():
    o, h, l, c = _uptrend_then(gap_high=113.0, gap_low=111.0, gap_close=112.0)
    # extend with calm bars above the gap so no stop/pattern exit fires first
    for k in range(6):
        o.append(112.0); h.append(112.5); l.append(111.5); c.append(112.0)
    bars = _bars(o, h, l, c)
    strat = GapPatternTypeA(filter_lookback=5, atr_length=5, time_index=3)
    fcs = [_fc(strat, bars, i) for i in range(len(c))]
    entry = fcs.index(10.0)
    held = [f for f in fcs[entry:] if f == 10.0]
    assert len(held) == 3, f"held {len(held)} bars, time_index=3"
    assert fcs[entry + 3] == 0.0, "did not exit on the 4th bar"


def test_pattern_exit_when_price_closes_back_through_the_gap():
    o, h, l, c = _uptrend_then(gap_high=113.0, gap_low=111.0, gap_close=112.0)
    # next bar collapses back below yesterday's pre-gap high (~109.4) -> pattern exit
    o.append(112.0); h.append(112.0); l.append(108.0); c.append(108.5)
    bars = _bars(o, h, l, c)
    strat = GapPatternTypeA(filter_lookback=5, atr_length=5, time_index=20)
    fcs = [_fc(strat, bars, i) for i in range(len(c))]
    assert fcs[-1] == 0.0, "did not pattern-exit when the gap closed"
