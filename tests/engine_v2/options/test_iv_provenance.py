"""Provenance on the IV series: one scale per ticker, enforced.

Owner ruling 2026-08-07 ([[2026-08-07 - The IV provenance rule, the solver is the
scale]]). IV rank is a percentile of a ticker against its OWN trailing window, so a
bias that is constant within that ticker cancels completely -- measured, two
different IV models on identical quotes pick the same top-ranked name on 95.5% of
63,016 ranked ticker-days (spearman 0.9988).

What does NOT cancel is a scale change INSIDE the window. It shifts every
post-change observation relative to its own past and pins ranks toward 0 or 1, while
every log line still looks healthy. That is the same shape as the zombie gate
(earnings.py), and the same shape as `chain_store` serving
yesterday's file as today's.

So the source is a permanent property of a TICKER, not of a batch or a load, and a
source change must REBUILD that ticker's history rather than append to it. This file
pins that as a loud refusal rather than a convention someone has to remember.
"""
import pandas as pd
import pytest

from src.engine_v2.options.iv_rank import IVHistory


def test_appending_an_observation_from_another_source_is_refused():
    h = IVHistory({"GDX": pd.Series({pd.Timestamp("2026-07-01"): 0.31})},
                  sources={"GDX": "thetadata/bs-v1"})
    with pytest.raises(ValueError, match="source"):
        h.append("GDX", pd.Timestamp("2026-07-02"), 0.33, source="schwab/bs-v1")


def _series(n, start="2025-01-01", value=0.20):
    idx = pd.bdate_range(start, periods=n)
    return pd.Series([value] * n, index=idx)


def test_an_appended_observation_participates_in_the_rank():
    """Storage is not the point -- the observation has to reach the percentile."""
    base = _series(200, value=0.20)
    h = IVHistory({"GDX": base}, sources={"GDX": "thetadata/bs-v1"})
    nxt = base.index[-1] + pd.offsets.BDay(1)

    h.append("GDX", nxt, 0.99, source="thetadata/bs-v1")

    # 0.99 is above all 200 priors, so it sits at the top of its own window.
    assert h.rank("GDX", nxt) == pytest.approx(1.0)


def test_the_first_observation_stamps_an_unstamped_series():
    h = IVHistory({"GDX": _series(3)})
    assert h.source("GDX") is None

    h.append("GDX", pd.Timestamp("2026-07-02"), 0.33, source="schwab/bs-v1")

    assert h.source("GDX") == "schwab/bs-v1"
    # ...and having taken a stamp, it now refuses a different one.
    with pytest.raises(ValueError, match="source"):
        h.append("GDX", pd.Timestamp("2026-07-03"), 0.34, source="thetadata/bs-v1")
