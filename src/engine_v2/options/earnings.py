"""Earnings calendar + the blackout predicate (spec 2026-08-03-earnings-blackout).

The gate's claim: a scheduled earnings print inside a short put's life is a
discontinuity 0.30-delta premium was not priced to compensate for, and the
chop scanner's vol-percentile ranking actively selects for names that have one
coming.

THREE states per ticker, and keeping them apart is the whole point. Both "this
ETF has no earnings" and "the lookup for this name failed" present as an empty
calendar; collapsing them is how a gate silently stops existing (the
zombie_check failure class, noted 2026-07-29). The mapping encodes them:

    dates(tk) -> [d1, d2, ...]   known, has prints
    dates(tk) -> []              known, has none (fund, or genuinely none)
    dates(tk) -> None            UNKNOWN -- the gate is blind to this name

Unknown does not block: "unknown state always allows" is the standing
convention from the phase-1/2 regime work. It must therefore be COUNTED by
callers, never shrugged at.
"""
from __future__ import annotations

import os

import pandas as pd

CALENDAR_DIR = os.path.join("data", "earnings")

# The after-market-close correction. A print released after the close on day X
# moves the stock on X+1, so the window extends one day past expiry. This is
# NOT a tuning knob and is deliberately not exposed on Config: parsing BMO/AMC
# out of the vendor's clock field is a per-vendor convention with its own
# failure modes, and widening by a day covers the AMC case unconditionally.
BLACKOUT_BUFFER_DAYS = 1


class EarningsCalendar:
    """Ticker -> sorted earnings dates. Absent ticker means unknown, not empty."""

    def __init__(self, by_ticker: dict | None = None):
        self._by = {t: sorted(pd.Timestamp(d).normalize() for d in ds)
                    for t, ds in (by_ticker or {}).items()}

    @classmethod
    def load(cls, directory: str = CALENDAR_DIR) -> "EarningsCalendar":
        """Read the puller's artifacts. A ticker enters the map only if its
        status is ok/none -- a `failed` row is deliberately left OUT so it reads
        as unknown rather than as a name with no prints."""
        cal_path = os.path.join(directory, "calendar.parquet")
        st_path = os.path.join(directory, "status.parquet")
        if not os.path.exists(st_path):
            return cls({})

        st = pd.read_parquet(st_path)
        known = set(st.loc[st["status"].isin(["ok", "none"]), "ticker"])
        by: dict = {t: [] for t in known}

        if os.path.exists(cal_path):
            cal = pd.read_parquet(cal_path)
            cal = cal[cal["ticker"].isin(known)]
            for tk, g in cal.groupby("ticker"):
                by[tk] = list(pd.to_datetime(g["earnings_date"]))
        return cls(by)

    def dates(self, ticker: str):
        """Sorted dates, [] if known-none, None if unknown. See module docstring."""
        return self._by.get(ticker)

    def known(self, ticker: str) -> bool:
        return ticker in self._by

    def __len__(self) -> int:
        return len(self._by)


def in_blackout(dates, obs_date, expiry, buffer_days: int = BLACKOUT_BUFFER_DAYS) -> bool:
    """True if a print falls in [obs_date, expiry + buffer], inclusive.

    `dates` None (unknown) or empty (known-none) both return False -- allow.
    The window's right edge is the ACTUAL chosen expiry, not target_dte:
    realized DTE varies inside the derived band, so target_dte would blur the
    edge by a few days in both directions."""
    if not dates:
        return False
    lo = pd.Timestamp(obs_date).normalize()
    hi = pd.Timestamp(expiry).normalize() + pd.Timedelta(days=buffer_days)
    return any(lo <= pd.Timestamp(d).normalize() <= hi for d in dates)
