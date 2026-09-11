"""The variance risk premium: what the option pays against what the stock does.

THE CLAIM. A short put earns `IV - future realized vol`. That spread is the
entire edge; the strike, the tenor and the roll are delivery mechanisms for it.
This bot has never read it. `rank_by="vol_pctile"` sorts on realized vol -- the
RISK TAKEN, never the price paid for taking it -- and the `iv_rank` arm
measured on 2026-08-04 sorted on the LEVEL of implied vol, which is confounded
with the level of risk. The spread is neither.

THE MEASUREMENT THAT MOTIVATES IT. Over 10,528 gate-passing ticker-days on the
live window, IV/RV by realized-vol decile:

    decile      1      3      5      7      9     10
    realized  0.130  0.187  0.233  0.283  0.384  0.533
    IV / RV   1.620  1.332  1.237  1.136  1.088  0.998

Monotone, every step. The calmest decile is paid 62% more than the volatility
it delivers; the jumpiest is paid exactly what it delivers, i.e. nothing. The
live ranker takes the top of that table by realized vol -- median IV/RV 1.098
against a pool median of 1.226. It is systematically the buyer of the
worst-priced risk in its own candidate set.

WHAT THIS IS NOT. It is not a forecast. IV is forward-looking and the realized
vol it is divided by is TRAILING, so part of the monotone slope above is honest
vol mean-reversion rather than mispricing: after a spike, trailing realized vol
is high and a lower forward IV is CORRECT. That is precisely why this belongs
in a ranker and not in a veto -- a reordering that is partly right still helps,
while a refusal that is partly wrong costs campaigns, and refusing campaigns is
how `min_iv_rank` cost 18 points of return on 2026-08-04.

THE UNITS. IV comes from the SOLVED history (`bench prep iv`), recorded at the
delta and DTE this bot actually sells, and realized vol comes from
`regime_series` -- both annualised, so the ratio is dimensionless and one
threshold means the same thing across 530 names.
"""
from __future__ import annotations

import math

#: Where an UNMEASURABLE name sorts. 1.0 is fair value on this statistic -- the
#: option priced exactly at the volatility the stock delivered -- so it is the
#: honest "no information" placement, outranking a measurably underpaid name and
#: losing to a measurably rich one.
#:
#: NOT a tuning knob, and deliberately NOT `NEUTRAL_IV_RANK`. That constant is
#: 0.5 because it is the midpoint of a PERCENTILE; this one is 1.0 because it is
#: the zero point of a RATIO. Sharing them would be a silent unit error, and
#: moving either re-introduces exactly the bias neutrality exists to remove:
#: sorting unknowns last is a filter wearing a sort's clothes (owner decision
#: 2026-08-05, and the reasoning in iv_rank.py applies here unchanged).
NEUTRAL_VRP = 1.0


def _usable(x) -> float | None:
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f) or f <= 0:
        return None
    return f


def iv_on(history, ticker: str, obs_date):
    """The most recent implied vol AT OR BEFORE `obs_date`, or None.

    Slices the series here rather than trusting the caller, for the same reason
    `IVHistory.rank` does: a caller cannot forget to do it, and cannot pass a
    pre-sliced series that is secretly too long.
    """
    if history is None:
        return None
    series = getattr(history, "series", None)
    s = series(ticker) if series is not None else None
    if s is None or len(s) == 0:
        return None
    import pandas as pd
    past = s.loc[:pd.Timestamp(obs_date)].dropna()
    if len(past) == 0:
        return None
    return _usable(past.iloc[-1])


def vrp_of(history, ticker: str, obs_date, realized_vol) -> float | None:
    """Implied over realized for one name on one day, or None if unmeasurable.

    A missing, zero or NaN realized vol is UNKNOWN, never infinity: dividing by
    it would score the calmest name in the book as the single richest
    opportunity in it -- the most flattering possible answer to the least
    measurable case, which is the shape of every zombie-gate failure this
    project has recorded.
    """
    rv = _usable(realized_vol)
    if rv is None:
        return None
    iv = iv_on(history, ticker, obs_date)
    if iv is None:
        return None
    return iv / rv


def sort_key(value) -> float:
    """Where a name sits in the queue. Unmeasurable sorts NEUTRAL."""
    v = _usable(value)
    return NEUTRAL_VRP if v is None else v


#: Batch 5 Arm 2 (2026-08-22): the viable-tier floor for rank_by="vrp_viable".
#: A CONSTANT, never a knob -- pinned to the measured 0.30-delta median
#: annualized yield (30.9%, yield-floor work), "at least median pay for this
#: delta", a prior measurement rather than a fitted number. Never swept: the
#: vrp-ranker verdict (2026-08-17) killed pure VRP on slot economics, not on
#: the ranking itself, so this arm holds economics fixed with one pinned rung
#: rather than reopening a tuning question that was never the failure.
VRP_VIABLE_MIN_ANN_YIELD = 0.30

#: Keeps tier-1 (VRP desc, ~0-5 typical) strictly ahead of tier-2 (vol_pctile
#: desc, bounded [0, 1]) inside ONE sortable key, so the pool tuple never
#: widens -- the reason `prefer_chop_half` carries its tier in a side dict
#: instead of the key itself does not apply here, because unlike that knob
#: this tiering IS the rank_by value, not a wrapper over an arbitrary one.
VRP_VIABLE_TIER_OFFSET = 10.0


def is_viable(ann_yield) -> bool:
    """Tier-1 membership for rank_by="vrp_viable". >=, not >, so the pinned
    constant means exactly what it says at the boundary."""
    return ann_yield is not None and ann_yield >= VRP_VIABLE_MIN_ANN_YIELD


def viable_sort_key(vrp_value) -> float:
    """Where a TIER-1 name sits in the queue: VRP descending (unmeasurable ->
    NEUTRAL, same rule as `sort_key`), offset clear of every tier-2 key."""
    return VRP_VIABLE_TIER_OFFSET + sort_key(vrp_value)
