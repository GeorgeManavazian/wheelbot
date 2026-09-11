"""Implied volatility for a short PUT, computed by us rather than read off a vendor.

The scale must not change inside a ticker's 252-observation window (owner ruling
2026-08-07). Vendors disagree on IV and the live bot pulls Schwab daily while the
history came from someone else, so the only way to hold one scale is to take
everyone's QUOTES and compute the IV here.

Convention is fitted to 6,240 live Schwab contracts, 2026-08-06: calendar days/365
(the 252 convention is -7.4 vol points off), inverted from the mid, rate and
dividend yield taken continuous off the chain header. See tests/engine_v2/options/
test_iv_solve.py for the full table and the OTM-only restriction it enforces.
"""
from __future__ import annotations

import math

DAYS_PER_YEAR = 365.0


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_put(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)


def implied_vol_put(price: float, underlying: float, strike: float, dte: int,
                    rate: float, div_yield: float) -> float:
    """Volatility that reprices this put to `price`. Calendar days / 365.

    Refuses an ITM put: the European/continuous-yield approximation is only
    supported OTM, and returning a number outside that band would be quietly
    wrong rather than loudly absent.
    """
    if strike > underlying:
        raise ValueError(
            f"put is in the money (strike {strike} > underlying {underlying}); "
            f"this solver is only supported out of the money -- see "
            f"tests/engine_v2/options/test_iv_solve.py")
    T = dte / DAYS_PER_YEAR
    lo, hi = 1e-6, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _bs_put(underlying, strike, T, rate, div_yield, mid) < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0
