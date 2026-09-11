"""Carver Ch.4-7 sizing: target-risk / instrument-sigma / forecast * IDM.
notional = target_risk * equity / instrument_sigma * (forecast / mean_mag) * idm
"""
from __future__ import annotations
from math import sqrt

HALF_KELLY_SR = 0.24
DEFAULT_TARGET_RISK = HALF_KELLY_SR / 2  # 0.12
FORECAST_MEAN_MAG = 10.0

def annualize_sigma(daily_sigma: float) -> float:
    return daily_sigma * sqrt(252)

def size_position(forecast: float, equity: float, instrument_sigma: float,
                  target_risk: float = DEFAULT_TARGET_RISK, idm: float = 1.0) -> float:
    if instrument_sigma <= 0:
        return 0.0
    return target_risk * equity / instrument_sigma * (forecast / FORECAST_MEAN_MAG) * idm

_IDM_MULTI = [(1, 1.0), (2, 1.2), (3, 1.4), (5, 1.7), (10, 2.1), (15, 2.5)]
_IDM_SINGLE = [(1, 1.0), (2, 1.1), (3, 1.2), (5, 1.3), (10, 1.35), (15, 1.4)]

def idm_lookup(n_instruments: int, kind: str = "multi_asset") -> float:
    table = _IDM_MULTI if kind == "multi_asset" else _IDM_SINGLE
    last = table[0][1]
    for threshold, val in table:
        if n_instruments >= threshold:
            last = val
        else:
            break
    return last
