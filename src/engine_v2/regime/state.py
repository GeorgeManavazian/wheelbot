"""Per-day market regime from daily closes alone. Fixed ex-ante thresholds —
never swept. All windows trailing: the state on day d uses closes <= d only.

NOTE: a legacy classifier exists at src/engine_v2/data/regime.py (bull/bear +
absolute-vol calm/high, consumed by backtest metrics). The two are DIFFERENT
taxonomies for different consumers; if you change thresholds here, check
whether that one needs the same intent. Consolidation deferred deliberately
(regime-advisor spec, declined items).

Effective warmup: the first emitted state needs the 200d SMA (WARMUP) AND a
vol percentile, which needs VOL_WINDOW returns + VOL_MIN observations —
in practice the first row lands ~273 trading days in, not 200."""
from __future__ import annotations
import numpy as np
import pandas as pd

WARMUP = 200          # SMA200 horizon; real first-state day is later (see above)
VOL_WINDOW = 21       # realized-vol window (days)
VOL_LOOKBACK = 756    # percentile lookback (~3y)
VOL_MIN = 252         # minimum history for the percentile (~1y)
CALM, STRESSED = 0.40, 0.75
HIGH_WINDOW = 252     # rolling-high window for drawdown

# The columns emitted before 2026-08-15, in their original order. Everything
# added since is APPENDED after these, so a consumer that reads the frame
# positionally (none known, but the guarantee is free) is unaffected.
BASE_COLUMNS = ["trend", "vol", "px_vs_200", "px_vs_50", "ma50_vs_200",
                "fast_spread", "drawdown", "realized_vol", "vol_pctile"]

# Added 2026-08-15 for the weather-gate re-labelling candidates. Every one is a
# QUANTITY the old frame never carried: the taxonomy tested two ordering
# relations and a relative-vol percentile, and nothing else. Computing them
# costs a few rolling means per ticker and changes no existing column, so the
# gate stays byte-identical until a knob asks for one.
#
#   chop_half           "A"/"B" -- which side of the residual bucket (see below)
#   px_vs_21            close vs the 21d mean: the dip itself, on the hold horizon
#   fast_spread_11_21   the 11/21 pair, matched to target_dte=11 (the live 9/20
#                       pair was inherited, never chosen against a matched one)
#   ma_band             (max - min) of {9,21,50}d as a fraction of price:
#                       CONVERGENCE, which is what "range-bound" actually means
#   ma50_slope_5/10/20  the 50d's own slope. `sma50 > sma200` is a LEVEL over a
#                       200-day window and can be months stale; slope is now.
#   sma9_slope_3        has the fastest average TURNED UP? (dip stopped vs still
#                       falling -- the level test `fast_spread` cannot say)
#   dip_z               (px - sma21) in units of the name's own 21d sigma, so one
#                       threshold means the same thing across 530 names
NEW_COLUMNS = ["chop_half", "px_vs_21", "fast_spread_11_21", "ma_band",
               "ma50_slope_5", "ma50_slope_10", "ma50_slope_20",
               "sma9_slope_3", "dip_z"]

COLUMNS = BASE_COLUMNS + NEW_COLUMNS

#: The two states `trend == "chop"` merges. Measured 2026-08-15 on 272 realized
#: campaigns: A is 74% of entries, assigns at 22.3% and earns $56/campaign; B is
#: 23% of slot-days, assigns at 12.9%, earns $346 and carried none of the six
#: losers. `px_vs_200` and `px_vs_50` are the SAME test inside chop -- in A
#: sma50 < sma200 < c, in B c < sma200 < sma50 -- so "above the 200 and below
#: the 50" is structurally impossible and the halves are exhaustive.
CHOP_HALVES = ("A", "B")


def chop_half_of(row) -> str | None:
    """"A" (above both MAs, 50 below 200), "B" (below both, 50 above 200), or
    None when the row cannot say. Derived from the two ordering relations rather
    than read off the frame, so it works on the hand-built dict rows the tests
    use and on a live `LiveMarket` row alike."""
    if row is None:
        return None
    px, ma = _num(row, "px_vs_200"), _num(row, "ma50_vs_200")
    if px is None or ma is None:
        return None
    if px >= 0 and ma < 0:
        return "A"
    if px < 0 and ma >= 0:
        return "B"
    return None            # a clean uptrend or downtrend: neither half


def _num(row, key):
    """A float field off a dict row or a pandas Series, or None if it is absent
    or NaN. Absent and NaN are the SAME answer on purpose -- `liquidity_ok` has
    refused on both since A2, and a gate that treats "no data" as "passes" is
    the zombie-gate failure this project has already paid for twice."""
    try:
        v = row[key]
    except (KeyError, IndexError, TypeError):
        return None
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def regime_series(closes: pd.Series) -> pd.DataFrame:
    c = closes.dropna().astype(float)
    if len(c) <= WARMUP:
        return pd.DataFrame(columns=COLUMNS)
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    sma9, sma20 = c.rolling(9).mean(), c.rolling(20).mean()   # short-horizon (~2wk)
    sma11, sma21 = c.rolling(11).mean(), c.rolling(21).mean() # horizon-matched
    logret = np.log(c / c.shift(1))
    rv = logret.rolling(VOL_WINDOW).std() * np.sqrt(252)
    # percentile of today's realized vol within its own trailing window
    pct = rv.rolling(VOL_LOOKBACK, min_periods=VOL_MIN).rank(pct=True)
    high = c.rolling(HIGH_WINDOW, min_periods=1).max()
    band = (pd.concat([sma9, sma21, sma50], axis=1).max(axis=1)
            - pd.concat([sma9, sma21, sma50], axis=1).min(axis=1)) / c
    df = pd.DataFrame({
        "px_vs_200": c / sma200 - 1,
        "px_vs_50": c / sma50 - 1,
        "ma50_vs_200": sma50 / sma200 - 1,
        "fast_spread": sma9 / sma20 - 1,   # 9d/20d gap: ~0 = flat over our hold
        "drawdown": c / high - 1,
        "realized_vol": rv,
        "vol_pctile": pct,
        "px_vs_21": c / sma21 - 1,
        "fast_spread_11_21": sma11 / sma21 - 1,
        "ma_band": band,
        "ma50_slope_5": sma50 / sma50.shift(5) - 1,
        "ma50_slope_10": sma50 / sma50.shift(10) - 1,
        "ma50_slope_20": sma50 / sma50.shift(20) - 1,
        "sma9_slope_3": sma9 / sma9.shift(3) - 1,
        # (px - sma21) in units of one 21-day sigma. rv is annualised, so scale
        # it back to the 21-day horizon the dip is measured over.
        "dip_z": (c - sma21) / (rv * c * np.sqrt(VOL_WINDOW / 252.0)),
    })
    up = (c > sma200) & (sma50 > sma200)
    down = (c < sma200) & (sma50 < sma200)
    df["trend"] = np.where(up, "uptrend", np.where(down, "downtrend", "chop"))
    df["vol"] = np.where(df["vol_pctile"] < CALM, "calm",
                np.where(df["vol_pctile"] > STRESSED, "stressed", "normal"))
    df["chop_half"] = np.where((c >= sma200) & (sma50 < sma200), "A",
                      np.where((c < sma200) & (sma50 >= sma200), "B", ""))
    df = df.iloc[WARMUP:].dropna(subset=["px_vs_200", "vol_pctile"])
    return df[COLUMNS]

def describe(row) -> str:
    side = "above" if row["px_vs_200"] >= 0 else "below"
    cross = "50>200" if row["ma50_vs_200"] >= 0 else "50<200"
    return (f"{row['trend'].capitalize()} ({abs(row['px_vs_200']):.1%} {side} 200d, {cross}), "
            f"{row['vol']} vol ({row['vol_pctile']:.0%} pctile), "
            f"{abs(row['drawdown']):.1%} off 252d high.")

#: WheelConfig field -> the keyword `is_good_renting_weather` knows it by.
#: ONE mapping, consumed by `weather_kwargs`, so a new gate knob reaches the
#: backtest AND the live bot from a single edit. `earnings_blackout` spent
#: weeks ON and blocking nothing because its wiring lived at four call sites
#: and one of them never passed it; this is that class of bug closed by
#: construction.
GATE_KNOBS = {
    "chop_max_ma_spread": "max_ma_spread",
    "chop_max_fast_spread": "max_fast_spread",
    "chop_max_fast_fall": "max_fast_fall",
    "chop_half": "chop_half",
    "chop_fast_pair": "fast_pair",
    "chop_max_realized_vol": "max_realized_vol",
    "chop_max_ma_band": "max_ma_band",
    "chop_min_ma50_slope": "min_ma50_slope",
    "chop_ma50_slope_window": "ma50_slope_window",
    "chop_drawdown_band": "drawdown_band",
    "chop_dip_z_band": "dip_z_band",
    "chop_require_fast_turn": "require_fast_turn",
    "chop_drop_trend_label": "drop_trend_label",
}

FAST_PAIRS = {None: "fast_spread", "9/20": "fast_spread",
              "11/21": "fast_spread_11_21"}


def weather_kwargs(cfg) -> dict:
    """Every weather-gate knob on a WheelConfig, as this module's keywords.

    Duck-typed via getattr so `regime/` keeps its no-import-from-`options/`
    direction, and so a checkout whose WheelConfig predates a knob simply omits
    it rather than raising."""
    out = {}
    for field, kw in GATE_KNOBS.items():
        if hasattr(cfg, field):
            out[kw] = getattr(cfg, field)
    return out


def _in_band(v, band) -> bool:
    """band = [lo, hi], inclusive. A band rather than a threshold on purpose:
    a band can be checked for a PLATEAU (does the neighbourhood agree?), which
    is the only defence this project has found against fitting a number to 54
    events -- see the -40% stop, which fired 3 times and did not survive its
    own neighbours."""
    lo, hi = float(band[0]), float(band[1])
    return lo <= v <= hi


def is_good_renting_weather(row, max_ma_spread=None, max_fast_spread=None,
                            max_fast_fall=None, *,
                            chop_half=None, fast_pair=None,
                            max_realized_vol=None, max_ma_band=None,
                            min_ma50_slope=None, ma50_slope_window=None,
                            drawdown_band=None, dip_z_band=None,
                            require_fast_turn=False,
                            drop_trend_label=False) -> bool:
    """Good-to-rent weather for the chop scanner: range-bound (chop) and not
    violently volatile (not stressed). Uptrends (hold instead) and downtrends
    (falling knife) are excluded; a None/unknown row is not good-to-rent.

    Opt-in guards (default None = off so the backtest stays byte-identical):
      max_ma_spread  -- |50d/200d - 1| <= this. STRUCTURAL (months): the bare
        crossover labels a fast move "chop" until the 50d catches through the
        200d, so a knife mid-fall (MAs pulling apart) slips through. Tight = flat.
      max_fast_spread -- |9d/20d - 1| <= this. TACTICAL, SYMMETRIC (~2wk): rejects
        any short-horizon leg, up or down.
      max_fast_fall  -- reject only if 9d/20d - 1 < -this. TACTICAL, DOWN-ONLY: a
        put seller only loses on a FALL, so drop a down-leg but KEEP an up-leg
        (which expires the put worthless). Prefer this over max_fast_spread.

    Added 2026-08-15, all default-off, each attacking one thing the label above
    never measures (see the ten-candidate brief):
      chop_half      -- "A" or "B": keep only one side of the residual bucket.
        `trend == "chop"` is EVERYTHING ELSE, and it merges a name recovering
        off a downtrend (A) with a name dipping inside an intact uptrend (B).
      fast_pair      -- "11/21" reads the horizon-matched pair instead of the
        inherited 9/20 for BOTH fast guards. target_dte is 11.
      max_realized_vol -- absolute annualised vol ceiling. `vol_pctile` is
        measured against the name's OWN history, so a 60%-vol name at its median
        passes and a 15%-vol name at its 90th is refused -- across 530 names that
        normalises away the quantity being controlled for.
      max_ma_band    -- (max-min of the 9/21/50 means) / price. CONVERGENCE:
        what "range-bound" is supposed to mean, rather than two orderings.
      min_ma50_slope / ma50_slope_window -- the 50d's own slope over N days.
        A 50 above the 200 but FALLING is a name breaking down, and the level
        test cannot tell it from one recovering.
      drawdown_band  -- [lo, hi] on `drawdown` (<= 0). At the high there is no
        dip to be paid for; below the floor the name is impaired, not dipping.
      dip_z_band     -- [lo, hi] on (px - sma21) in 21-day sigmas. Makes one
        threshold mean the same thing in a 15%-vol and a 60%-vol name.
      require_fast_turn -- px below the 21d mean AND the 9d mean rising over 3
        days: a dip that has STOPPED, not one still in progress.
      drop_trend_label -- True drops the three-way label entirely and gates on
        the direct facts only. The upper bound of what re-labelling can buy.

    A knob that is SET and whose field is missing or NaN REFUSES, matching
    `liquidity_ok`: a condition that cannot be measured is not one to sell into.
    The three original guards keep their original field access, unchanged."""
    if row is None:
        return False
    if not drop_trend_label and row["trend"] != "chop":
        return False
    if row["vol"] == "stressed":
        return False
    if max_ma_spread is not None and abs(row["ma50_vs_200"]) > max_ma_spread:
        return False

    fast_col = FAST_PAIRS.get(fast_pair)
    if fast_col is None:
        raise ValueError(
            f"fast_pair must be one of {sorted(k for k in FAST_PAIRS if k)}, "
            f"got {fast_pair!r}")
    if max_fast_spread is not None or max_fast_fall is not None:
        fast = (row["fast_spread"] if fast_col == "fast_spread"
                else _num(row, fast_col))
        if fast is None:
            return False
        if max_fast_spread is not None and abs(fast) > max_fast_spread:
            return False
        if max_fast_fall is not None and fast < -max_fast_fall:
            return False

    if chop_half is not None:
        if chop_half not in CHOP_HALVES:
            raise ValueError(f"chop_half must be one of {CHOP_HALVES}, "
                             f"got {chop_half!r}")
        if chop_half_of(row) != chop_half:
            return False
    if max_realized_vol is not None:
        rv = _num(row, "realized_vol")
        if rv is None or rv > max_realized_vol:
            return False
    if max_ma_band is not None:
        b = _num(row, "ma_band")
        if b is None or b > max_ma_band:
            return False
    if min_ma50_slope is not None:
        col = f"ma50_slope_{int(ma50_slope_window or 10)}"
        s = _num(row, col)
        if s is None or s < min_ma50_slope:
            return False
    if drawdown_band is not None:
        dd = _num(row, "drawdown")
        if dd is None or not _in_band(dd, drawdown_band):
            return False
    if dip_z_band is not None:
        z = _num(row, "dip_z")
        if z is None or not _in_band(z, dip_z_band):
            return False
    if require_fast_turn:
        px21, turn = _num(row, "px_vs_21"), _num(row, "sma9_slope_3")
        if px21 is None or turn is None:
            return False
        if not (px21 < 0 and turn > 0):
            return False
    return True
