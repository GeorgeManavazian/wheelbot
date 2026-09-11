"""The 2026-08-15 weather-gate knobs: each is a no-op at its default, and each
refuses the thing it claims to refuse.

WHY BOTH HALVES ARE REQUIRED. The default-off half is what lets `frozen` resolve
byte-identical while ten candidates sit in the tree -- without it, adding a knob
is a silent change to the live bot. The refusal half is what stops a knob being
a zombie: `earnings_blackout` was ON in the config and blocking nothing for
weeks, because every test it had proved the flag was readable, not that it
refused anything.

Rows here are plain dicts, matching the existing gate tests -- `_row_before`
hands the engine a pandas Series, and both support the same `row[key]` access
the gate uses.
"""
import numpy as np
import pandas as pd
import pytest

from src.engine_v2.options.wheel import WheelConfig
from src.engine_v2.regime.state import (BASE_COLUMNS, CHOP_HALVES, COLUMNS,
                                        GATE_KNOBS, chop_half_of,
                                        is_good_renting_weather, regime_series,
                                        weather_kwargs)

# A row that passes the gate with every knob off. Carries every field the new
# knobs read, so "off" can be proved against values that WOULD refuse.
GOOD = {
    "trend": "chop", "vol": "normal",
    "px_vs_200": -0.04, "px_vs_50": -0.02, "ma50_vs_200": 0.01,
    "fast_spread": 0.002, "drawdown": -0.10,
    "realized_vol": 0.28, "vol_pctile": 0.55,
    "chop_half": "B", "px_vs_21": -0.03, "fast_spread_11_21": 0.001,
    "ma_band": 0.03, "ma50_slope_5": 0.004, "ma50_slope_10": 0.006,
    "ma50_slope_20": 0.012, "sma9_slope_3": 0.003, "dip_z": -0.8,
}


def row(**over):
    r = dict(GOOD)
    r.update(over)
    return r


# ---------------------------------------------------------------------------
# The frame grew columns; it did not change the ones it had.
# ---------------------------------------------------------------------------

def _series(n=900, seed=0):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0004, 0.012, n)
    px = 100 * np.exp(np.cumsum(steps))
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.Series(px, index=idx)


def test_regime_series_keeps_its_original_columns_first():
    df = regime_series(_series())
    assert list(df.columns) == COLUMNS
    assert list(df.columns)[:len(BASE_COLUMNS)] == BASE_COLUMNS


def test_regime_series_new_columns_are_finite_after_warmup():
    df = regime_series(_series())
    assert len(df) > 100
    for c in ("px_vs_21", "fast_spread_11_21", "ma_band", "ma50_slope_5",
              "ma50_slope_10", "ma50_slope_20", "sma9_slope_3", "dip_z"):
        assert df[c].notna().all(), f"{c} carries NaN after warmup"
    assert (df["ma_band"] >= 0).all()          # a max minus a min over a price


def test_regime_series_chop_half_partitions_chop_exactly():
    """Inside chop the two halves are exhaustive and mutually exclusive, and
    every non-chop row has no half. This is the claim the whole exercise rests
    on -- if a chop row could be neither, `chop_half="B"` would be silently
    dropping rows the note says it keeps."""
    df = regime_series(_series())
    chop = df[df["trend"] == "chop"]
    assert len(chop) > 20
    assert set(chop["chop_half"]) <= set(CHOP_HALVES)
    assert (chop["chop_half"] != "").all()
    assert (df[df["trend"] != "chop"]["chop_half"] == "").all()


def test_chop_half_of_matches_the_frame():
    df = regime_series(_series())
    for _, r in df[df["trend"] == "chop"].head(50).iterrows():
        assert chop_half_of(r) == r["chop_half"]


def test_short_history_frame_still_carries_every_column():
    df = regime_series(_series(n=50))
    assert list(df.columns) == COLUMNS and df.empty


# ---------------------------------------------------------------------------
# Off is off. One test per knob, against a value that WOULD refuse.
# ---------------------------------------------------------------------------

REFUSING_VALUES = {
    "chop_half": "A",                      # GOOD is half B
    "max_realized_vol": 0.10,              # GOOD carries 0.28
    "max_ma_band": 0.01,                   # GOOD carries 0.03
    "min_ma50_slope": 0.05,                # GOOD carries 0.006 at 10d
    "drawdown_band": [-0.05, -0.01],       # GOOD carries -0.10
    "dip_z_band": [0.5, 1.5],              # GOOD carries -0.8
    "require_fast_turn": True,             # proved separately (needs a turn row)
}


@pytest.mark.parametrize("kw", sorted(REFUSING_VALUES))
def test_each_knob_is_a_no_op_when_unset(kw):
    """The gate with NOTHING set passes GOOD even though GOOD carries a value
    that each knob would refuse."""
    assert is_good_renting_weather(row()) is True


@pytest.mark.parametrize("kw,val", sorted(REFUSING_VALUES.items()))
def test_each_knob_refuses_what_it_claims(kw, val):
    if kw == "require_fast_turn":
        # GOOD is already below its 21d mean and turning up, so it PASSES;
        # the refusal case is a dip still falling.
        assert is_good_renting_weather(row(), require_fast_turn=True) is True
        assert is_good_renting_weather(row(sma9_slope_3=-0.004),
                                       require_fast_turn=True) is False
        return
    assert is_good_renting_weather(row(), **{kw: val}) is False


def test_default_wheelconfig_gate_is_the_old_gate():
    """`weather_kwargs(WheelConfig())` must leave the gate exactly where it was:
    only the three 2026-07-18 guards, all off."""
    kw = weather_kwargs(WheelConfig())
    assert kw["max_ma_spread"] is None
    assert kw["max_fast_spread"] is None
    assert kw["max_fast_fall"] is None
    assert all(kw[k] in (None, False) for k in kw)
    assert is_good_renting_weather(row(), **kw) is True
    assert is_good_renting_weather(row(trend="uptrend"), **kw) is False


def test_weather_kwargs_covers_every_chop_field_on_the_config():
    """A chop_* field that is not in GATE_KNOBS never reaches the gate, from
    anywhere -- which is the zombie-gate shape. Fail here, not in production."""
    cfg = WheelConfig()
    chop_fields = {f for f in vars(cfg) if f.startswith("chop_")}
    assert chop_fields == set(GATE_KNOBS), (
        f"unwired: {sorted(chop_fields - set(GATE_KNOBS))}")


# ---------------------------------------------------------------------------
# What each knob does when it IS on
# ---------------------------------------------------------------------------

def test_chop_half_keeps_its_own_half_and_refuses_the_other():
    b = row(px_vs_200=-0.04, ma50_vs_200=0.01)     # below 200, 50 above 200
    a = row(px_vs_200=0.04, ma50_vs_200=-0.01)     # above 200, 50 below 200
    assert chop_half_of(b) == "B" and chop_half_of(a) == "A"
    assert is_good_renting_weather(b, chop_half="B") is True
    assert is_good_renting_weather(a, chop_half="B") is False
    assert is_good_renting_weather(a, chop_half="A") is True
    assert is_good_renting_weather(b, chop_half="A") is False


def test_fast_pair_reads_the_other_column():
    """9/20 flat, 11/21 falling hard. The same threshold must reach opposite
    verdicts, or the knob is not switching column."""
    r = row(fast_spread=0.000, fast_spread_11_21=-0.05)
    assert is_good_renting_weather(r, max_fast_fall=0.01) is True
    assert is_good_renting_weather(r, max_fast_fall=0.01,
                                   fast_pair="11/21") is False
    r2 = row(fast_spread=-0.05, fast_spread_11_21=0.000)
    assert is_good_renting_weather(r2, max_fast_fall=0.01) is False
    assert is_good_renting_weather(r2, max_fast_fall=0.01,
                                   fast_pair="11/21") is True


def test_fast_pair_9_20_is_the_explicit_spelling_of_the_default():
    r = row(fast_spread=-0.05)
    assert is_good_renting_weather(r, max_fast_fall=0.01, fast_pair="9/20") \
        is is_good_renting_weather(r, max_fast_fall=0.01)


def test_realized_vol_ceiling_is_absolute_not_percentile():
    """The point of C4: a name at a middling percentile of its OWN history can
    still be violently volatile in absolute terms, and vol_pctile cannot see
    it. Same vol_pctile, opposite verdicts."""
    calm_pctile_wild_name = row(realized_vol=0.85, vol_pctile=0.50)
    assert is_good_renting_weather(calm_pctile_wild_name) is True
    assert is_good_renting_weather(calm_pctile_wild_name,
                                   max_realized_vol=0.50) is False
    assert is_good_renting_weather(row(realized_vol=0.30, vol_pctile=0.50),
                                   max_realized_vol=0.50) is True


def test_ma_band_measures_convergence():
    assert is_good_renting_weather(row(ma_band=0.02), max_ma_band=0.05) is True
    assert is_good_renting_weather(row(ma_band=0.09), max_ma_band=0.05) is False


def test_ma50_slope_window_selects_its_column():
    """A 50d rising over 5 days and falling over 20 -- the window is the
    experiment, so reading the wrong column is a silent wrong answer."""
    r = row(ma50_slope_5=0.01, ma50_slope_10=0.0, ma50_slope_20=-0.03)
    assert is_good_renting_weather(r, min_ma50_slope=0.0,
                                   ma50_slope_window=5) is True
    assert is_good_renting_weather(r, min_ma50_slope=0.0,
                                   ma50_slope_window=20) is False
    # None means 10, the documented default
    assert is_good_renting_weather(r, min_ma50_slope=0.0) is \
        is_good_renting_weather(r, min_ma50_slope=0.0, ma50_slope_window=10)


def test_drawdown_band_refuses_both_edges():
    band = [-0.20, -0.05]
    assert is_good_renting_weather(row(drawdown=-0.12), drawdown_band=band) is True
    assert is_good_renting_weather(row(drawdown=-0.01), drawdown_band=band) is False
    assert is_good_renting_weather(row(drawdown=-0.35), drawdown_band=band) is False
    # inclusive at both edges, so a plateau scan has no gaps in it
    assert is_good_renting_weather(row(drawdown=-0.05), drawdown_band=band) is True
    assert is_good_renting_weather(row(drawdown=-0.20), drawdown_band=band) is True


def test_dip_z_band_is_scale_free():
    """A 3% dip is one sigma in a wild name and three in a quiet one. The band
    must keep the first and refuse the second on the SAME percentage move."""
    band = [-1.5, -0.25]
    assert is_good_renting_weather(row(px_vs_21=-0.03, dip_z=-1.0),
                                   dip_z_band=band) is True
    assert is_good_renting_weather(row(px_vs_21=-0.03, dip_z=-3.0),
                                   dip_z_band=band) is False
    assert is_good_renting_weather(row(px_vs_21=-0.03, dip_z=-0.1),
                                   dip_z_band=band) is False


def test_require_fast_turn_separates_a_stopped_dip_from_a_falling_one():
    stopped = row(px_vs_21=-0.03, sma9_slope_3=0.004)
    falling = row(px_vs_21=-0.03, sma9_slope_3=-0.004)
    above = row(px_vs_21=0.02, sma9_slope_3=0.004)      # not a dip at all
    assert is_good_renting_weather(stopped, require_fast_turn=True) is True
    assert is_good_renting_weather(falling, require_fast_turn=True) is False
    assert is_good_renting_weather(above, require_fast_turn=True) is False


def test_drop_trend_label_admits_names_the_taxonomy_refuses():
    up = row(trend="uptrend")
    down = row(trend="downtrend")
    assert is_good_renting_weather(up) is False
    assert is_good_renting_weather(down) is False
    assert is_good_renting_weather(up, drop_trend_label=True) is True
    assert is_good_renting_weather(down, drop_trend_label=True) is True


def test_dropping_the_label_never_drops_the_stressed_vol_guard():
    """C11 removes the TAXONOMY, not the volatility refusal. Losing the second
    silently would make the control arm a different experiment."""
    assert is_good_renting_weather(row(trend="uptrend", vol="stressed"),
                                   drop_trend_label=True) is False


# ---------------------------------------------------------------------------
# Unmeasurable refuses; nonsense raises.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kw,val,field", [
    ("max_realized_vol", 0.5, "realized_vol"),
    ("max_ma_band", 0.05, "ma_band"),
    ("min_ma50_slope", 0.0, "ma50_slope_10"),
    ("drawdown_band", [-0.2, -0.05], "drawdown"),
    ("dip_z_band", [-1.5, -0.25], "dip_z"),
])
def test_a_set_knob_refuses_when_its_field_is_missing_or_nan(kw, val, field):
    """Same stance as liquidity_ok: a threshold that is set and a field that
    cannot be read means REFUSE. Allowing would be a gate reporting protection
    it is not giving -- on data availability, silently."""
    missing = row()
    missing.pop(field)
    assert is_good_renting_weather(missing, **{kw: val}) is False
    assert is_good_renting_weather(row(**{field: float("nan")}),
                                   **{kw: val}) is False


def test_a_bad_half_or_pair_raises_rather_than_falling_back():
    with pytest.raises(ValueError):
        is_good_renting_weather(row(), chop_half="C")
    with pytest.raises(ValueError):
        is_good_renting_weather(row(), fast_pair="12/26")


def test_none_row_is_still_not_good_with_every_knob_on():
    assert is_good_renting_weather(None, chop_half="B", max_realized_vol=0.5,
                                   drop_trend_label=True) is False
