from src.engine_v2.regime.state import is_good_renting_weather


def _row(trend, vol):
    return {"trend": trend, "vol": vol}


def test_chop_normal_is_good():
    assert is_good_renting_weather(_row("chop", "normal")) is True


def test_chop_calm_is_good():
    assert is_good_renting_weather(_row("chop", "calm")) is True


def test_chop_stressed_is_not_good():
    assert is_good_renting_weather(_row("chop", "stressed")) is False


def test_uptrend_is_not_good():
    assert is_good_renting_weather(_row("uptrend", "normal")) is False


def test_downtrend_is_not_good():
    assert is_good_renting_weather(_row("downtrend", "normal")) is False


def test_none_row_is_not_good():
    assert is_good_renting_weather(None) is False


def _row2(trend, vol, ma50_vs_200):
    return {"trend": trend, "vol": vol, "ma50_vs_200": ma50_vs_200}


# --- ma-spread guard (2026-07-18): tight MAs = true chop; MAs pulling apart =
# a trend forming (the falling-knife / early-uptrend false positives). Opt-in ---
def test_ma_spread_guard_keeps_tight_chop():
    # BA-like: 50d only 1.7% from 200d, within a 3% band -> still good to rent
    assert is_good_renting_weather(_row2("chop", "normal", 0.017), max_ma_spread=0.03) is True


def test_ma_spread_guard_rejects_forming_downtrend():
    # VALE-like: MAs 6.7% apart -> reject
    assert is_good_renting_weather(_row2("chop", "normal", 0.067), max_ma_spread=0.03) is False


def test_ma_spread_guard_rejects_forming_uptrend():
    # WFC-like: 50d 3.9% below 200d (abs 3.9% > 3%) -> reject
    assert is_good_renting_weather(_row2("chop", "normal", -0.039), max_ma_spread=0.03) is False


def test_ma_spread_guard_default_off_is_unchanged():
    # no threshold -> old behavior; ma50_vs_200 is never consulted (may be absent)
    assert is_good_renting_weather(_row2("chop", "normal", 0.50)) is True
    assert is_good_renting_weather({"trend": "chop", "vol": "normal"}) is True


def _row3(trend, vol, ma50_vs_200, fast_spread):
    return {"trend": trend, "vol": vol, "ma50_vs_200": ma50_vs_200,
            "fast_spread": fast_spread}


# --- fast (9/20) spread guard (2026-07-18): flat on the ~2-week horizon that
# matches our ~11-day hold. Catches tactical legs the 50/200 view calls flat ---
def test_fast_spread_guard_keeps_short_flat():
    # BA-like: 9d/20d only -0.1% apart -> flat over our window -> keep
    assert is_good_renting_weather(_row3("chop", "normal", 0.017, -0.001),
                                   max_ma_spread=0.03, max_fast_spread=0.01) is True


def test_fast_spread_guard_rejects_short_leg():
    # AGNC-like: 50/200 flat (-0.3%) but 9/20 +2.1% -> a live up-leg -> reject
    assert is_good_renting_weather(_row3("chop", "normal", -0.003, 0.021),
                                   max_ma_spread=0.03, max_fast_spread=0.01) is False


def test_fast_spread_guard_default_off_ignores_fast():
    # no fast threshold -> fast_spread never consulted (old behavior preserved)
    assert is_good_renting_weather(_row3("chop", "normal", 0.017, 0.50)) is True


# --- down-only tactical gate (2026-07-18): a put seller only fears a FALL, so
# reject a down-leg but KEEP an up-leg (which expires the put worthless) ---
def test_fast_fall_gate_rejects_down_leg():
    # AA/VALE-like: 9d 5% below 20d -> falling -> reject
    assert is_good_renting_weather(_row3("chop", "normal", 0.0, -0.05),
                                   max_fast_fall=0.01) is False


def test_fast_fall_gate_keeps_up_leg():
    # AGNC/WFC-like: 9d 5% ABOVE 20d -> rising -> KEEP (put-seller's friend)
    assert is_good_renting_weather(_row3("chop", "normal", 0.0, 0.05),
                                   max_fast_fall=0.01) is True


def test_fast_fall_gate_keeps_flat():
    assert is_good_renting_weather(_row3("chop", "normal", 0.0, -0.001),
                                   max_fast_fall=0.01) is True


def test_fast_fall_gate_default_off():
    assert is_good_renting_weather(_row3("chop", "normal", 0.0, -0.90)) is True
