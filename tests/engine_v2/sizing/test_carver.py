import pytest
from src.engine_v2.sizing.carver import (
    size_position, annualize_sigma, idm_lookup,
    HALF_KELLY_SR, DEFAULT_TARGET_RISK,
)

def test_carver_worked_example_starter():
    # Ch.5 p.107: forecast +10 (mean), equity=$100k, annual sigma=20%,
    # target=12%, idm=1 -> notional = 0.12 * 100k / 0.20 * (10/10) = $60k
    n = size_position(forecast=10, equity=100_000, instrument_sigma=0.20)
    assert n == pytest.approx(60_000)

def test_forecast_20_doubles_position():
    n10 = size_position(10, 100_000, 0.20)
    n20 = size_position(20, 100_000, 0.20)
    assert n20 == pytest.approx(2 * n10)

def test_negative_forecast_short_position():
    assert size_position(-10, 100_000, 0.20) == pytest.approx(-60_000)

def test_zero_sigma_returns_zero_no_zerodiv():
    assert size_position(10, 100_000, 0.0) == 0.0

def test_annualize_sigma():
    assert annualize_sigma(0.01) == pytest.approx(0.01 * (252 ** 0.5))

def test_idm_lookup_known_cells():
    assert idm_lookup(1) == 1.0
    assert idm_lookup(2) == 1.2
    assert idm_lookup(5) == 1.7
    assert idm_lookup(50) == 2.5
    assert idm_lookup(1, kind="single_asset") == 1.0
    assert idm_lookup(50, kind="single_asset") == 1.4

def test_half_kelly_constant():
    assert HALF_KELLY_SR == 0.24
    assert DEFAULT_TARGET_RISK == HALF_KELLY_SR / 2  # 0.12
