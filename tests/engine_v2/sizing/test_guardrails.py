import pytest
from src.engine_v2.sizing.guardrails import (
    check_speed_limit, enforce_account_cap, lifo_derisk,
    SPEED_LIMIT, HARD_ACCOUNT_CAP,
    SpeedLimitViolation, AccountCapExceeded,
)

def test_speed_limit_pass():
    check_speed_limit(0.05)

def test_speed_limit_at_boundary_passes():
    check_speed_limit(SPEED_LIMIT)  # equal is OK

def test_speed_limit_above_raises():
    with pytest.raises(SpeedLimitViolation):
        check_speed_limit(0.081)

def test_account_cap_pass():
    enforce_account_cap(0.25)

def test_account_cap_at_boundary_passes():
    enforce_account_cap(HARD_ACCOUNT_CAP)

def test_account_cap_above_raises():
    with pytest.raises(AccountCapExceeded):
        enforce_account_cap(0.31)

def test_lifo_drops_newest_first():
    assert lifo_derisk(["SPY", "TLT", "GLD", "IWM"], to_drop=2) == ["IWM", "GLD"]

def test_lifo_no_drop_empty():
    assert lifo_derisk(["SPY"], to_drop=0) == []

def test_lifo_drop_more_than_have():
    with pytest.raises(ValueError):
        lifo_derisk(["SPY"], to_drop=5)
