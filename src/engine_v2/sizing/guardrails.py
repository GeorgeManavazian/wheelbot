"""Hard-coded sizing guardrails per Carver + engine spec."""
from __future__ import annotations

SPEED_LIMIT = 0.08
HARD_ACCOUNT_CAP = 0.30

class SpeedLimitViolation(ValueError):
    pass

class AccountCapExceeded(ValueError):
    pass

def check_speed_limit(risk_adj_cost: float) -> None:
    if risk_adj_cost > SPEED_LIMIT:
        raise SpeedLimitViolation(
            f"risk_adj_cost={risk_adj_cost:.4f} > SPEED_LIMIT={SPEED_LIMIT}"
        )

def enforce_account_cap(target_risk: float) -> None:
    if target_risk > HARD_ACCOUNT_CAP:
        raise AccountCapExceeded(
            f"target_risk={target_risk} > HARD_ACCOUNT_CAP={HARD_ACCOUNT_CAP}"
        )

def lifo_derisk(active_instruments: list[str], to_drop: int) -> list[str]:
    if to_drop < 0 or to_drop > len(active_instruments):
        raise ValueError(f"cannot drop {to_drop} from {len(active_instruments)} instruments")
    return list(reversed(active_instruments[-to_drop:])) if to_drop else []
