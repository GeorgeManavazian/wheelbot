"""V2 Strategy plugin contract. Plugins are pure signal generators — no sizing,
no execution, no look-ahead. Missing metadata is a load-time failure."""
from __future__ import annotations
from typing import Protocol
import pandas as pd

FORECAST_MIN, FORECAST_MAX = -20.0, 20.0
FORECAST_MEAN_MAG = 10.0

class PluginContractError(TypeError):
    pass

class StrategyV2(Protocol):
    display_name: str
    mechanism: str
    parameter_grid: dict
    holding_period_cap: int
    def forecast(self, bars: pd.DataFrame, asof: pd.Timestamp) -> pd.Series: ...

REQUIRED_ATTRS = ("display_name", "mechanism", "parameter_grid", "holding_period_cap")

def validate_plugin(cls) -> None:
    for attr in REQUIRED_ATTRS:
        if not hasattr(cls, attr):
            raise PluginContractError(f"{cls.__name__}: missing required attr {attr!r}")
    if not getattr(cls, "display_name"):
        raise PluginContractError(f"{cls.__name__}: display_name is empty")
    if not getattr(cls, "mechanism"):
        raise PluginContractError(f"{cls.__name__}: mechanism is empty")
    if not callable(getattr(cls, "forecast", None)):
        raise PluginContractError(f"{cls.__name__}: forecast must be callable")
