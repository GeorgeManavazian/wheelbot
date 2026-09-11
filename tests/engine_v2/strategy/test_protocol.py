import pytest
import pandas as pd
from src.engine_v2.strategy.protocol import validate_plugin, PluginContractError

class GoodStrat:
    display_name = "Test Strat"
    mechanism = "Mean reversion on 5d Z-score"
    parameter_grid = {"lookback": [5, 10]}
    holding_period_cap = 5
    def forecast(self, bars, asof):
        return pd.Series(dtype=float)

class MissingMechanism:
    display_name = "X"
    parameter_grid = {}
    holding_period_cap = 1
    def forecast(self, bars, asof): return pd.Series(dtype=float)

class MissingDisplayName:
    mechanism = "x"
    parameter_grid = {}
    holding_period_cap = 1
    def forecast(self, bars, asof): return pd.Series(dtype=float)

class NoHoldingPeriodCap:
    display_name = "X"; mechanism = "x"; parameter_grid = {}
    def forecast(self, bars, asof): return pd.Series(dtype=float)

def test_good_strategy_validates():
    validate_plugin(GoodStrat)

def test_missing_mechanism_rejected():
    with pytest.raises(PluginContractError, match="mechanism"):
        validate_plugin(MissingMechanism)

def test_missing_display_name_rejected():
    with pytest.raises(PluginContractError, match="display_name"):
        validate_plugin(MissingDisplayName)

def test_missing_holding_period_cap_rejected():
    with pytest.raises(PluginContractError, match="holding_period_cap"):
        validate_plugin(NoHoldingPeriodCap)
