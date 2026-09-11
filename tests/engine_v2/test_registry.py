import pytest
from src.engine_v2.strategy.registry import STRATEGIES, get_strategy
from src.engine_v2.strategy.protocol import validate_plugin

def test_registry_lists_known_plugins():
    names = list(STRATEGIES)
    assert any("Counter-Trend" in n for n in names)
    assert len(names) >= 2

def test_every_registered_plugin_is_valid():
    for cls in STRATEGIES.values():
        validate_plugin(cls)  # raises if not

def test_get_strategy_roundtrips():
    name = next(iter(STRATEGIES))
    assert get_strategy(name) is STRATEGIES[name]

def test_get_strategy_unknown_raises():
    with pytest.raises(KeyError):
        get_strategy("nope")
