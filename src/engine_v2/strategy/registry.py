"""Central list of runnable strategies. Add a plugin = drop a file in this
package + one line here. The dashboard reads STRATEGIES to build its dropdown."""
from __future__ import annotations
from .counter_trend import CounterTrendDipBuy
from .gap_pattern import GapPatternTypeA

_PLUGINS = [CounterTrendDipBuy, GapPatternTypeA]

STRATEGIES: dict[str, type] = {cls.display_name: cls for cls in _PLUGINS}

def get_strategy(display_name: str) -> type:
    return STRATEGIES[display_name]  # raises KeyError on unknown
