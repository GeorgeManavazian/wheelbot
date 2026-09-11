import pandas as pd
from src.engine_v2.options.chain import Contract
from src.engine_v2.options.wheel import WheelConfig
from src.engine_v2.options.portfolio import PortfolioState, step_one_day


class _StubMarket:
    """Minimal Market: one held ticker, no chain today (forces the expiry path),
    a spot above the put strike (so the put expires worthless), empty universe
    (no new entries)."""
    universe = []
    def __init__(self, spot_val): self._spot = spot_val
    def chain(self, t, d): return None
    def spot(self, t, d, fb): return self._spot
    def settle_price(self, t, e): return self._spot
    def regime_row(self, t, d): return None
    def eligible(self, t, d): return True


def _cfg():
    return WheelConfig(ticker="GDX", put_delta=0.20, call_delta=0.50,
                       target_dte=7, take_profit_pct=0.50, starting_capital=100_000.0,
                       call_min_strike="basis")


def test_step_expires_put_worthless_and_drops_campaign():
    d = pd.Timestamp("2021-01-15")   # == expiry
    put = Contract("GDX", d, 30.0, "P")
    pos = {"ticker": "GDX", "shares": 0, "phase": "PUT", "basis": None,
           "premium": 99.0, "campaign": 1, "last_spot": 35.0,
           "short": {"contract": put, "contracts": 1, "credit": 1.0, "last_mid": 1.0}}
    state = PortfolioState(cash=100_099.0, positions=[pos], campaign=1)
    r = step_one_day(state, _StubMarket(spot_val=35.0), d, _cfg(),
                     selector="chop", n_slots=1)
    # spot 35 > strike 30 -> PUT_EXPIRED, no cash change on expiry
    actions = [t.action for t in r.trades]
    assert "PUT_EXPIRED" in actions
    assert state.positions == []            # campaign went flat -> dropped
    assert state.cash == 100_099.0
    assert state.days_flat == 1             # ended the day flat
    assert isinstance(r.equity, float)
