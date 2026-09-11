"""A17: the state schema must be able to EXPRESS a partial close -- "k of n
contracts bought back, n-k still short" -- so that a realistic fill model has
somewhere to put its answer. Measured need: in the hour the TP limit was first
touched, 55.4% of hours traded fewer contracts than the bot's largest real
fill (181). The fill MODEL stays deferred (owner ruling); these tests exercise
the CAPACITY by injecting a partial FillDecision directly. The instant-fill
default path must stay byte-identical (third test + the A17 fingerprints)."""
import types

import pandas as pd

from src.engine_v2.options.chain import Contract
from src.engine_v2.options.portfolio import PortfolioState, step_one_day
from src.engine_v2.options.wheel import WheelConfig

D = pd.Timestamp("2026-07-21")


class M:
    universe = ["GDX"]

    def chain(self, tk, d):
        return None

    def spot(self, tk, d, fb):
        return 38.0                    # below the 40 strike -> assignment

    def settle_price(self, tk, expiry):
        return 38.0

    def regime_row(self, tk, day):
        return None

    def eligible(self, tk, day):
        return True


def _state(n=10, expiry=D):
    c = Contract("GDX", pd.Timestamp(expiry), 40.0, "P")
    return PortfolioState(cash=100_000.0, positions=[{
        "ticker": "GDX", "shares": 0, "phase": "PUT", "basis": None,
        "premium": 1000.0, "campaign": 1, "last_spot": 41.0,
        "short": {"contract": c, "contracts": n, "credit": 1.00,
                  "last_mid": 1.00}}])


def _cfg():
    return WheelConfig(ticker="GDX", starting_capital=100_000.0, put_delta=0.30,
                       call_delta=0.50, target_dte=11, take_profit_pct=0.60,
                       call_min_strike="basis")


def _partial(k, price=0.30):
    # duck-typed FillDecision: a model that filled k contracts of the leg
    return types.SimpleNamespace(filled=True, price=price,
                                 cost=price * 100 * k + 0.65 * k,
                                 stamp=D, via="quote", filled_contracts=k)


def test_partial_close_leaves_remainder_short(monkeypatch):
    from src.engine_v2.options import portfolio
    st = _state(n=10, expiry=D + pd.Timedelta(days=5))
    monkeypatch.setattr(portfolio, "try_take_profit", lambda **kw: _partial(4))
    r = step_one_day(st, M(), D, _cfg(), selector="chop", n_slots=1)
    close = [t for t in r.trades if t.action == "CLOSE_PUT"]
    assert len(close) == 1 and close[0].contracts == 4, \
        "A17: a 4-of-10 fill must book 4 contracts, not the whole leg"
    short = st.positions[0]["short"]
    assert short is not None and short["contracts"] == 6, \
        "A17: 6 contracts must remain short after a 4-of-10 fill"


def test_partial_close_then_same_day_expiry_settles_remainder(monkeypatch):
    """I3: `n` is bound once per position per day; a partial TP followed by
    same-day expiry must settle the REMAINING size, not the pre-fill size."""
    from src.engine_v2.options import portfolio
    st = _state(n=10, expiry=D)                    # expires today
    monkeypatch.setattr(portfolio, "try_take_profit", lambda **kw: _partial(4))
    r = step_one_day(st, M(), D, _cfg(), selector="chop", n_slots=1)
    assigned = [t for t in r.trades if t.action == "ASSIGNED"]
    assert assigned and assigned[0].contracts == 6, \
        "A17/I3: same-day expiry must settle the 6 remaining contracts"
    assert st.positions[0]["shares"] == 600


def test_full_fill_still_compacts_exactly_as_today(monkeypatch):
    # the instant-fill default: filled_contracts == contracts -> short None,
    # emptied PUT slot compacted -- today's behavior, unchanged
    from src.engine_v2.options import portfolio
    st = _state(n=10, expiry=D + pd.Timedelta(days=5))
    monkeypatch.setattr(portfolio, "try_take_profit", lambda **kw: _partial(10))
    r = step_one_day(st, M(), D, _cfg(), selector="chop", n_slots=1)
    assert st.positions == []
    assert [t.action for t in r.trades] == ["CLOSE_PUT"]


def test_partial_stamps_original_size(monkeypatch):
    """Skeptic F5: the first partial must record opened_contracts, or the
    stored "6 remain" is indistinguishable from a 6-lot leg."""
    from src.engine_v2.options import portfolio
    st = _state(n=10, expiry=D + pd.Timedelta(days=5))
    monkeypatch.setattr(portfolio, "try_take_profit", lambda **kw: _partial(4))
    step_one_day(st, M(), D, _cfg(), selector="chop", n_slots=1)
    short = st.positions[0]["short"]
    assert short["opened_contracts"] == 10 and short["contracts"] == 6


def test_zero_contract_fill_is_refused_loudly():
    """Skeptic F4: filled=True with filled_contracts=0 books a ghost trade
    and bleeds cash. The helper must raise, not bleed."""
    import pytest
    from src.engine_v2.options.portfolio import close_short_fill
    st = _state(n=10, expiry=D + pd.Timedelta(days=5))
    ghost = types.SimpleNamespace(filled=True, price=0.30, cost=50.0,
                                  stamp=D, via="quote", filled_contracts=0)
    with pytest.raises(ValueError):
        close_short_fill(st.positions[0], ghost, 100_000.0, [])
