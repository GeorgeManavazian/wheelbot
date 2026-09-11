"""E1: pin the equity mark EXACTLY for a covered position (shares AND short).

Equity mutations #1-#3 (assigned-share book value x0.99, x0.5, and -$1 whenever
shares are held) survived all 630 tests because nothing asserted the equity
NUMBER -- only orderings and inequalities. This is the exact-arithmetic pin:
equity == cash + shares*spot - ask*mult*contracts, to the cent, no tolerance
wide enough to hide a mutation."""
import pandas as pd
import pytest

from src.engine_v2.options.portfolio import PortfolioState, step_one_day
from src.engine_v2.options.wheel import WheelConfig

D = pd.Timestamp("2026-07-21")
COLS = ["date", "expiry", "dte", "strike", "right", "bid", "ask", "mid",
        "close", "delta", "iv", "underlying"]


class M:
    universe = ["GDX"]

    def __init__(self, ch, spot):
        self._ch, self._spot = ch, spot

    def chain(self, tk, d):
        return self._ch

    def spot(self, tk, d, fb):
        return self._spot

    def settle_price(self, tk, expiry):
        return self._spot

    def regime_row(self, tk, day):
        return None

    def eligible(self, tk, day):
        return True


def _chain(rows):
    ch = pd.DataFrame(rows, columns=COLS)
    ch["date"] = pd.to_datetime(ch["date"])
    ch["expiry"] = pd.to_datetime(ch["expiry"])
    return ch


def test_covered_position_equity_is_exact_to_the_cent():
    """200 shares + 2 short calls, every input chosen so the expected number
    is computable by hand: cash after the sale, shares at spot, liability at
    the ASK (owner decision 2026-07-29)."""
    spot, strike, bid, ask, comm = 65.0, 70.0, 3.00, 3.10, 0.65
    ch = _chain([[D, D + pd.Timedelta(days=11), 11, strike, "C", bid, ask,
                  (bid + ask) / 2, (bid + ask) / 2, 0.50, 0.2, spot]])
    st = PortfolioState(cash=10_000.0, positions=[{
        "ticker": "GDX", "shares": 200, "phase": "CALL", "basis": 60.0,
        "premium": 0.0, "campaign": 1, "last_spot": spot, "short": None}])
    cfg = WheelConfig(ticker="GDX", starting_capital=100_000.0, put_delta=0.30,
                      call_delta=0.50, target_dte=11, take_profit_pct=0.60,
                      commission_per_contract=comm, call_min_strike="basis")
    r = step_one_day(st, M(ch, spot), D, cfg, selector="plain", n_slots=1)

    assert [t.action for t in r.trades] == ["SELL_CALL"]
    n = r.trades[0].contracts
    assert n == 2                       # 200 shares -> 2 contracts
    cash = 10_000.0 + bid * 100 * n - comm * n
    expected = cash + 200 * spot - ask * 100 * n
    assert r.equity == pytest.approx(expected, abs=1e-9), (
        f"E1: covered-position equity drifted from the hand computation "
        f"({r.equity} != {expected}) -- shares must book at spot x count, "
        f"the short at the ask, nothing else")
    assert st.cash == pytest.approx(cash, abs=1e-9)


def test_bare_shares_equity_is_exact_to_the_penny():
    """Shares held, NO short (the covered call is refused by an empty chain):
    equity == cash + shares*spot exactly. Kills the -$1-when-shares-held
    class on the uncovered branch too."""
    spot = 65.0
    ch = _chain([])   # nothing listed today
    st = PortfolioState(cash=10_000.0, positions=[{
        "ticker": "GDX", "shares": 300, "phase": "CALL", "basis": 60.0,
        "premium": 0.0, "campaign": 1, "last_spot": spot, "short": None}])
    cfg = WheelConfig(ticker="GDX", starting_capital=100_000.0, put_delta=0.30,
                      call_delta=0.50, target_dte=11, take_profit_pct=0.60,
                      commission_per_contract=0.65, call_min_strike="basis")
    r = step_one_day(st, M(ch, spot), D, cfg, selector="plain", n_slots=1)
    assert r.trades == []
    assert r.equity == pytest.approx(10_000.0 + 300 * spot, abs=1e-9), \
        "E1: bare-shares equity drifted with no trade at all"
