"""Owner decision 2026-07-29, option B: mark the short book at the ASK.

A short option is a liability that can only be discharged by BUYING it back, and
you buy at the offer. Marking it at the midpoint credited the book half a spread
it can never capture. Recomputed at exitable prices, the live paper run's
reported +$31,752 became +$12,061 and 7 of the 25 accounts flipped from profit
to loss.

This deliberately breaks the byte-identical anchor for the portfolio engine.
Every backtest number produced before this change was made with mid marks and is
non-comparable with anything produced after it.
"""
import pandas as pd

from src.engine_v2.options.chain import Contract
from src.engine_v2.options.portfolio import PortfolioState, step_one_day
from src.engine_v2.options.wheel import WheelConfig

D = pd.Timestamp("2026-07-31")
EXP = pd.Timestamp("2026-08-21")
COLS = ["date", "expiry", "strike", "right", "dte", "delta",
        "bid", "ask", "mid", "underlying"]


def _chain(bid=0.90, ask=1.30, mid=1.10):
    return pd.DataFrame([{"date": D, "expiry": EXP, "strike": 30.0, "right": "P",
                          "dte": 21, "delta": -0.30, "bid": bid, "ask": ask,
                          "mid": mid, "underlying": 35.0}], columns=COLS)


class _Market:
    """One held ticker, no entry candidates (empty universe)."""
    universe = []

    def __init__(self, chain=None, spot=35.0):
        self._chain, self._spot = chain, spot

    def chain(self, t, d): return self._chain
    def spot(self, t, d, fb): return self._spot
    def settle_price(self, t, e): return self._spot
    def regime_row(self, t, d): return None
    def eligible(self, t, d): return True


def _cfg(take_profit=None):
    return WheelConfig(ticker="GDX", put_delta=0.30, call_delta=0.50, target_dte=21,
                       take_profit_pct=take_profit, starting_capital=100_000.0,
                       call_min_strike="basis")


def _regime_row():
    """A good-to-rent day: range-bound, calm, high vol percentile."""
    return pd.Series({"trend": "chop", "vol": "normal", "vol_pctile": 0.9,
                      "ma50_vs_200": 0.0, "fast_spread": 0.0})


def _position(contracts=2, credit=2.00, last_mid=2.00, last_ask=None):
    short = {"contract": Contract("GDX", EXP, 30.0, "P"), "contracts": contracts,
             "credit": credit, "last_mid": last_mid}
    if last_ask is not None:
        short["last_ask"] = last_ask
    return {"ticker": "GDX", "shares": 0, "phase": "PUT", "basis": None,
            "premium": credit * 100 * contracts, "campaign": 1,
            "last_spot": 35.0, "short": short}


def test_equity_marks_the_short_book_at_ask_not_mid():
    state = PortfolioState(cash=100_000.0, positions=[_position()], campaign=1)
    r = step_one_day(state, _Market(_chain(bid=0.90, ask=1.30, mid=1.10)), D,
                     _cfg(), selector="chop", n_slots=1)
    assert r.equity == 100_000.0 - 1.30 * 100 * 2      # ask, not the 1.10 mid


def test_the_ask_is_stored_on_the_leg_alongside_the_mid():
    """The mid stays recorded — it is still the fairest read of what the
    position is worth — but it is no longer what equity is computed from."""
    state = PortfolioState(cash=100_000.0, positions=[_position()], campaign=1)
    step_one_day(state, _Market(_chain()), D, _cfg(), selector="chop", n_slots=1)
    short = state.positions[0]["short"]
    assert short["last_ask"] == 1.30
    assert short["last_mid"] == 1.10


def test_an_unmarkable_day_carries_the_last_ask_not_the_last_mid():
    """No chain today (pull failure, or a strike outside the window that the
    held-leg merge could not rescue) -> carry the last exitable price."""
    state = PortfolioState(cash=100_000.0,
                           positions=[_position(last_mid=1.10, last_ask=1.30)],
                           campaign=1)
    r = step_one_day(state, _Market(chain=None), D, _cfg(), selector="chop", n_slots=1)
    assert r.equity == 100_000.0 - 1.30 * 100 * 2


def test_a_state_written_before_this_change_falls_back_to_its_mid():
    """The 25 live accounts hold 43 legs recorded under the old scheme, with no
    last_ask on them. They must mark at the stale mid for exactly one day, not
    crash the run and not silently mark at zero."""
    state = PortfolioState(cash=100_000.0,
                           positions=[_position(last_mid=1.10)],   # no last_ask
                           campaign=1)
    r = step_one_day(state, _Market(chain=None), D, _cfg(), selector="chop", n_slots=1)
    assert r.equity == 100_000.0 - 1.10 * 100 * 2


def test_a_leg_opened_today_is_marked_at_ask_on_its_first_day():
    """Entry records the leg from the same Mark it was sold on, so it must carry
    the ask from the start — otherwise day one of every campaign silently falls
    back to the mid and books half a spread of phantom gain at inception."""
    class _EntryMarket(_Market):
        universe = ["GDX"]
        def regime_row(self, t, d):
            return _regime_row()

    state = PortfolioState(cash=100_000.0, positions=[])
    r = step_one_day(state, _EntryMarket(_chain(bid=0.90, ask=1.30, mid=1.10)), D,
                     _cfg(), selector="chop", n_slots=1)
    assert [t.action for t in r.trades] == ["SELL_PUT"]
    short = state.positions[0]["short"]
    assert short["last_ask"] == 1.30 and short["last_mid"] == 1.10
    n = short["contracts"]
    assert r.equity == state.cash - 1.30 * 100 * n


def test_residual_settlement_buys_the_book_back_at_ask():
    """The batch finalizer force-closes whatever is still open on the last date.
    Settling that at the mid is the same unrealizable half-spread, just spent in
    one lump at the end of a backtest."""
    from src.engine_v2.options.portfolio import run_portfolio_wheel
    chain = pd.concat([_chain(bid=0.90, ask=1.30, mid=1.10).assign(date=d)
                       for d in pd.bdate_range(D, periods=3)], ignore_index=True)
    idx = pd.bdate_range(D - pd.Timedelta(days=10), periods=13)
    states = pd.DataFrame([_regime_row().to_dict()] * len(idx), index=idx)
    res = run_portfolio_wheel({"GDX": chain}, _cfg(), {"GDX": states},
                              selector="chop", n_slots=1, universe=["GDX"])
    assert res.residual_settled
    sell = res.trades[-1]
    assert sell.action == "SELL_PUT"
    # cash after the sale, less the cost of buying the book back at the 1.30 ask
    assert res.final_cash == sell.cash_after - 1.30 * 100 * sell.contracts


def test_take_profit_still_measures_against_the_ask():
    """Unchanged behaviour, pinned: the exit test was always correct — it is the
    equity mark that was wrong. Credit 2.00, 60% TP -> trigger at ask <= 0.80."""
    state = PortfolioState(cash=100_000.0, positions=[_position()], campaign=1)
    r = step_one_day(state, _Market(_chain(bid=0.50, ask=0.75, mid=0.62)), D,
                     _cfg(take_profit=0.60), selector="chop", n_slots=1)
    assert [t.action for t in r.trades] == ["CLOSE_PUT"]
    assert r.trades[0].price_per_contract == 0.75
