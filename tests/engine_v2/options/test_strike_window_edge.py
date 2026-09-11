"""B11: when the 12-strike survey window clips the delta ladder, the bottom
row wins select_contract's idxmin with a HIGHER |delta| than target -- a
riskier entry than configured, silently (measured: 3.24% of ticker-days, higher
delta 100% of the time, max err 0.196). The engines must say so:
`strike_window_edge` fires when the chosen put sits at the surveyed bottom
AND is riskier than target. The strict riskier-than clause keeps single-row
synthetic fixtures (delta exactly at target) quiet. Call side deliberately
unwarned: post-A4 an at-top call is usually the exchange's real extreme, not
a window artifact."""
import pandas as pd

from src.engine_v2.options.portfolio import PortfolioState, step_one_day
from src.engine_v2.options.wheel import WheelConfig

D = pd.Timestamp("2026-07-21")
COLS = ["date", "expiry", "dte", "strike", "right", "bid", "ask", "mid",
        "close", "delta", "iv", "underlying"]


def _clipped_chain(with_target_row=False):
    # window bottom 88 carries delta -0.34 (nearest to -0.30 -> selected);
    # deltas rise away from target as strikes climb. The true -0.30 strike
    # lies BELOW the window.
    rows = []
    for i, k in enumerate(range(88, 100)):
        delta = -(0.34 + 0.04 * i)
        rows.append([D, D + pd.Timedelta(days=11), 11, float(k), "P", 1.00,
                     1.10, 1.05, 1.05, delta, 0.2, 100.0])
    if with_target_row:
        rows[5][9] = -0.30   # a mid-window row exactly at target
    ch = pd.DataFrame(rows, columns=COLS)
    ch["date"] = pd.to_datetime(ch["date"])
    ch["expiry"] = pd.to_datetime(ch["expiry"])
    return ch


class M:
    universe = ["GDX"]

    def __init__(self, ch):
        self._ch = ch

    def chain(self, tk, d):
        return self._ch

    def spot(self, tk, d, fb):
        return 100.0

    def settle_price(self, tk, expiry):
        return 100.0

    def regime_row(self, tk, day):
        return None

    def eligible(self, tk, day):
        return True


def _cfg():
    return WheelConfig(ticker="GDX", starting_capital=100_000.0, put_delta=0.30,
                       call_delta=0.50, target_dte=11, take_profit_pct=0.60,
                       call_min_strike="basis")


def test_clipped_bottom_selection_warns():
    st = PortfolioState(cash=100_000.0, positions=[])
    r = step_one_day(st, M(_clipped_chain()), D, _cfg(), selector="plain",
                     n_slots=1)
    assert [t.action for t in r.trades] == ["SELL_PUT"]
    assert r.trades[0].contract.strike == 88.0
    assert any(w[1] == "strike_window_edge" and w[2] == "GDX"
               for w in r.warnings), \
        "B11: a clipped, riskier-than-target selection went unwarned"


def test_midwindow_target_selection_is_quiet():
    st = PortfolioState(cash=100_000.0, positions=[])
    r = step_one_day(st, M(_clipped_chain(with_target_row=True)), D, _cfg(),
                     selector="plain", n_slots=1)
    assert [t.action for t in r.trades] == ["SELL_PUT"]
    assert r.trades[0].contract.strike == 93.0
    assert not any(w[1] == "strike_window_edge" for w in r.warnings)


def test_bottom_at_exact_target_is_quiet():
    # single-strike synthetic chains sit at the bottom BY CONSTRUCTION with
    # delta == target; the strict riskier-than clause must keep them quiet
    ch = _clipped_chain()
    ch.loc[ch["strike"] == 88.0, "delta"] = -0.30
    st = PortfolioState(cash=100_000.0, positions=[])
    r = step_one_day(st, M(ch), D, _cfg(), selector="plain", n_slots=1)
    assert [t.action for t in r.trades] == ["SELL_PUT"]
    assert not any(w[1] == "strike_window_edge" for w in r.warnings)


# ---- per-engine wiring pins (A3b-skeptic lesson: every call site gets its
# own warn AND quiet assertion, or a per-site warn-always/never mutant
# survives the suite) ----

def _solo_cfg():
    return WheelConfig(ticker="GDX", starting_capital=100_000.0,
                       put_delta=0.30, call_delta=0.50, target_dte=11,
                       take_profit_pct=0.60, call_min_strike="basis")


def _router_states():
    st = pd.DataFrame([("2026-07-20", "chop", "normal", 0.5)],
                      columns=["date", "trend", "vol", "vol_pctile"]
                      ).set_index("date")
    st.index = pd.to_datetime(st.index)
    return st


def test_wheel_engine_clipped_entry_warns():
    from src.engine_v2.options.wheel import run_wheel
    r = run_wheel(_clipped_chain(), _solo_cfg())
    assert any(t.action == "SELL_PUT" for t in r.trades)
    assert any(w[1] == "strike_window_edge" and w[2] == "GDX"
               for w in (r.warnings or [])), \
        "B11: solo wheel engine left a clipped entry unwarned"


def test_wheel_engine_target_entry_quiet():
    from src.engine_v2.options.wheel import run_wheel
    r = run_wheel(_clipped_chain(with_target_row=True), _solo_cfg())
    assert any(t.action == "SELL_PUT" for t in r.trades)
    assert not any(w[1] == "strike_window_edge" for w in (r.warnings or []))


def test_router_engine_clipped_entry_warns():
    from src.engine_v2.options.regime_router import run_regime_router
    r = run_regime_router(_clipped_chain(), _solo_cfg(), _router_states())
    assert any(t.action == "SELL_PUT" for t in r.trades)
    assert any(w[1] == "strike_window_edge" and w[2] == "GDX"
               for w in (r.warnings or [])), \
        "B11: router engine left a clipped entry unwarned"


def test_router_engine_target_entry_quiet():
    from src.engine_v2.options.regime_router import run_regime_router
    r = run_regime_router(_clipped_chain(with_target_row=True), _solo_cfg(),
                          _router_states())
    assert any(t.action == "SELL_PUT" for t in r.trades)
    assert not any(w[1] == "strike_window_edge" for w in (r.warnings or []))


def test_spliced_held_row_below_bottom_does_not_mask_the_warning():
    """Skeptic F3: dropping the held_only filter from at_risky_window_edge
    survived both suites. With a held leg spliced BELOW the tradeable bottom
    (the live configuration -- drifted held strikes sit under the 12-strike
    window), an unfiltered min() lands on the held row, the selected strike
    no longer equals it, and the clip warning silently vanishes exactly
    where it matters most."""
    ch = _clipped_chain()
    held = ch.iloc[[0]].copy()
    held["strike"] = 80.0            # below the 88 tradeable bottom
    held["delta"] = -0.60
    held["held_only"] = True
    ch = pd.concat([ch, held], ignore_index=True)

    st = PortfolioState(cash=100_000.0, positions=[])
    r = step_one_day(st, M(ch), D, _cfg(), selector="plain", n_slots=1)
    assert [t.action for t in r.trades] == ["SELL_PUT"]
    assert r.trades[0].contract.strike == 88.0   # held row never a candidate
    assert any(w[1] == "strike_window_edge" for w in r.warnings), \
        "B11/F3: a spliced held row below the window masked the clip warning"
