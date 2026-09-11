"""IV-rank entry gate: refuse a short put whose implied volatility is cheap
relative to that same ticker's own recent history.

The claim, measured 2026-08-04 on 42,226 gated ticker-days across 153 names
(2024-01-16 -> 2026-07-01, delta 0.40 / DTE 5-10 / TP 25%, scratchpad/
iv_signal.py + iv_analyse.py): mean return on collateral is **-11.8 bps**
across all entries, and crosses into positive territory only in the top decile
of IV rank.

    IV rank >=   n        mean bps   eqw bps   win %   assign %
    (none)       42226    -11.8      --        81.8    20.5
    0.80         10726     -6.9      -8.2      82.3    20.0
    0.85          7626     -0.3      -8.6      82.3    20.0
    0.88          5971     +8.8      +7.6      83.0    19.3
    0.90          4893    +13.7     +12.1      83.8    18.4
    0.95          2497    +19.7     +21.9      84.5    17.8

Monotone ramp, not a spike -- and positive in 2024, 2025 and 2026 separately
(+34.3 / +33.3 / +27.1 bps top decile vs rest). Win rate rises and assignment
rate falls together, so the mechanism is coherent: paid more, hit less.

This is Natenberg's relative volatility rank ([[Natenberg 03 Implied Volatility
Essentials]] p.72-74) -- "when relative volatility is high (8-10), focus on
selling premium" -- applied to the one variable the bot loads and then never
reads (`iv`, chain.py:43).

The variance risk premium (IV - trailing RV) was measured alongside it and is
the WEAKER signal (tops out at +4.8 bps, near-flat in 2025). Stacking both was
measured and does not beat rank alone: the edge lives in the top ~10%, so
tercile-crossing washes it out. One gate, deliberately.

Spec: this file + scratchpad/iv_signal.py
"""
import pandas as pd
import pytest

from src.engine_v2.options.iv_rank import IVHistory, iv_rank_ok
from src.engine_v2.options.portfolio import (PortfolioState, run_portfolio_wheel,
                                             step_one_day)
from src.engine_v2.options.wheel import WheelConfig


def _cfg(**kw):
    return WheelConfig(ticker="X", starting_capital=100_000.0, put_delta=0.40,
                       target_dte=7, take_profit_pct=0.25, **kw)


def _series(values, start="2024-01-01"):
    """A daily IV series; index is what the rank slices on."""
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


# --- the predicate ----------------------------------------------------------

def test_floor_unset_allows_everything():
    """Default OFF: every pre-existing result must stay byte-identical."""
    cfg = _cfg()
    assert iv_rank_ok(0.0, cfg)[0] is True
    assert iv_rank_ok(None, cfg)[0] is True


def test_rank_below_floor_is_refused():
    cfg = _cfg(min_iv_rank=0.90)
    ok, why = iv_rank_ok(0.42, cfg)
    assert ok is False
    assert why == "iv_rank_below_floor"


def test_rank_at_floor_is_allowed():
    """Boundary is inclusive -- a rank exactly at the floor is not 'below' it."""
    cfg = _cfg(min_iv_rank=0.90)
    assert iv_rank_ok(0.90, cfg)[0] is True


def test_unknown_rank_allows_but_is_named():
    """Unknown state always allows (the standing phase-1/2 convention), but the
    reason is distinct so callers can COUNT it rather than shrug at it -- the
    zombie_check failure class."""
    cfg = _cfg(min_iv_rank=0.90)
    ok, why = iv_rank_ok(None, cfg)
    assert ok is True
    assert why == "iv_rank_unknown"


# --- the history ------------------------------------------------------------

def test_rank_is_the_fraction_of_PRIOR_observations_below_today():
    """Denominator is the PRIOR observations, today excluded.

    This convention is load-bearing, not cosmetic: the 0.90 threshold was
    measured with `(w[:-1] < w[-1]).mean()` in scratchpad/iv_signal.py. Putting
    today back in the denominator shifts every rank by 1/n, and the measured
    floor would no longer mean what it meant when it was measured.

    Ten prior values, nine of them below today -> 9/10.
    """
    prior = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.99]
    hist = IVHistory({"AAA": _series(prior + [0.50])}, min_obs=10)
    idx = _series([0] * 11).index
    assert hist.rank("AAA", idx[-1]) == pytest.approx(0.9)


def test_rank_is_one_when_every_prior_observation_is_below():
    hist = IVHistory({"AAA": _series([0.10] * 10 + [0.50])}, min_obs=10)
    idx = _series([0] * 11).index
    assert hist.rank("AAA", idx[-1]) == pytest.approx(1.0)


def test_rank_ignores_observations_after_the_observation_date():
    """THE look-ahead guard. The series carries a huge future IV spike; ranking
    an early, quiet date must not see it. Slicing happens inside rank(), so a
    caller cannot forget to do it."""
    vals = [0.20] * 10 + [5.0] * 10          # calm, then a spike
    hist = IVHistory({"AAA": _series(vals)}, min_obs=5)
    idx = _series([0] * 20).index

    # As of the last calm day, every prior observation equals today -> rank 0.
    assert hist.rank("AAA", idx[9]) == pytest.approx(0.0)
    # Only once the spike is in the past does the rank move.
    assert hist.rank("AAA", idx[19]) > 0.4


def test_thin_history_is_unknown_not_zero():
    """Fewer than min_obs trailing points cannot be ranked. Returning 0.0 here
    would silently refuse every early entry under a floor."""
    hist = IVHistory({"AAA": _series([0.2, 0.3, 0.4])}, min_obs=252)
    idx = _series([0] * 3).index
    assert hist.rank("AAA", idx[-1]) is None


def test_absent_ticker_is_unknown():
    hist = IVHistory({"AAA": _series([0.2] * 300)}, min_obs=10)
    assert hist.rank("ZZZ", pd.Timestamp("2024-06-01")) is None
    assert hist.known("ZZZ") is False
    assert hist.known("AAA") is True


def test_rank_window_is_bounded():
    """Ancient history must age out, or a name that was once wild is permanently
    ranked low. Window is the trailing `window` observations, not all of them."""
    vals = [9.0] * 100 + [0.10] * 300 + [0.20]   # wild long ago, calm since
    hist = IVHistory({"AAA": _series(vals)}, window=252, min_obs=10)
    idx = _series([0] * len(vals)).index
    # The 100 ancient spikes are outside the 252-observation window, so today's
    # 0.20 ranks at the TOP of the calm regime rather than the bottom overall.
    assert hist.rank("AAA", idx[-1]) > 0.9


# --- wired into the entry loop ----------------------------------------------

D = pd.Timestamp("2026-07-21")
EXPIRY = D + pd.Timedelta(days=7)
COLS = ["date", "expiry", "dte", "strike", "right", "bid", "ask", "mid",
        "close", "delta", "iv", "underlying",
        "open_interest", "volume", "bid_size", "ask_size"]


def _chain():
    ch = pd.DataFrame([[D, EXPIRY, 7, 40.0, "P", 0.90, 1.00, 0.95, 0.95,
                        -0.40, 0.2, 41.0, 500.0, 100.0, 10.0, 10.0]],
                      columns=COLS)
    ch["date"] = pd.to_datetime(ch["date"])
    ch["expiry"] = pd.to_datetime(ch["expiry"])
    return ch


class M:
    """Market fake that HAS the iv_rank capability."""
    universe = ["DOW"]

    def __init__(self, rank=None):
        self._ch = _chain()
        self._rank = rank

    def chain(self, tk, d):
        return self._ch

    def spot(self, tk, d, fb):
        return 41.0

    def settle_price(self, tk, expiry):
        return 41.0

    def regime_row(self, tk, day):
        return None

    def eligible(self, tk, day):
        return True

    def iv_rank(self, tk, day):
        return self._rank


class MNoCapability(M):
    """Market fake WITHOUT iv_rank -- the ~20 pre-existing test fakes."""

    def __getattribute__(self, name):
        if name == "iv_rank":
            raise AttributeError(name)
        return super().__getattribute__(name)


def _pcfg(**kw):
    return WheelConfig(ticker="DOW", starting_capital=100_000.0, put_delta=0.40,
                       call_delta=0.50, target_dte=7, take_profit_pct=0.25,
                       call_min_strike="basis", **kw)


def _step(market, cfg):
    st = PortfolioState(cash=100_000.0, positions=[])
    return st, step_one_day(st, market, D, cfg, selector="plain", n_slots=1)


def test_gate_off_sells_regardless_of_rank():
    """Default OFF: a bottom-decile name still trades, byte-identical."""
    st, r = _step(M(rank=0.01), _pcfg())
    assert [t.action for t in r.trades] == ["SELL_PUT"]


def test_cheap_iv_is_vetoed_and_visible():
    st, r = _step(M(rank=0.42), _pcfg(min_iv_rank=0.90))
    assert [t.action for t in r.trades] == [], "sold a put at bottom-half IV rank"
    assert st.positions == []
    assert any(w[1] == "entry_gated_iv_rank" for w in r.warnings), \
        "a gated entry must be visible in warnings, not silent"


def test_rich_iv_still_sells():
    st, r = _step(M(rank=0.95), _pcfg(min_iv_rank=0.90))
    assert [t.action for t in r.trades] == ["SELL_PUT"]
    assert not any(w[1] == "entry_gated_iv_rank" for w in r.warnings)


def test_unknown_rank_allows_the_entry():
    """Standing convention: unknown state always allows."""
    st, r = _step(M(rank=None), _pcfg(min_iv_rank=0.90))
    assert [t.action for t in r.trades] == ["SELL_PUT"]


def test_market_without_the_capability_leaves_the_gate_inert():
    st, r = _step(MNoCapability(), _pcfg(min_iv_rank=0.90))
    assert [t.action for t in r.trades] == ["SELL_PUT"]


def test_warning_is_deduped_per_day_and_ticker():
    """The n_slots while-loop revisits gated tickers on every iteration."""
    st = PortfolioState(cash=100_000.0, positions=[])
    r = step_one_day(st, M(rank=0.10), D, _pcfg(min_iv_rank=0.90),
                     selector="plain", n_slots=3)
    assert sum(1 for w in r.warnings if w[1] == "entry_gated_iv_rank") == 1


def test_floor_without_a_history_refuses_to_run():
    """A gate believed to be on must never be structurally inert -- same stance
    as earnings_blackout and the A2 liquidity gate."""
    with pytest.raises(ValueError, match="needs an IV history"):
        run_portfolio_wheel({"SPY": _chain()}, _pcfg(min_iv_rank=0.90),
                            regime_states={"SPY": pd.DataFrame()})


# --- building the history from the chains -----------------------------------

def _multiday_chain(ivs, delta=-0.40):
    """One put per day at the target delta, plus a decoy at a wrong delta whose
    IV must NOT be the one recorded."""
    rows = []
    for i, iv in enumerate(ivs):
        day = D + pd.Timedelta(days=i)
        exp = day + pd.Timedelta(days=7)
        rows.append([day, exp, 7, 40.0, "P", 0.90, 1.00, 0.95, 0.95,
                     delta, iv, 41.0, 500.0, 100.0, 10.0, 10.0])
        rows.append([day, exp, 7, 20.0, "P", 0.05, 0.10, 0.075, 0.075,
                     -0.05, 9.99, 41.0, 500.0, 100.0, 10.0, 10.0])
    ch = pd.DataFrame(rows, columns=COLS)
    ch["date"] = pd.to_datetime(ch["date"])
    ch["expiry"] = pd.to_datetime(ch["expiry"])
    return ch


def test_from_chains_records_the_contract_the_engine_would_sell():
    """The ranked series must be the IV of the put actually sold, not the
    chain's cheapest or an average -- otherwise the measured 0.90 floor is
    ranking a different quantity than the one it was measured on."""
    hist = IVHistory.from_chains({"DOW": _multiday_chain([0.20, 0.30, 0.40])},
                                 _pcfg(), min_obs=2)
    assert hist.known("DOW")
    # Third day: both prior observations (0.20, 0.30) are below 0.40 -> 1.0.
    # If the 9.99 decoy were included, this would be 0.0.
    assert hist.rank("DOW", D + pd.Timedelta(days=2)) == pytest.approx(1.0)


def test_from_chains_covers_every_ticker():
    hist = IVHistory.from_chains(
        {"DOW": _multiday_chain([0.2, 0.3]), "KO": _multiday_chain([0.1, 0.5])},
        _pcfg(), min_obs=2)
    assert hist.known("DOW") and hist.known("KO")
    assert len(hist) == 2
