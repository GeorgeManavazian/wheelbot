"""A10: corporate actions -- detect and refuse (owner scope: never trade
through one silently; no modeling).

The $42,750 hole: Schwab price history is split-adjusted
retroactively, stored strike/shares/basis/last_spot are not. On the morning
after a 2:1 split the engine booked ASSIGNED 900 @ $100 against a $52.50
adjusted close, zero warnings. The discriminator: a split RESTATES history -- the fresh
series' close at prev_d no longer equals the stored last_spot, off by
exactly the ratio, with no market noise; a real crash never rewrites
yesterday. Capability-gated like A15: only a market providing prior_close()
(LiveMarket) can fire the guard; BatchMarket never grows it, so batch stays
byte-identical by construction.

Provisional defaults (owner asleep, veto cheap, bot paused): confirm band
ratio <=0.80 / >=1.25; gap backstop 25%; freeze clearing manual-only;
suspects print but only freezes email."""
import pandas as pd
import pytest

from src.engine_v2.options.portfolio import PortfolioState, step_one_day
from src.engine_v2.options.wheel import WheelConfig

D2 = pd.Timestamp("2026-07-21")            # today
D1 = pd.Timestamp("2026-07-20")            # prev stepped day
COLS = ["date", "expiry", "dte", "strike", "right", "bid", "ask", "mid",
        "close", "delta", "iv", "underlying"]


class Liveish:
    """Market WITH the prior_close capability (the LiveMarket shape)."""
    universe = ["XYZ"]

    def __init__(self, spot_today, prior, chain=None):
        self._spot, self._prior, self._ch = spot_today, prior, chain

    def chain(self, tk, d):
        return self._ch

    def spot(self, tk, d, fb):
        return self._spot

    def settle_price(self, tk, expiry):
        return self._spot

    def prior_close(self, tk, d):
        return self._prior

    def regime_row(self, tk, day):
        return None

    def eligible(self, tk, day):
        return True


def _held_put(last_spot=104.0, strike=100.0, expiry=D2, contracts=9):
    from src.engine_v2.options.select import Contract
    return {"ticker": "XYZ", "shares": 0, "phase": "PUT", "basis": None,
            "premium": 0.0, "campaign": 1, "last_spot": last_spot,
            "short": {"contract": Contract("XYZ", expiry, strike, "P"),
                      "contracts": contracts, "credit": 2.00, "last_mid": 2.0}}


def _cfg():
    return WheelConfig(ticker="XYZ", starting_capital=100_000.0,
                       put_delta=0.30, call_delta=0.50, target_dte=11,
                       take_profit_pct=0.60, commission_per_contract=0.0,
                       call_min_strike="basis")


def _step(market, pos, prev_d=D1):
    st = PortfolioState(cash=10_000.0, positions=[pos], prev_d=prev_d)
    r = step_one_day(st, market, D2, _cfg(), selector="plain", n_slots=1)
    return st, r


def test_restatement_band_edges_pin_the_owner_defaults():
    """Group A boundary sweep survivor M8 (2026-08-02): widening the band to
    0.5/2.0 survived all 896 tests, because every drill uses 2:1 or 1:4
    splits that freeze under ANY band. A 3:2 split (ratio 0.667) or a 5:4
    (0.8) is a common corporate action, and an unfrozen one is the
    $42,750 phantom-assignment class. Pin just inside and just outside the
    owner's 0.80/1.25 band (not the exact float edges); a band-widening
    mutant must fail the outside pins. Far expiry: nothing settles, so a
    non-freeze day books no trades and the freeze flag is the only signal."""
    far = pd.Timestamp("2026-12-18")
    for ratio, must_freeze in [(0.79, True), (0.81, False),
                               (1.24, False), (1.26, True)]:
        pos = _held_put(expiry=far)
        st, r = _step(Liveish(spot_today=104.0 * ratio, prior=104.0 * ratio),
                      pos)
        assert bool(pos.get("ca_frozen")) == must_freeze, \
            (f"A10 band edge: ratio {ratio} expected "
             f"{'freeze' if must_freeze else 'no freeze'}")
        assert any(w[1] == "ca_confirmed_frozen" for w in r.warnings) \
            == must_freeze


def test_split_restatement_freezes_instead_of_phantom_assignment():
    """The $42,750 hole verbatim: 2:1 split, adjusted close 52.50, stored
    last_spot 104. Old engine: ASSIGNED 900 @ 100, $42,750 fabricated. The
    fresh series' prev_d close reads 52.00 = stored 104 x 0.50 -> freeze."""
    pos = _held_put()
    st, r = _step(Liveish(spot_today=52.50, prior=52.00), pos)
    assert [t.action for t in r.trades] == [], \
        "A10: engine traded through a corporate action"
    assert any(w[1] == "ca_confirmed_frozen" for w in r.warnings)
    assert pos.get("ca_frozen"), "the freeze must be sticky on the position"
    assert pos["last_spot"] == 104.0, \
        "A10: last_spot updated on freeze -- the detector erased its own evidence"


def test_real_crash_defers_one_session_then_settles_late():
    """GL-class one-day -50% crash, NO restatement (prior_close == stored).
    Day 1: settlement deferred, leg open, no freeze. Day 2: ratio clean, gap
    tiny -> settles via the existing late-expiry path."""
    pos = _held_put()
    st, r = _step(Liveish(spot_today=52.50, prior=104.0), pos)
    assert [t.action for t in r.trades] == []
    assert any(w[1] == "ca_suspect_settlement_deferred" for w in r.warnings)
    assert not pos.get("ca_frozen")
    assert pos.get("ca_watch"), "a suspect day must open the re-check watch"

    st.prev_d = D2
    day3 = D2 + pd.Timedelta(days=1)
    r2 = step_one_day(st, LiveishNext(52.60, {str(D2.date()): 52.50,
                                              str(D1.date()): 104.0}),
                      day3, _cfg(), selector="plain", n_slots=1)
    assert [t.action for t in r2.trades] == ["ASSIGNED"], \
        "A10: a real crash must cost exactly one deferred session"
    assert any(w[1] == "expiry_resolved_late" for w in r2.warnings)


class LiveishNext(Liveish):
    """prior_close keyed by date for multi-day sequences."""

    def __init__(self, spot_today, prior_by_date, chain=None):
        super().__init__(spot_today, None, chain)
        self._pbd = prior_by_date

    def prior_close(self, tk, d):
        return self._pbd.get(str(pd.Timestamp(d).date()))


def test_reverse_split_restatement_freezes():
    pos = _held_put(last_spot=5.0, strike=4.5)
    st, r = _step(Liveish(spot_today=50.0, prior=50.0), pos)   # x10 restated
    assert [t.action for t in r.trades] == []
    assert any(w[1] == "ca_confirmed_frozen" for w in r.warnings)


def test_frozen_position_skips_tp_and_covered_call_every_run():
    """A frozen leg must do NOTHING -- no TP (even at a juicy mark), no
    covered call, no settlement -- until a human clears it."""
    from src.engine_v2.options.select import Contract
    exp = D2 + pd.Timedelta(days=7)
    ch = pd.DataFrame([[D2, exp, 7, 100.0, "P", 0.01, 0.05, 0.03, 0.03,
                        -0.05, 0.2, 104.0]], columns=COLS)
    ch["date"] = pd.to_datetime(ch["date"])
    ch["expiry"] = pd.to_datetime(ch["expiry"])
    pos = _held_put(expiry=exp)
    pos["ca_frozen"] = {"date": "2026-07-20", "stored_spot": 104.0,
                        "ratio": 0.5}
    st, r = _step(Liveish(spot_today=104.0, prior=104.0, chain=ch), pos)
    assert [t.action for t in r.trades] == [], \
        "A10: a frozen position traded"
    assert any(w[1] == "ca_confirmed_frozen" for w in r.warnings), \
        "the freeze must stay loud every run until cleared"


def test_frozen_position_keeps_its_prefreeze_mark():
    """A10f (boundary sweep, 2026-08-02): the frozen skip covered TP,
    settlement and the covered call -- but section 3 still re-marked the
    frozen leg from the possibly-restated chain every step, mutating
    state.json in unknown units mid-restatement (2.10 -> 4.40 in the sweep
    repro), feeding snapshots/dashboard, and setting the price the batch
    residual finalizer would buy back at. A frozen leg carries its
    pre-freeze mark until a human clears it; equity uses that carried mark."""
    from src.engine_v2.options.select import Contract
    exp = D2 + pd.Timedelta(days=7)
    # chain quotes the (restated-unit) leg far from the stored 2.00/2.10 mark
    ch = pd.DataFrame([[D2, exp, 7, 100.0, "P", 4.30, 4.40, 4.35, 4.35,
                        -0.60, 0.2, 52.0]], columns=COLS)
    ch["date"] = pd.to_datetime(ch["date"])
    ch["expiry"] = pd.to_datetime(ch["expiry"])
    pos = _held_put(expiry=exp)
    pos["short"]["last_mid"] = 2.00
    pos["short"]["last_ask"] = 2.10
    pos["ca_frozen"] = {"date": "2026-07-20", "stored_spot": 104.0,
                        "ratio": 0.5}
    st, r = _step(Liveish(spot_today=52.0, prior=52.0, chain=ch), pos)
    assert pos["short"]["last_ask"] == 2.10 and pos["short"]["last_mid"] == 2.00, \
        "A10f: a frozen leg was re-marked from the restated chain"
    # equity carries the pre-freeze ask: cash + 0 shares - 2.10*100*9
    assert r.equity == pytest.approx(10_000.0 - 2.10 * 100 * 9)


def test_watch_catches_one_day_late_restatement():
    """Split morning, Schwab history lags: day 1 looks like a crash (gap
    fires, restatement clean) -> watch. Day 2 the adjustment lands: the
    WATCHED date's close now reads half the spot stored THEN -> freeze."""
    pos = _held_put(expiry=D2 + pd.Timedelta(days=30))
    st, r = _step(Liveish(spot_today=52.50, prior=104.0), pos)
    assert pos.get("ca_watch") and not pos.get("ca_frozen")

    st.prev_d = D2
    day3 = D2 + pd.Timedelta(days=1)
    r2 = step_one_day(st, LiveishNext(52.60, {str(D1.date()): 52.0,
                                              str(D2.date()): 52.5}),
                      day3, _cfg(), selector="plain", n_slots=1)
    assert pos.get("ca_frozen"), \
        "A10: a one-day-late restatement slipped past the watch"
    assert any(w[1] == "ca_confirmed_frozen" for w in r2.warnings)


def test_watch_expires_quietly_after_bound():
    pos = _held_put(expiry=D2 + pd.Timedelta(days=60))
    st, r = _step(Liveish(spot_today=52.50, prior=104.0), pos)
    assert pos.get("ca_watch")
    st.prev_d = D2
    late = D2 + pd.Timedelta(days=10)          # past the 7-calendar-day bound
    step_one_day(st, LiveishNext(52.60, {str(D1.date()): 104.0,
                                         str(D2.date()): 52.5}),
                 late, _cfg(), selector="plain", n_slots=1)
    assert not pos.get("ca_watch"), "an expired watch must clear itself"
    assert not pos.get("ca_frozen")


def test_batch_market_never_grows_the_capability():
    """A15-symmetric pin: the batch engine can never restate against itself;
    if BatchMarket grows prior_close, the guard fires on backtests and every
    golden breaks silently the other way."""
    from src.engine_v2.options.market import BatchMarket
    assert not hasattr(BatchMarket, "prior_close")


def test_capability_less_market_is_untouched():
    """No prior_close -> no guard, byte-identical behavior: deep-ITM expiry
    still assigns exactly as before (the whole existing suite is the broader
    pin; this is the local one)."""
    class Plain:                       # NO prior_close anywhere in the MRO
        universe = ["XYZ"]

        def chain(self, tk, d):
            return None

        def spot(self, tk, d, fb):
            return 52.50

        def settle_price(self, tk, expiry):
            return 52.50

        def regime_row(self, tk, day):
            return None

        def eligible(self, tk, day):
            return True

    assert not hasattr(Plain, "prior_close")
    pos = _held_put()
    st, r = _step(Plain(), pos)
    assert [t.action for t in r.trades] == ["ASSIGNED"]
