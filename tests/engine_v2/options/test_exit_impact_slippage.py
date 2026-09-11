"""B-exit: SELL_CALL and CLOSE_PUT cannot refuse an oversized fill -- a
covered call is exempt from every liquidity gate (refusing one leaves shares
naked) and a put buyback closes a position already held, not a new one to
decline. `liquidity_ok`/`liquidity_size_cap` only ever look at the entry
path, so those two legs were priced at the clean quote no matter how far past
the day's open interest or volume the order sat.

Measured on the promoted config's own fill log (liq-junk-floor, 2026-08-17)
found a covered call sold at 110% of that day's open interest, another at
2,100% of that day's volume, 16 of 98 call fills on a strike with zero volume
that day. `exit_impact_slippage` prices that: reuses the SAME ratio already
promoted for entry sizing (`liq_max_pct_of_open_interest`/`liq_max_pct_of_volume`)
rather than a new fitted threshold, and charges one extra bid-ask spread per
full multiple past that ratio, capped at `exit_impact_max_spreads` so a
zero-volume day prices as expensive rather than undefined.

These tests pin the function directly (`select.exit_impact_slippage`) and
the two `portfolio.step_one_day` call sites that consume it (covered-call
write and put buyback), matching the fixture/style of
test_liquidity_size_cap.py."""
import pandas as pd

from src.engine_v2.options.chain import Contract
from src.engine_v2.options.select import exit_impact_slippage
from src.engine_v2.options.portfolio import PortfolioState, step_one_day
from src.engine_v2.options.wheel import WheelConfig

D = pd.Timestamp("2026-07-21")
EXPIRY = D + pd.Timedelta(days=11)
COLS = ["date", "expiry", "dte", "strike", "right", "bid", "ask", "mid",
        "close", "delta", "iv", "underlying",
        "open_interest", "volume", "bid_size", "ask_size"]


def _chain(bid=0.90, ask=1.30, oi=200.0, vol=1000.0, strike=40.0, right="C"):
    ch = pd.DataFrame([[D, EXPIRY, 11, strike, right, bid, ask,
                        (bid + ask) / 2, (bid + ask) / 2, -0.30, 0.2,
                        strike + 1.0, oi, vol, 10.0, 10.0]], columns=COLS)
    ch["date"] = pd.to_datetime(ch["date"])
    ch["expiry"] = pd.to_datetime(ch["expiry"])
    return ch


def _contract(strike=40.0, right="C"):
    return Contract("DOW", EXPIRY, strike, right)


# --- pure function: exit_impact_slippage -----------------------------------

def test_off_by_default_is_zero():
    """No cap knob set -> 0.0, byte-identical to every prior run, regardless
    of how oversized the order is."""
    ch = _chain(oi=10, vol=5)
    assert exit_impact_slippage(ch, D, _contract(), 100, _cfg_obj()) == 0.0


def test_zero_when_both_ratio_knobs_are_none_even_with_a_cap_set():
    """A cap with nothing to ratio against is a no-op, not a crash."""
    cfg = _cfg_obj(exit_impact_max_spreads=1.0)
    assert exit_impact_slippage(_chain(), D, _contract(), 100, cfg) == 0.0


def test_order_inside_the_ratio_costs_nothing():
    """15 contracts against oi=200/vol=1000 at a 10%/25% ratio: the caps are
    20 (10% of 200) and 250 (25% of 1000) -- 15 sits under both. Zero charge."""
    ch = _chain(oi=200, vol=1000, bid=0.90, ask=1.30)
    cfg = _cfg_obj(exit_impact_max_spreads=1.0,
                    liq_max_pct_of_open_interest=0.10,
                    liq_max_pct_of_volume=0.25)
    assert exit_impact_slippage(ch, D, _contract(), 15, cfg) == 0.0


def test_order_past_the_ratio_costs_one_spread_per_full_multiple():
    """10% of oi=100 is 10 contracts; an order of 30 sits at 3x that ratio ->
    excess = 3 - 1 = 2 full spreads, spread = ask-bid = 0.40."""
    ch = _chain(oi=100, vol=100_000, bid=0.90, ask=1.30)
    cfg = _cfg_obj(exit_impact_max_spreads=5.0,
                    liq_max_pct_of_open_interest=0.10,
                    liq_max_pct_of_volume=0.25)
    got = exit_impact_slippage(ch, D, _contract(), 30, cfg)
    assert got == pytest_approx(2.0 * 0.40)


def test_the_tighter_of_oi_and_volume_ratios_wins():
    """oi=100k makes the OI leg slack; vol=40 at a 25% ratio caps at 10
    contracts, and an order of 30 sits at 3x that -> the volume leg binds,
    same 2-spread charge as the previous test even though OI would say 0."""
    ch = _chain(oi=100_000, vol=40, bid=0.90, ask=1.30)
    cfg = _cfg_obj(exit_impact_max_spreads=5.0,
                    liq_max_pct_of_open_interest=0.10,
                    liq_max_pct_of_volume=0.25)
    got = exit_impact_slippage(ch, D, _contract(), 30, cfg)
    assert got == pytest_approx(2.0 * 0.40)


def test_charge_is_capped_at_exit_impact_max_spreads():
    """A 2,100%-of-volume order (the measured case) would compute an
    enormous excess -- the charge must stop at the declared cap, not scale
    with how absurd the fill is."""
    ch = _chain(oi=100_000, vol=1, bid=0.90, ask=1.30)  # 30/1 >> any ratio
    cfg = _cfg_obj(exit_impact_max_spreads=1.0,
                    liq_max_pct_of_open_interest=0.10,
                    liq_max_pct_of_volume=0.25)
    got = exit_impact_slippage(ch, D, _contract(), 30, cfg)
    assert got == pytest_approx(1.0 * 0.40), \
        "charge must be capped at exactly one spread, however far past the ratio"


def test_zero_volume_day_is_priced_as_expensive_not_infinite_or_free():
    """A strike with zero volume that day (16 of 98 real fills in the log)
    must still hit the cap, not divide by zero into inf-as-a-float, and
    must NOT be silently treated as costless."""
    ch = _chain(oi=100_000, vol=0.0, bid=0.90, ask=1.30)
    cfg = _cfg_obj(exit_impact_max_spreads=2.0,
                    liq_max_pct_of_open_interest=0.10,
                    liq_max_pct_of_volume=0.25)
    got = exit_impact_slippage(ch, D, _contract(), 5, cfg)
    assert got == pytest_approx(2.0 * 0.40)


def test_unmeasurable_field_is_priced_as_expensive_same_stance_as_the_gate():
    """NaN open interest -> same stance as liquidity_ok: cannot be measured
    means treated as maximally bad, capped at the declared ceiling."""
    ch = _chain(oi=200, vol=1000, bid=0.90, ask=1.30)
    ch["open_interest"] = float("nan")
    cfg = _cfg_obj(exit_impact_max_spreads=3.0,
                    liq_max_pct_of_open_interest=0.10)
    got = exit_impact_slippage(ch, D, _contract(), 5, cfg)
    assert got == pytest_approx(3.0 * 0.40)


def test_no_row_for_the_contract_is_zero_not_a_crash():
    ch = _chain(strike=40.0)
    cfg = _cfg_obj(exit_impact_max_spreads=1.0, liq_max_pct_of_open_interest=0.10)
    assert exit_impact_slippage(ch, D, _contract(strike=999.0), 5, cfg) == 0.0


def test_zero_width_market_is_zero_cost():
    """bid == ask -> spread is 0 -> nothing to charge regardless of size."""
    ch = _chain(bid=1.00, ask=1.00, oi=10, vol=10)
    cfg = _cfg_obj(exit_impact_max_spreads=5.0, liq_max_pct_of_open_interest=0.10)
    assert exit_impact_slippage(ch, D, _contract(), 100, cfg) == 0.0


def test_held_only_rows_are_excluded_same_as_liquidity_ok():
    """Mirrors liquidity_ok's row lookup: a held_only-flagged snapshot row
    (spliced in for a position missing from that day's real pull) must not
    be read as the real market."""
    ch = _chain(oi=5, vol=5, bid=0.10, ask=0.90)
    ch["held_only"] = [True]
    cfg = _cfg_obj(exit_impact_max_spreads=5.0, liq_max_pct_of_open_interest=0.10)
    assert exit_impact_slippage(ch, D, _contract(), 100, cfg) == 0.0


# --- pytest.approx without an extra import line at the top -----------------
import pytest
pytest_approx = pytest.approx


def _cfg_obj(**kw):
    return WheelConfig(ticker="DOW", starting_capital=100_000.0, put_delta=0.30,
                       call_delta=0.50, target_dte=11, take_profit_pct=0.60,
                       call_min_strike="basis", **kw)


# --- wired into the engine: portfolio.step_one_day --------------------------

class M:
    universe = ["DOW"]

    def __init__(self, ch):
        self._ch = ch

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


def _step(ch, cfg, capital=100_000.0, n_slots=1, positions=None):
    st = PortfolioState(cash=capital, positions=positions or [])
    r = step_one_day(st, M(ch), D, cfg, selector="plain", n_slots=n_slots)
    return st, r


def _held_pos(shares, basis=38.0, last_spot=41.0):
    return {"ticker": "DOW", "shares": shares, "phase": "CALL", "basis": basis,
            "premium": 0.0, "campaign": 1, "last_spot": last_spot,
            "short": None}


def test_off_by_default_the_write_path_is_byte_identical():
    """Defaults None -> covered-call write behaves exactly as before this
    model existed. Reuses the assigned-shares shape from the portfolio
    tests: a held position with no short leg, eligible to write a call."""
    ch = _chain(right="C", oi=5, vol=0, strike=40.0, bid=0.90, ask=1.30)
    cfg = _cfg_obj()
    st, r = _step(ch, cfg, positions=[_held_pos(100.0)])
    actions = [t.action for t in r.trades]
    assert "SELL_CALL" in actions
    write = next(t for t in r.trades if t.action == "SELL_CALL")
    assert write.price_per_contract == pytest_approx(0.90), \
        "no exit_impact_max_spreads set -> the write prices at the clean bid"


def test_an_oversized_write_is_priced_worse_and_can_miss_the_credit_floor():
    """The whole point of the model: pricing happens BEFORE the
    feasibility checks, so a call too large to sell at a real price can also
    fail to clear write_credit_ok, not just book a worse proceeds number.
    25 contracts (2500 shares) against oi=10 is wildly past a 10% ratio; at
    bid=0.05/ask=0.10 (spread 0.05) one capped spread brings the priced bid
    to 0.00, below write_credit_floor's MIN_TICK -- the clean-quote path
    would write it, the impact-priced path must refuse."""
    ch = _chain(right="C", oi=10, vol=0, strike=40.0, bid=0.05, ask=0.10)
    clean_cfg = _cfg_obj()
    impact_cfg = _cfg_obj(exit_impact_max_spreads=1.0,
                           liq_max_pct_of_open_interest=0.10,
                           liq_max_pct_of_volume=0.25)
    _, clean = _step(ch, clean_cfg, positions=[_held_pos(2500.0)])
    _, impacted = _step(ch, impact_cfg, positions=[_held_pos(2500.0)])
    clean_wrote = "SELL_CALL" in [t.action for t in clean.trades]
    impacted_wrote = "SELL_CALL" in [t.action for t in impacted.trades]
    assert clean_wrote, "sanity: the clean-quote path must write at bid=0.05"
    assert not impacted_wrote, \
        "the impact-priced write must miss the credit floor once charged " \
        "for its own size (bid 0.05 - one 0.05-wide spread <= MIN_TICK)"


def test_entry_side_is_never_touched_by_the_model():
    """MECHANISM guard from the variant note: this model touches only the
    covered-call write and the put buyback, never SELL_PUT. A SELL_PUT
    entry with the exact same knobs set must be unaffected."""
    ch = _chain(right="P", oi=200, vol=1000, strike=40.0, bid=0.90, ask=1.00)
    cfg = _cfg_obj(exit_impact_max_spreads=1.0,
                    liq_max_pct_of_open_interest=0.10,
                    liq_max_pct_of_volume=0.25)
    st, r = _step(ch, cfg)
    assert [t.action for t in r.trades] == ["SELL_PUT"]
    entry = r.trades[0]
    assert entry.price_per_contract == pytest_approx(0.90), \
        "the entry leg must price at the clean bid -- impact is exit-only"
