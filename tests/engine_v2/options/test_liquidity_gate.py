"""A2: the liquidity gate. An ungated bot requested 757 DOW contracts against
~84/day traded (9x the whole market); a rel-spread/OI/volume veto would have
refused 114 of its 145 real entries -- fills that could not have happened, not lost profit. The
gate is a VETO on new short-put entries (and roll destinations): it must never
touch closes, expiry settlement, held-leg marks, or covered calls
(owner-provisional 2026-08-01). Default-off at the dataclass like every other
gate; production turns it on in FROZEN."""
import pandas as pd

from src.engine_v2.options.portfolio import PortfolioState, step_one_day
from src.engine_v2.options.wheel import WheelConfig

D = pd.Timestamp("2026-07-21")
COLS = ["date", "expiry", "dte", "strike", "right", "bid", "ask", "mid",
        "close", "delta", "iv", "underlying",
        "open_interest", "volume", "bid_size", "ask_size"]


def _chain(bid=0.90, ask=1.30, oi=5.0, vol=0.0):
    # DOW-shaped: 36% rel-spread, 5 open interest, zero volume
    ch = pd.DataFrame([[D, D + pd.Timedelta(days=11), 11, 40.0, "P", bid, ask,
                        (bid + ask) / 2, (bid + ask) / 2, -0.30, 0.2, 41.0,
                        oi, vol, 10.0, 10.0]], columns=COLS)
    ch["date"] = pd.to_datetime(ch["date"])
    ch["expiry"] = pd.to_datetime(ch["expiry"])
    return ch


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


def _cfg(**liq):
    return WheelConfig(ticker="DOW", starting_capital=100_000.0, put_delta=0.30,
                       call_delta=0.50, target_dte=11, take_profit_pct=0.60,
                       call_min_strike="basis", **liq)


def _step(ch, cfg):
    st = PortfolioState(cash=100_000.0, positions=[])
    r = step_one_day(st, M(ch), D, cfg, selector="plain", n_slots=1)
    return st, r


def test_gated_engine_refuses_the_illiquid_entry():
    """The A2 defect: with the production thresholds set, the engine must NOT
    sell a 36%-spread, 5-OI, zero-volume put."""
    st, r = _step(_chain(), _cfg(liq_max_rel_spread=0.10,
                                 liq_min_open_interest=250, liq_min_volume=25))
    assert [t.action for t in r.trades] == [], \
        "A2: engine sold an entry the liquidity gate must refuse"
    assert st.positions == []
    assert any(w[1] == "entry_gated_illiquid" for w in r.warnings), \
        "A2: a gated entry must be visible in warnings, not silent"


def test_gate_off_by_default_sells_exactly_as_before():
    # dataclass defaults are None -> plain path byte-identical (the anchors)
    st, r = _step(_chain(), _cfg())
    assert [t.action for t in r.trades] == ["SELL_PUT"]
    assert r.trades[0].contracts == 25


def test_liquid_contract_passes_all_three_legs():
    st, r = _step(_chain(bid=1.20, ask=1.30, oi=3000.0, vol=250.0),
                  _cfg(liq_max_rel_spread=0.10, liq_min_open_interest=250,
                       liq_min_volume=25))
    assert [t.action for t in r.trades] == ["SELL_PUT"]


def test_missing_field_with_set_threshold_refuses():
    """Backtest-shaped chain (no OI/volume columns) + a set OI threshold must
    REFUSE -- a contract whose liquidity cannot be measured is not one to
    sell. (Backtest configs simply leave those legs None.)"""
    ch = _chain(bid=1.20, ask=1.30).drop(columns=["open_interest", "volume",
                                                  "bid_size", "ask_size"])
    st, r = _step(ch, _cfg(liq_min_open_interest=250))
    assert [t.action for t in r.trades] == []


def test_predicate_legs_individually():
    from src.engine_v2.options.select import liquidity_ok, select_contract
    ch = _chain(bid=1.20, ask=1.30, oi=3000.0, vol=250.0)
    c = select_contract(ch, D, "P", 0.30, 11, "DOW")
    ok, why = liquidity_ok(ch, D, c, _cfg(liq_max_rel_spread=0.10,
                                          liq_min_open_interest=250,
                                          liq_min_volume=25))
    assert ok and why == ""
    ok, why = liquidity_ok(_chain(bid=0.90, ask=1.30, oi=3000.0, vol=250.0),
                           D, c, _cfg(liq_max_rel_spread=0.10))
    assert not ok and why == "rel_spread"
    ok, why = liquidity_ok(_chain(bid=1.20, ask=1.30, oi=5.0, vol=250.0),
                           D, c, _cfg(liq_min_open_interest=250))
    assert not ok and why == "open_interest"
    ok, why = liquidity_ok(_chain(bid=1.20, ask=1.30, oi=3000.0, vol=0.0),
                           D, c, _cfg(liq_min_volume=25))
    assert not ok and why == "volume"
    # NaN with a set threshold fails
    ok, why = liquidity_ok(_chain(bid=1.20, ask=1.30, oi=float("nan")),
                           D, c, _cfg(liq_min_open_interest=250))
    assert not ok and why == "open_interest"
    # all thresholds None -> pass, regardless of how ugly the row is
    ok, why = liquidity_ok(_chain(), D, c, _cfg())
    assert ok


def test_predicate_body_pins():
    """Skeptic F1: five body mutants survived the original tests. Each
    assertion here exists to kill one -- do not delete without replacing the
    mutant run."""
    from src.engine_v2.options.select import liquidity_ok, select_contract
    ch = _chain(bid=1.20, ask=1.30, oi=3000.0, vol=250.0)
    c = select_contract(ch, D, "P", 0.30, 11, "DOW")
    cfg_all = _cfg(liq_max_rel_spread=0.10, liq_min_open_interest=250,
                   liq_min_volume=25)
    # M1: crossed book must fail closed, not slide into the spread formula
    crossed = _chain(bid=1.30, ask=1.20, oi=3000.0, vol=250.0)
    assert liquidity_ok(crossed, D, c, cfg_all) == (False, "no_two_sided_market")
    # M2: rel-spread EXACTLY at the threshold passes (> semantics, not >=).
    # 3.5/4.5 are binary-exact: 1.0/4.0 == 0.25 with no float dust.
    at = _chain(bid=3.5, ask=4.5, oi=3000.0, vol=250.0)
    assert liquidity_ok(at, D, c, _cfg(liq_max_rel_spread=0.25))[0]
    # M3: OI/volume EXACTLY at the floor pass (< semantics, not <=)
    edge = _chain(bid=1.20, ask=1.30, oi=250.0, vol=25.0)
    assert liquidity_ok(edge, D, c, cfg_all)[0]
    # M4: a contract with no row at all is refused, never passed
    from src.engine_v2.options.chain import Contract
    ghost = Contract("DOW", D + pd.Timedelta(days=11), 99.0, "P")
    assert liquidity_ok(ch, D, ghost, cfg_all) == (False, "row_missing")
    # M5: the denominator is the COMPUTED midpoint, never the `mid` column --
    # live's mid is Schwab's mark. A huge mark on a wide book must still fail.
    trap = _chain(bid=0.90, ask=1.30, oi=3000.0, vol=250.0)
    trap["mid"] = 40.0                     # (ask-bid)/mid_column would be 1%
    assert liquidity_ok(trap, D, c, cfg_all) == (False, "rel_spread")
    # F7: a spliced mark-only duplicate row must never answer for the contract
    dup = pd.concat([_chain(bid=0.01, ask=5.00, oi=0.0, vol=0.0),
                     _chain(bid=1.20, ask=1.30, oi=3000.0, vol=250.0)],
                    ignore_index=True)
    dup["held_only"] = [True, False]
    assert liquidity_ok(dup, D, c, cfg_all)[0]


def test_solo_engines_warn_on_gated_entry():
    """Skeptic F2: the backtest engines must not veto silently -- a gated day
    and a no-weather day would be indistinguishable in days_flat."""
    from src.engine_v2.options.wheel import run_wheel
    ch = _chain()   # illiquid
    cfg = _cfg(liq_max_rel_spread=0.10)
    r = run_wheel(ch, cfg)
    assert not r.trades
    assert any(w[1] == "entry_gated_illiquid" for w in (r.warnings or []))


def test_portfolio_warning_deduped_across_slots(monkeypatch):
    # Skeptic F4: n_slots=2 used to append the identical warning per iteration
    st = PortfolioState(cash=100_000.0, positions=[])
    r = step_one_day(st, M(_chain()), D,
                     _cfg(liq_max_rel_spread=0.10), selector="plain", n_slots=2)
    gated = [w for w in r.warnings if w[1] == "entry_gated_illiquid"]
    assert gated == [(D, "entry_gated_illiquid", "DOW")]
