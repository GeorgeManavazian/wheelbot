"""The A2 gate measured the wrong thing.

Measured on the live 2026-08-10 chains: of 27 good-to-rent names, the gate
refused 25 and `rel_spread` was the proximate cause on EVERY one -- dropping
the OI and volume legs entirely changed nothing (2/27 either way). The spread
leg refuses on CHEAPNESS, not illiquidity: option prices move on a 1-cent
grid, so at a $0.10 mid the tightest market that can physically exist is
already 10% of mid. WBD (bid 0.09 / ask 0.14, open interest 6798, volume 113 —
one of the busiest contracts on the board) was refused; CCL passed only
because its premium was expensive enough in absolute cents that an ordinary
market was a small fraction of it.

The case the gate was built for -- 757 DOW contracts against ~84/day
traded -- is a RATIO of order size to available liquidity, and the gate never
saw order size at all: liquidity_ok(chain, date, contract, cfg) takes no
account, no capital, no contract count. So a 5k account buying 1 contract was
refused for exactly the same reason as a 500k account buying 757, and the live
log printed an identical 25-ticker refusal for all 25 accounts.

These tests pin the replacement: OI/volume floors keep their veto, and the
size ratio CAPS the order rather than refusing the name -- refusing would leave
the slot idle, which is the starvation failure the rest of the engine fights.
"""
import pandas as pd

from src.engine_v2.options.portfolio import PortfolioState, step_one_day
from src.engine_v2.options.wheel import WheelConfig

D = pd.Timestamp("2026-07-21")
COLS = ["date", "expiry", "dte", "strike", "right", "bid", "ask", "mid",
        "close", "delta", "iv", "underlying",
        "open_interest", "volume", "bid_size", "ask_size"]


def _chain(bid=0.90, ask=1.30, oi=5.0, vol=0.0, strike=40.0):
    ch = pd.DataFrame([[D, D + pd.Timedelta(days=11), 11, strike, "P", bid, ask,
                        (bid + ask) / 2, (bid + ask) / 2, -0.30, 0.2,
                        strike + 1.0, oi, vol, 10.0, 10.0]], columns=COLS)
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


def _cfg(capital=100_000.0, **liq):
    return WheelConfig(ticker="DOW", starting_capital=capital, put_delta=0.30,
                       call_delta=0.50, target_dte=11, take_profit_pct=0.60,
                       call_min_strike="basis", **liq)


def _step(ch, cfg, capital=100_000.0, n_slots=1):
    st = PortfolioState(cash=capital, positions=[])
    r = step_one_day(st, M(ch), D, cfg, selector="plain", n_slots=n_slots)
    return st, r


# --- the spread leg must stop refusing cheap-but-liquid contracts ----------

def test_cheap_but_liquid_contract_is_no_longer_refused():
    """The WBD case. A 5-cent-wide market on a 9-cent bid is 43% relative --
    and 6798 open contracts say it is one of the most tradeable puts listed.
    With the spread leg off and the real liquidity floors on, it trades."""
    ch = _chain(bid=0.09, ask=0.14, oi=6798, vol=113, strike=26.0)
    st, r = _step(ch, _cfg(liq_max_rel_spread=None,
                           liq_min_open_interest=250, liq_min_volume=25))
    assert [t.action for t in r.trades] == ["SELL_PUT"], \
        "a 6798-OI contract must not be refused for being cheap"


def test_illiquid_contract_is_still_refused_by_the_floors():
    """Dropping the spread leg must not disarm the gate: the DOW-shaped
    5-OI / zero-volume contract stays refused, and stays loud."""
    st, r = _step(_chain(oi=5.0, vol=0.0),
                  _cfg(liq_max_rel_spread=None,
                       liq_min_open_interest=250, liq_min_volume=25))
    assert [t.action for t in r.trades] == []
    assert any(w[1] == "entry_gated_illiquid" for w in r.warnings)


# --- the size cap: the ratio the gate never checked -------------------------

def test_order_is_capped_to_a_share_of_open_interest():
    """100k against a $40 strike affords 25 contracts, but 5% of 200 open
    interest is 10. The engine sells 10 -- it does not refuse the name."""
    ch = _chain(bid=0.90, ask=1.00, oi=200, vol=1000)
    st, r = _step(ch, _cfg(liq_max_rel_spread=None,
                           liq_min_open_interest=100, liq_min_volume=25,
                           liq_max_pct_of_open_interest=0.05,
                           liq_max_pct_of_volume=0.10))
    assert [t.action for t in r.trades] == ["SELL_PUT"]
    assert r.trades[0].contracts == 10, \
        f"expected the OI cap to bind at 10, got {r.trades[0].contracts}"


def test_order_is_capped_to_a_share_of_volume():
    """The tighter of the two legs wins: 10% of 60 traded is 6."""
    ch = _chain(bid=0.90, ask=1.00, oi=100_000, vol=60)
    st, r = _step(ch, _cfg(liq_max_rel_spread=None,
                           liq_min_open_interest=100, liq_min_volume=25,
                           liq_max_pct_of_open_interest=0.05,
                           liq_max_pct_of_volume=0.10))
    assert r.trades[0].contracts == 6


def test_the_cap_is_per_account_so_a_small_account_is_untouched():
    """The whole point: each account is its own world. The same contract that
    caps the 100k account does not bind the 5k one, which can only afford 1."""
    ch = _chain(bid=0.90, ask=1.00, oi=200, vol=1000)
    liq = dict(liq_max_rel_spread=None, liq_min_open_interest=100,
               liq_min_volume=25, liq_max_pct_of_open_interest=0.05,
               liq_max_pct_of_volume=0.10)
    _, small = _step(ch, _cfg(capital=5_000.0, **liq), capital=5_000.0)
    _, big = _step(ch, _cfg(capital=100_000.0, **liq), capital=100_000.0)
    assert small.trades[0].contracts == 1, "5k affords 1 contract, uncapped"
    assert big.trades[0].contracts == 10, "100k affords 25, capped to 10"


def test_a_cap_below_one_contract_refuses_and_says_so():
    """When even one contract is too much of the market, refuse -- but under
    its own reason, never silently."""
    ch = _chain(bid=0.90, ask=1.00, oi=300, vol=30)
    st, r = _step(ch, _cfg(liq_max_rel_spread=None,
                           liq_min_open_interest=250, liq_min_volume=25,
                           liq_max_pct_of_open_interest=0.001,
                           liq_max_pct_of_volume=0.001))
    assert [t.action for t in r.trades] == []
    assert any(w[1] == "entry_gated_size" for w in r.warnings), \
        "a size-refused entry must be visible in warnings, not silent"


def test_an_unmeasurable_field_refuses_when_the_cap_is_set():
    """Same stance as every other threshold in this engine: a rule that is SET
    but cannot be measured refuses, it never waves the entry through."""
    ch = _chain(bid=0.90, ask=1.00, oi=200, vol=1000)
    ch["open_interest"] = float("nan")
    st, r = _step(ch, _cfg(liq_max_rel_spread=None,
                           liq_min_open_interest=None, liq_min_volume=None,
                           liq_max_pct_of_open_interest=0.05))
    assert [t.action for t in r.trades] == []
    assert any(w[1] == "entry_gated_size" for w in r.warnings)


def test_size_cap_off_by_default_is_byte_identical():
    """Defaults None -> the plain path is unchanged: 25 contracts, as the
    pre-existing anchor test pins."""
    st, r = _step(_chain(), _cfg())
    assert [t.action for t in r.trades] == ["SELL_PUT"]
    assert r.trades[0].contracts == 25
