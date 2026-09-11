"""A12: equal-split budgeting strands capital at small account sizes.
$5k with N=5 -> $1,000/slot -> any strike > $10 sizes to n=0 and the loop
abandons the day with ~$4.1k idle (5k_N5 measured at 18.0% utilization, ONE
campaign in 7 sessions). The capital x N grid then measures which slots can
afford anything, not N. Fix: equal split FIRST (unchanged when affordable);
when nothing fits, re-split over fewer effective slots down to one --
concentration only when the alternative is idleness (provisional)."""
import pandas as pd

from src.engine_v2.options.portfolio import PortfolioState, step_one_day
from src.engine_v2.options.wheel import WheelConfig

D = pd.Timestamp("2026-07-21")
COLS = ["date", "expiry", "dte", "strike", "right", "bid", "ask", "mid",
        "close", "delta", "iv", "underlying"]


def _chain(strike, bid=1.00):
    ch = pd.DataFrame([[D, D + pd.Timedelta(days=11), 11, strike, "P", bid,
                        bid + 0.10, bid + 0.05, bid + 0.05, -0.30, 0.2,
                        strike + 1.0]], columns=COLS)
    ch["date"] = pd.to_datetime(ch["date"])
    ch["expiry"] = pd.to_datetime(ch["expiry"])
    return ch


class M:
    def __init__(self, chains):
        self._chains = chains
        self.universe = list(chains)

    def chain(self, tk, d):
        return self._chains[tk]

    def spot(self, tk, d, fb):
        return 41.0

    def settle_price(self, tk, expiry):
        return 41.0

    def regime_row(self, tk, day):
        return None

    def eligible(self, tk, day):
        return True


def _cfg():
    return WheelConfig(ticker="SPY", starting_capital=5_000.0, put_delta=0.30,
                       call_delta=0.50, target_dte=11, take_profit_pct=0.60,
                       call_min_strike="basis")


def test_small_account_no_longer_strands_capital():
    """$5k, N=5, only a $30-strike listed: equal split offers $1k/slot (fits
    nothing), but the full $5k affords one $3k contract. The old loop went
    idle; the account must place ONE concentrated position instead."""
    st = PortfolioState(cash=5_000.0, positions=[])
    r = step_one_day(st, M({"GDX": _chain(30.0)}), D, _cfg(), selector="plain",
                     n_slots=5)
    assert [t.action for t in r.trades] == ["SELL_PUT"], \
        "A12: $5k sat idle while a $3k contract was listed"
    assert r.trades[0].contracts == 1


def test_equal_split_unchanged_when_affordable():
    # $100k, N=2, two cheap names: both slots fill exactly as before, best
    # premium (highest vol_pctile is absent -> universe order) first
    st = PortfolioState(cash=100_000.0, positions=[])
    r = step_one_day(st, M({"GDX": _chain(30.0), "SLV": _chain(20.0)}), D,
                     WheelConfig(ticker="SPY", starting_capital=100_000.0,
                                 put_delta=0.30, call_delta=0.50, target_dte=11,
                                 take_profit_pct=0.60, call_min_strike="basis"),
                     selector="plain", n_slots=2)
    assert [t.action for t in r.trades] == ["SELL_PUT", "SELL_PUT"]
    assert {t.contract.root for t in r.trades} == {"GDX", "SLV"}
    # slot 1: $50k/slot -> 16 x $30-strike; slot 2 re-divides AFTER slot 1's
    # premium lands: (100k + 1589.6 - 48k)/1 // 2000 = 26
    assert sorted(t.contracts for t in r.trades) == [16, 26]


class MRanked(M):
    def __init__(self, chains, rows):
        super().__init__(chains)
        self._rows = rows

    def regime_row(self, tk, day):
        return self._rows[tk]


def test_fallback_buys_richest_ranked_not_cheapest():
    """Owner decision 2026-08-01 (A12 skeptic F3): when the equal split
    affords nothing, the pooled money buys the BEST-RANKED name that fits at
    any concentration, not the cheapest name at the least concentration.
    $3.5k: CHEAP ($1.2k collateral, pctile 0.10) fits at k=2; RICH ($3.0k,
    pctile 0.90) fits only at k=1. The account must buy RICH and then
    honestly idle -- not CHEAP."""
    st = PortfolioState(cash=3_500.0, positions=[])
    rows = {"CHEAP": {"trend": "uptrend", "vol": "calm", "vol_pctile": 0.10},
            "RICH": {"trend": "uptrend", "vol": "calm", "vol_pctile": 0.90}}
    r = step_one_day(st, MRanked({"CHEAP": _chain(12.0), "RICH": _chain(30.0)},
                                 rows), D, _cfg(), selector="plain", n_slots=5)
    assert [(t.action, t.contract.root) for t in r.trades] == \
        [("SELL_PUT", "RICH")], \
        "A12 owner routing: fallback must buy the richest-ranked fitting name"


def test_fallback_sizes_at_least_concentration_that_fits():
    """The fallback buys the best name at the LARGEST k that affords it --
    one contract's worth of concentration, never the whole pot. $4.4k / N4:
    equal split ($1.1k) misses the $1.2k contract; k=3 fits n=1. Sizing at
    k=1 would buy 3 contracts -- forbidden."""
    st = PortfolioState(cash=4_400.0, positions=[])
    r = step_one_day(st, M({"GDX": _chain(12.0)}), D, _cfg(), selector="plain",
                     n_slots=4)
    assert [t.action for t in r.trades] == ["SELL_PUT"]
    assert r.trades[0].contracts == 1, \
        "A12: fallback must size at the least concentration that fits"


def test_totally_unaffordable_day_still_goes_idle():
    # $1k against a $30 strike: even the full pot affords nothing -> idle,
    # correctly (no fractional contracts)
    st = PortfolioState(cash=1_000.0, positions=[])
    r = step_one_day(st, M({"GDX": _chain(30.0)}), D, _cfg(), selector="plain",
                     n_slots=5)
    assert r.trades == []
