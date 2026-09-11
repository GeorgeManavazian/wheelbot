"""Earnings blackout gate (spec 2026-08-03-earnings-blackout-design).

The chop scanner ranks candidates by vol percentile descending, and implied vol
rises into a scheduled print -- so the ranking systematically prefers names with
an earnings gap coming. This gate vetoes those entries. Puts only, veto never
substitute, visible in warnings, default off.

The three-state calendar (dates / known-none / unknown) is load-bearing: both an
ETF and a failed lookup present as an empty calendar, and collapsing them is how
a gate stops existing without anyone noticing.
"""
import pandas as pd
import pytest

from src.engine_v2.options.earnings import (BLACKOUT_BUFFER_DAYS,
                                            EarningsCalendar, in_blackout)
from src.engine_v2.options.portfolio import (PortfolioState, run_portfolio_wheel,
                                             step_one_day)
from src.engine_v2.options.wheel import WheelConfig

D = pd.Timestamp("2026-07-21")
EXPIRY = D + pd.Timedelta(days=11)          # 2026-08-01
COLS = ["date", "expiry", "dte", "strike", "right", "bid", "ask", "mid",
        "close", "delta", "iv", "underlying",
        "open_interest", "volume", "bid_size", "ask_size"]


def _chain():
    ch = pd.DataFrame([[D, EXPIRY, 11, 40.0, "P", 0.90, 1.00, 0.95, 0.95,
                        -0.30, 0.2, 41.0, 500.0, 100.0, 10.0, 10.0]],
                      columns=COLS)
    ch["date"] = pd.to_datetime(ch["date"])
    ch["expiry"] = pd.to_datetime(ch["expiry"])
    return ch


class M:
    """Market fake that HAS the calendar capability."""
    universe = ["DOW"]

    def __init__(self, earnings_map=None):
        self._ch = _chain()
        self._earnings = earnings_map or {}

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

    def earnings_dates(self, tk):
        return self._earnings.get(tk)      # absent -> None -> unknown

    def earnings_known(self) -> bool:
        """The capability DECLARATION step_one_day requires (added 2026-08-09,
        when the gate went live). Deliberately about the calendar's contents,
        not the class -- see the guard in portfolio.step_one_day."""
        return bool(self._earnings)


class MNoCapability(M):
    """Market fake WITHOUT earnings_dates -- the ~20 pre-existing test fakes."""
    earnings_dates = None

    def __getattribute__(self, name):
        if name in ("earnings_dates", "earnings_known"):
            raise AttributeError(name)
        return super().__getattribute__(name)


def _cfg(**kw):
    return WheelConfig(ticker="DOW", starting_capital=100_000.0, put_delta=0.30,
                       call_delta=0.50, target_dte=11, take_profit_pct=0.60,
                       call_min_strike="basis", **kw)


def _step(market, cfg):
    st = PortfolioState(cash=100_000.0, positions=[])
    return st, step_one_day(st, market, D, cfg, selector="plain", n_slots=1)


# ---------------------------------------------------------------- predicate

@pytest.mark.parametrize("day,blocked", [
    ("2026-07-20", False),   # day before observation -- already happened
    ("2026-07-21", True),    # observation day itself, inclusive
    ("2026-07-27", True),    # mid-life, the ordinary case
    ("2026-08-01", True),    # expiry day itself
    ("2026-08-02", True),    # expiry +1: the after-close (AMC) correction
    ("2026-08-03", False),   # expiry +2: past the window, the put is settled
])
def test_window_edges(day, blocked):
    assert in_blackout([pd.Timestamp(day)], D, EXPIRY) is blocked


def test_amc_buffer_is_one_day():
    """Pin the constant: the +1 exists because a print released after the close
    on day X moves the stock on X+1. If this changes, the spec changed."""
    assert BLACKOUT_BUFFER_DAYS == 1


def test_unknown_and_known_none_both_allow():
    assert in_blackout(None, D, EXPIRY) is False      # unknown -> allow + count
    assert in_blackout([], D, EXPIRY) is False        # ETF -> genuinely none


def test_any_date_in_a_list_blocks():
    dates = [pd.Timestamp("2026-01-15"), pd.Timestamp("2026-07-27"),
             pd.Timestamp("2026-11-02")]
    assert in_blackout(dates, D, EXPIRY) is True


def test_time_of_day_is_ignored_not_parsed():
    """Timestamps carry the BMO/AMC signal in the clock; we drop it and widen
    the window instead. A 16:00 stamp on expiry day must still block."""
    assert in_blackout([pd.Timestamp("2026-08-01 16:00")], D, EXPIRY) is True


# ---------------------------------------------------------------- calendar

def test_calendar_load_keeps_the_three_states_apart(tmp_path):
    d = tmp_path / "earnings"
    d.mkdir()
    pd.DataFrame({
        "ticker": ["AAPL", "AAPL"],
        "earnings_date": pd.to_datetime(["2026-07-27", "2026-10-29"]),
    }).to_parquet(d / "calendar.parquet", index=False)
    pd.DataFrame({
        "ticker": ["AAPL", "SPY", "NVDA"],
        "status": ["ok", "none", "failed: HTTPError: 429"],
        "n_dates": [2, 0, 0],
        "pulled_at": pd.to_datetime(["2026-08-03"] * 3),
    }).to_parquet(d / "status.parquet", index=False)

    cal = EarningsCalendar.load(str(d))
    assert len(cal.dates("AAPL")) == 2          # ok      -> dates
    assert cal.dates("SPY") == []               # none    -> known-empty
    assert cal.dates("NVDA") is None            # failed  -> UNKNOWN, not empty
    assert cal.dates("MSFT") is None            # never pulled -> unknown
    assert cal.known("SPY") and not cal.known("NVDA")


def test_calendar_load_missing_files_is_empty_not_a_crash(tmp_path):
    cal = EarningsCalendar.load(str(tmp_path))
    assert len(cal) == 0 and cal.dates("AAPL") is None


# ---------------------------------------------------------------- engine

def test_gate_off_by_default_sells_exactly_as_before():
    """Default False -> plain path byte-identical, even with a print sitting
    squarely inside the window."""
    st, r = _step(M({"DOW": [pd.Timestamp("2026-07-27")]}), _cfg())
    assert [t.action for t in r.trades] == ["SELL_PUT"]


def test_gated_engine_refuses_the_entry_and_says_so():
    st, r = _step(M({"DOW": [pd.Timestamp("2026-07-27")]}),
                  _cfg(earnings_blackout=True))
    assert [t.action for t in r.trades] == [], \
        "sold a put through a scheduled print"
    assert st.positions == []
    assert any(w[1] == "entry_gated_earnings" for w in r.warnings), \
        "a gated entry must be visible in warnings, not silent"


def test_gate_on_but_print_outside_the_window_still_sells():
    st, r = _step(M({"DOW": [pd.Timestamp("2026-09-15")]}),
                  _cfg(earnings_blackout=True))
    assert [t.action for t in r.trades] == ["SELL_PUT"]
    assert not any(w[1] == "entry_gated_earnings" for w in r.warnings)


def test_unknown_ticker_allows_the_entry():
    """Standing convention: unknown state always allows. The blindness is
    counted at the calendar level, not by refusing to trade.

    The calendar knows about SOME OTHER name, deliberately. This test used an
    EMPTY calendar until 2026-08-09, which conflated the two states the whole
    module exists to keep apart: 'loaded fine, no data for DOW' (allow) and
    'nothing loaded at all' (refuse -- a failed pull, not a quiet universe).
    An empty calendar can no longer stand in for the first."""
    st, r = _step(M({"AAPL": [pd.Timestamp("2026-07-27")]}),
                  _cfg(earnings_blackout=True))
    assert [t.action for t in r.trades] == ["SELL_PUT"]


def test_market_without_the_capability_now_REFUSES_rather_than_going_inert():
    """⚠ CONTRACT CHANGED 2026-08-09, when the gate was turned on in FROZEN.

    This test previously asserted the opposite -- that a market with no calendar
    left the gate inert, so the ~20 pre-existing fakes would not have to grow a
    method. That was tolerable only while earnings_blackout defaulted to False.

    With the gate ON in the live config it is not: the LIVE bot calls
    step_one_day DIRECTLY (live/run_daily.py), so 'inert when uncapable' meant
    the one path that matters had no refusal at all. It would have reported
    protection it was not giving, and logged nothing, because per-ticker
    unknown-allows is the correct behaviour it was imitating.

    Same stance step_one_day already takes on rank_by='iv_rank': absent
    declaration is not consent. A market that cannot answer the gate must say so
    and stop the run."""
    with pytest.raises(ValueError, match="earnings"):
        _step(MNoCapability(), _cfg(earnings_blackout=True))


def test_warning_is_deduped_per_day_and_ticker():
    """The n_slots while-loop revisits gated tickers on every iteration."""
    st = PortfolioState(cash=100_000.0, positions=[])
    r = step_one_day(st, M({"DOW": [pd.Timestamp("2026-07-27")]}), D,
                     _cfg(earnings_blackout=True), selector="plain", n_slots=3)
    assert sum(1 for w in r.warnings if w[1] == "entry_gated_earnings") == 1


def test_blackout_without_a_calendar_refuses_to_run():
    """A gate believed to be on must never be structurally inert -- same stance
    as the A2 liquidity gate refusing an unmeasurable contract."""
    with pytest.raises(ValueError, match="needs an earnings calendar"):
        run_portfolio_wheel({"SPY": _chain()}, _cfg(earnings_blackout=True),
                            {"SPY": pd.DataFrame()}, selector="chop")
