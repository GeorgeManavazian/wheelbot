"""A15: bounded settlement reach-back for the BATCH engine only.

The A14-era settlement fix reads only the expiry day's own close (correct
live: refuse + warn + retry when history is restored). But the batch driver
has no later run -- a ticker whose chain lacks rows on an expiry date (SPY
has two such days) left the leg warning daily forever, consuming its slot to
the end of the backtest, then bought back by the residual finalizer at a
carried ask instead of settling. The solo engine kept its own reach-back;
only the portfolio/batch path regressed. Fix: BatchMarket.bounded_settle_price
(last close within 5 calendar days at/before expiry); step_one_day falls
through to it ONLY when the market provides it -- LiveMarket does not, so
live behavior is byte-identical."""
import pandas as pd

from src.engine_v2.options.wheel import WheelConfig
from src.engine_v2.options.portfolio import run_portfolio_wheel

COLS = ["date", "expiry", "dte", "strike", "right", "bid", "ask", "mid",
        "close", "delta", "iv", "underlying"]


def _chain(rows):
    ch = pd.DataFrame(rows, columns=COLS)
    ch["date"] = pd.to_datetime(ch["date"])
    ch["expiry"] = pd.to_datetime(ch["expiry"])
    return ch


def _cfg():
    return WheelConfig(starting_capital=50_000.0, put_delta=0.30,
                       call_delta=0.30, take_profit_pct=None,
                       commission_per_contract=0.0, call_min_strike="basis")


def _states():
    df = pd.DataFrame([("2024-01-01", "uptrend", "calm", 0.20)],
                      columns=["date", "trend", "vol", "vol_pctile"]
                      ).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df


def test_gap_day_expiry_settles_from_bounded_reachback():
    """Expiry 01-09 absent from the chain; last close 01-08 (deep ITM).
    The batch engine must ASSIGN at that close on the first post-expiry day
    instead of zombieing the leg to the residual finalizer."""
    rows = [
        ["2024-01-02", "2024-01-09", 7, 470.0, "P", 2.00, 2.10, 2.05, 2.05,
         -0.30, 0.1, 472.0],
        ["2024-01-08", "2024-01-09", 1, 470.0, "P", 118.0, 120.0, 119.0,
         119.0, -0.99, 0.1, 350.0],
        # post-expiry day exists; expiry day itself has NO rows
        ["2024-01-10", "2024-01-17", 7, 300.0, "P", 2.00, 2.10, 2.05, 2.05,
         -0.30, 0.1, 350.0],
    ]
    res = run_portfolio_wheel({"SPY": _chain(rows)}, _cfg(),
                              {"SPY": _states()})
    assigned = [t for t in res.trades if t.action == "ASSIGNED"]
    assert assigned, "A15: gap-day expiry zombied instead of settling"
    assert pd.Timestamp(assigned[0].date) == pd.Timestamp("2024-01-10")
    assert res.residual_settled is False
    assert res.final_shares.get("SPY", 0) >= 100
    # skeptic F6: pin the payload TYPE too -- the A14 generic printer names
    # the leg from w[2]; a ticker string there degrades the disclosure
    assert any(w[1] == "expiry_settled_reachback"
               and getattr(w[2], "strike", None) == 470.0
               for w in res.warnings)


def test_reachback_is_bounded_not_unbounded():
    """Last close 8 days before expiry: OUTSIDE the 5-day bound -- the leg
    must stay honestly unsettleable (never settle on stale territory)."""
    rows = [
        ["2024-01-02", "2024-01-10", 8, 470.0, "P", 2.00, 2.10, 2.05, 2.05,
         -0.30, 0.1, 472.0],
        ["2024-01-12", "2024-01-19", 7, 300.0, "P", 2.00, 2.10, 2.05, 2.05,
         -0.30, 0.1, 350.0],
    ]
    res = run_portfolio_wheel({"SPY": _chain(rows)}, _cfg(),
                              {"SPY": _states()})
    assert not [t for t in res.trades if t.action == "ASSIGNED"]
    assert any(w[1] == "expiry_unsettleable" for w in res.warnings)
    assert res.residual_settled is True


def test_normal_settle_stays_quiet_and_primary():
    """Warn-always / reachback-primary mutant killer: when the expiry day's
    own close EXISTS, settlement must use it (assign at the 01-09 close, not
    an earlier one) and emit NO expiry_settled_reachback warning."""
    rows = [
        ["2024-01-02", "2024-01-09", 7, 470.0, "P", 2.00, 2.10, 2.05, 2.05,
         -0.30, 0.1, 472.0],
        # expiry day present, deep ITM
        ["2024-01-09", "2024-01-09", 0, 470.0, "P", 118.0, 120.0, 119.0,
         119.0, -0.99, 0.1, 350.0],
        ["2024-01-10", "2024-01-17", 7, 300.0, "P", 2.00, 2.10, 2.05, 2.05,
         -0.30, 0.1, 350.0],
    ]
    res = run_portfolio_wheel({"SPY": _chain(rows)}, _cfg(),
                              {"SPY": _states()})
    assert [t for t in res.trades if t.action == "ASSIGNED"]
    assert not any(w[1] == "expiry_settled_reachback" for w in res.warnings)
