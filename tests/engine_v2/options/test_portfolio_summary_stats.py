import math
import pandas as pd
from src.engine_v2.options.report import portfolio_summary_stats


def _campaigns():
    # one closed winner, one open-and-underwater (pnl_mtm < 0)
    return pd.DataFrame([
        dict(campaign_id=1, open_at_end=False, pnl_realized=199.35, pnl_mtm=199.35, pct_return=0.005),
        dict(campaign_id=2, open_at_end=True, pnl_realized=-2900.65, pnl_mtm=-400.65, pct_return=-0.9),
    ])


def test_twin_win_rates_diverge():
    s = portfolio_summary_stats(_campaigns())
    # finished: only campaign 1 is closed, and it won -> 100%
    assert s["finished_win_rate"] == 1.0
    # sold-today: campaign 1 pnl_mtm>0 (win), campaign 2 pnl_mtm<0 (loss) -> 50%
    assert s["soldtoday_win_rate"] == 0.5
    assert s["n_open"] == 1
    assert s["n_campaigns"] == 2
    # avg % per win = pct_return of the single closed winner
    assert abs(s["avg_pct_per_win"] - 0.005) < 1e-9


def test_all_closed_winners_gap_is_zero():
    df = pd.DataFrame([
        dict(campaign_id=1, open_at_end=False, pnl_realized=100.0, pnl_mtm=100.0, pct_return=0.01),
        dict(campaign_id=2, open_at_end=False, pnl_realized=50.0, pnl_mtm=50.0, pct_return=0.02),
    ])
    s = portfolio_summary_stats(df)
    assert s["finished_win_rate"] == 1.0
    assert s["soldtoday_win_rate"] == 1.0   # no hidden losses -> numbers agree
    assert s["n_open"] == 0


def test_no_closed_campaigns_finished_rate_is_nan():
    df = pd.DataFrame([
        dict(campaign_id=1, open_at_end=True, pnl_realized=-10.0, pnl_mtm=5.0, pct_return=-0.1),
    ])
    s = portfolio_summary_stats(df)
    assert math.isnan(s["finished_win_rate"])
    assert math.isnan(s["avg_pct_per_win"])


def test_empty_campaigns_no_crash():
    df = pd.DataFrame(columns=["campaign_id", "open_at_end", "pnl_realized", "pnl_mtm", "pct_return"])
    s = portfolio_summary_stats(df)
    assert s["n_campaigns"] == 0
    assert s["n_open"] == 0
    assert math.isnan(s["finished_win_rate"])
    assert math.isnan(s["soldtoday_win_rate"])
    assert math.isnan(s["avg_pct_per_win"])
