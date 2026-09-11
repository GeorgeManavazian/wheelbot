import pandas as pd
import pytest
from src.engine_v2.options.wheel import WheelConfig, WheelResult, Trade
from src.engine_v2.options.chain import Contract
from src.engine_v2.regime.autopsy import campaign_regimes, autopsy_table

def _closes(trend="up"):
    # 400 bdays; rising steadily -> uptrend after warmup
    idx = pd.bdate_range("2023-01-02", periods=400)
    base = pd.Series(range(400), index=idx, dtype=float)
    px = 100 + base * (0.1 if trend == "up" else -0.1)
    return px.clip(lower=5).rename("X")

def _t(date, action, cid, strike=100.0):
    return Trade(pd.Timestamp(date), action,
                 Contract("X", pd.Timestamp("2024-06-21"), strike, "P"),
                 1, 1.0, 0.0, cid)

def test_campaigns_tagged_and_grouped():
    trades = [
        _t("2024-05-01", "SELL_PUT", 1), _t("2024-05-08", "CLOSE_PUT", 1),
        _t("2024-05-09", "SELL_PUT", 2), _t("2024-05-10", "ROLL_CLOSE", 2),
        _t("2024-05-10", "ROLL_OPEN", 2), _t("2024-05-20", "PUT_EXPIRED", 2),
    ]
    res = WheelResult(pd.Series(dtype=float), trades, 0.0, 0)
    cfg = WheelConfig(commission_per_contract=0.0)
    up = _closes("up")
    tagged = campaign_regimes(res, cfg, market_closes=up, ticker_closes=up)
    assert set(tagged["market_trend"]) == {"uptrend"}
    assert set(tagged["ticker_trend"]) == {"uptrend"}
    grouped = autopsy_table(tagged, by="market")
    assert grouped.iloc[0]["n_campaigns"] == 2
    assert grouped.iloc[0]["n_rolls"] == 1

def test_open_campaign_excluded_from_win_rate():
    trades = [_t("2024-05-01", "SELL_PUT", 1)]   # never closed
    res = WheelResult(pd.Series(dtype=float), trades, 0.0, 100,
                      residual_settled=True)
    cfg = WheelConfig(commission_per_contract=0.0)
    up = _closes("up")
    tagged = campaign_regimes(res, cfg, up, up)
    grouped = autopsy_table(tagged)
    assert grouped["n_campaigns"].sum() == 0
    assert grouped.attrs["n_open_excluded"] == 1

def test_warmup_open_date_is_unknown():
    trades = [_t("2023-03-01", "SELL_PUT", 1), _t("2023-03-08", "CLOSE_PUT", 1)]
    res = WheelResult(pd.Series(dtype=float), trades, 0.0, 0)
    cfg = WheelConfig(commission_per_contract=0.0)
    up = _closes("up")
    tagged = campaign_regimes(res, cfg, up, up)
    assert tagged.iloc[0]["market_trend"] == "unknown"

def test_tag_uses_strictly_prior_day_state():
    # campaign opens the day the trend flips at the close: tag must be the
    # PRIOR day's state (a trader placing the trade cannot know today's close).
    from src.engine_v2.regime.state import regime_series
    from src.engine_v2.regime.autopsy import _tag
    up = _closes("up")
    states = regime_series(up)
    d = states.index[-1]                     # last state day
    fake = states.copy()
    fake.loc[d, "trend"] = "downtrend"       # pretend the close flipped today
    assert _tag([d], fake, "trend") == [states.iloc[-2]["trend"]]

def test_tag_stale_state_is_unknown():
    from src.engine_v2.regime.state import regime_series
    from src.engine_v2.regime.autopsy import _tag
    states = regime_series(_closes("up"))
    late = states.index[-1] + pd.Timedelta(days=30)   # after series ends
    assert _tag([late], states, "trend") == ["unknown"]
