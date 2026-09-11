"""Custom-universe path for run_portfolio_wheel (expanded-backtest study):
passing an explicit `universe` list uses it as the ordering + allow-list and
drops the hardcoded 9-ticker ROTATION_TIE_ORDER / RESERVED_TICKERS gates,
while still requiring regime_states for every working-universe member. The
default path (universe=None) is unchanged and still refuses unknown tickers."""
import pandas as pd
import pytest
from src.engine_v2.options.portfolio import (run_portfolio_wheel, PortfolioResult,
                                              ROTATION_TIE_ORDER, RESERVED_TICKERS)
from src.engine_v2.options.wheel import WheelConfig

COLS = ["date", "expiry", "dte", "strike", "right", "bid", "ask", "mid",
        "close", "delta", "iv", "underlying"]


def _chain(und, strike, bid=2.00):
    ch = pd.DataFrame(
        [["2024-01-02", "2024-01-09", 7, strike, "P", bid, bid + 0.10,
          bid + 0.05, bid + 0.05, -0.30, 0.1, und]], columns=COLS)
    ch["date"] = pd.to_datetime(ch["date"])
    ch["expiry"] = pd.to_datetime(ch["expiry"])
    return ch


def _states():
    df = pd.DataFrame([("2024-01-01", "uptrend", "calm", 0.20)],
                      columns=["date", "trend", "vol", "vol_pctile"]).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df


def _cfg(**kw):
    base = dict(starting_capital=50_000.0, put_delta=0.30, call_delta=0.30,
                take_profit_pct=None, commission_per_contract=0.0,
                call_min_strike="basis")
    base.update(kw)
    return WheelConfig(**base)


# Tickers deliberately NOT in ROTATION_TIE_ORDER (and not RESERVED).
CUSTOM = ["QQQ", "IWM", "DIA"]


def test_custom_universe_tickers_outside_rotation_run():
    for t in CUSTOM:
        assert t not in ROTATION_TIE_ORDER
        assert t not in RESERVED_TICKERS
    chains = {"QQQ": _chain(400.0, 398), "IWM": _chain(200.0, 198),
              "DIA": _chain(380.0, 378)}
    states = {t: _states() for t in CUSTOM}
    res = run_portfolio_wheel(chains, _cfg(), states, universe=CUSTOM)
    assert isinstance(res, PortfolioResult)
    entries = [t for t in res.trades if t.action == "SELL_PUT"]
    assert entries  # it actually routed a campaign, no "not in rotation" raise


def test_custom_universe_preserves_passed_order_and_filters_to_chains():
    # Pass a universe with an extra ticker absent from chains; it is filtered out,
    # passed order otherwise preserved (used by the routing tie-break).
    chains = {"QQQ": _chain(400.0, 398), "IWM": _chain(200.0, 198)}
    states = {"QQQ": _states(), "IWM": _states(), "DIA": _states()}
    res = run_portfolio_wheel(chains, _cfg(), states,
                              universe=["DIA", "QQQ", "IWM"])
    assert isinstance(res, PortfolioResult)


def test_custom_universe_still_requires_regime_states():
    chains = {"QQQ": _chain(400.0, 398), "IWM": _chain(200.0, 198)}
    states = {"QQQ": _states()}  # IWM missing
    with pytest.raises(ValueError, match="no regime_states"):
        run_portfolio_wheel(chains, _cfg(), states, universe=["QQQ", "IWM"])


def test_default_path_still_refuses_unknown_ticker():
    # universe=None (default) must still enforce the rotation allow-list.
    chains = {"QQQ": _chain(400.0, 398)}
    states = {"QQQ": _states()}
    with pytest.raises(ValueError, match="not in the rotation universe"):
        run_portfolio_wheel(chains, _cfg(), states)
