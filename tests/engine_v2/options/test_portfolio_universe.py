import pandas as pd
import pytest
from src.engine_v2.options.portfolio import (run_portfolio_wheel, ROTATION_TIE_ORDER,
                                             RESERVED_TICKERS)
from src.engine_v2.options.wheel import WheelConfig


def test_dev_universe_is_the_nine():
    assert ROTATION_TIE_ORDER == ("SPY", "GDX", "SLV", "XOP",
                                  "AAPL", "AMZN", "NVDA", "META", "FB")


def test_reserved_ticker_refused():
    # A reserved ticker in the chains dict must raise, never run.
    cfg = WheelConfig(ticker="SPY", put_delta=0.20, call_delta=0.50,
                      target_dte=7, take_profit_pct=0.50, call_min_strike="basis")
    fake_chain = pd.DataFrame({"date": [pd.Timestamp("2021-01-04")],
                               "underlying": [100.0]})
    with pytest.raises(ValueError, match="reserved"):
        run_portfolio_wheel({"XBI": fake_chain}, cfg, {"XBI": pd.DataFrame()})


def test_reserved_constant_unchanged():
    assert RESERVED_TICKERS == ("XBI", "EEM", "EWZ", "TLT", "ARKK")
