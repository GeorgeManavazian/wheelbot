"""A market order must cross a real bid-ask spread.

The orchestrator fed fill_order `bid == ask == close`, so the modelled spread was
identically zero and a round trip cost only square-root impact. Chan's rule (vault
note "Chan 06 Intraday and Microstructure"): a market order crosses the FULL spread
on entry AND exit. Schwab charges $0 commission on online ETF trades, so for the
equity phase the spread IS the trading cost.
"""
import pytest

from src.engine_v2.execution.sim import bid_ask, Order, fill_order


def test_bid_ask_brackets_the_mid_by_half_spread():
    bid, ask = bid_ask(close=100.0, half_spread_bps=1.0)
    assert ask == pytest.approx(100.0 * (1 + 1e-4))
    assert bid == pytest.approx(100.0 * (1 - 1e-4))
    assert bid < 100.0 < ask


def test_zero_spread_collapses_to_the_close():
    bid, ask = bid_ask(close=100.0, half_spread_bps=0.0)
    assert bid == ask == 100.0


def test_market_buy_pays_exactly_the_half_spread_over_mid_with_no_slippage():
    """Isolate spread from impact: slippage coefficients set to zero."""
    bid, ask = bid_ask(close=100.0, half_spread_bps=2.0)
    fill = fill_order(Order("SPY", "buy", "market", 10.0), bid=bid, ask=ask,
                      adv=1e12, slippage_a=0.0, slippage_b=0.0)
    assert fill.price == pytest.approx(100.0 * (1 + 2e-4))  # pays the ask, i.e. mid + 2bps


def test_market_sell_pays_the_half_spread_under_mid_with_no_slippage():
    bid, ask = bid_ask(close=100.0, half_spread_bps=2.0)
    fill = fill_order(Order("SPY", "sell", "market", 10.0), bid=bid, ask=ask,
                      adv=1e12, slippage_a=0.0, slippage_b=0.0)
    assert fill.price == pytest.approx(100.0 * (1 - 2e-4))  # hits the bid, i.e. mid - 2bps
