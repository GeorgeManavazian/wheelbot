"""The shared take-profit fill seam (A18). These tests pin each mode to the
rule its engine had before the seam existed -- they are behavior pins, not
aspirations. If one fails after an edit to fills.py, an engine's fill rule
changed, which is a strategy decision the owner has explicitly deferred."""
import numpy as np
import pandas as pd

from src.engine_v2.options.chain import Mark
from src.engine_v2.options.fills import try_take_profit, FillDecision
from src.engine_v2.options.wheel import WheelConfig, buy_cost

D = pd.Timestamp("2024-01-05")
EXP = pd.Timestamp("2024-01-12")


def _cfg(tp=0.60, comm=0.65):
    return WheelConfig(ticker="X", starting_capital=50_000.0, put_delta=0.30,
                       call_delta=0.30, take_profit_pct=tp,
                       commission_per_contract=comm, call_min_strike="basis")


def _bars(closes, hours=None):
    hours = hours or range(10, 10 + len(closes))
    return pd.DataFrame({"timestamp": [D + pd.Timedelta(hours=h) for h in hours],
                         "close": closes})


def test_quote_fill_matches_buy_cost_exactly():
    mark = Mark(0.30, 0.40, 0.35)          # thresh = 0.4 * 1.00 = 0.40, ask hits exactly
    dec = try_take_profit(mark=mark, credit=1.00, contracts=10, cfg=_cfg(),
                          day=D, expiry=EXP)
    assert dec.filled and dec.via == "quote"
    assert dec.price == 0.40
    assert dec.cost == buy_cost(mark, 10, _cfg())    # the one arithmetic
    assert dec.stamp is D


def test_quote_day_stamp_override():
    now = pd.Timestamp("2024-01-05 12:34")
    dec = try_take_profit(mark=Mark(0.1, 0.2, 0.15), credit=1.00, contracts=1,
                          cfg=_cfg(), day=D, expiry=EXP, day_stamp=now)
    assert dec.filled and dec.stamp is now


def test_quote_above_threshold_holds():
    dec = try_take_profit(mark=Mark(0.30, 0.41, 0.35), credit=1.00, contracts=1,
                          cfg=_cfg(), day=D, expiry=EXP)
    assert not dec.filled and dec.via == "no_fill"


def test_print_fills_next_valid_bar_not_crossing_bar():
    bars = _bars([0.35, 1.10, 0.20])       # bar0 crosses, fill at bar1's 1.10
    dec = try_take_profit(mark=None, credit=1.00, contracts=1, cfg=_cfg(),
                          day=D, expiry=EXP, bars=bars)
    assert dec.filled and dec.via == "print"
    assert dec.price == 1.10
    assert dec.stamp == D + pd.Timedelta(hours=11)


def test_print_skips_phantom_zero_bars():
    bars = _bars([0.35, 0.0, 1.10])        # 0.0 is no-trade, next VALID bar fills
    dec = try_take_profit(mark=None, credit=1.00, contracts=1, cfg=_cfg(),
                          day=D, expiry=EXP, bars=bars)
    assert dec.filled and dec.price == 1.10


def test_print_last_bar_cross_falls_through_to_quote():
    bars = _bars([2.00, 0.35])             # cross on the LAST bar: no next bar
    dec = try_take_profit(mark=Mark(0.30, 0.38, 0.34), credit=1.00, contracts=1,
                          cfg=_cfg(), day=D, expiry=EXP, bars=bars)
    assert dec.filled and dec.via == "quote" and dec.price == 0.38


def test_print_and_quote_genuinely_diverge():
    """The A18 divergence, named: the same day where the print rule fills and
    the quote rule refuses (bar prints cross; the book never does). This is
    the class of campaign (5.3%, 159 measured) the backtest wins that live
    can never close. The seam preserves the divergence; resolving it is the
    deferred strategy question."""
    bars = _bars([0.35, 0.50, 0.60])
    ask_never_crosses = Mark(0.50, 0.55, 0.52)   # thresh 0.40 < ask always
    via_print = try_take_profit(mark=ask_never_crosses, credit=1.00, contracts=1,
                                cfg=_cfg(), day=D, expiry=EXP, bars=bars)
    via_quote = try_take_profit(mark=ask_never_crosses, credit=1.00, contracts=1,
                                cfg=_cfg(), day=D, expiry=EXP, bars=None)
    assert via_print.filled and via_print.via == "print"
    assert not via_quote.filled


def test_guard_order_tp_disabled_before_threshold_arithmetic():
    # credit=None would raise in the threshold arithmetic; the disabled guard
    # must answer first (the A6 amendment lesson: guard order is load-bearing)
    for tp in (None, 1.0, 1.5):
        dec = try_take_profit(mark=Mark(0.1, 0.1, 0.1), credit=None, contracts=1,
                              cfg=_cfg(tp=tp), day=D, expiry=EXP)
        assert dec == FillDecision(False, via="tp_disabled")


def test_expiry_day_refused():
    dec = try_take_profit(mark=Mark(0.0, 0.01, 0.005), credit=1.00, contracts=1,
                          cfg=_cfg(), day=EXP, expiry=EXP)
    assert not dec.filled and dec.via == "expiry_day"


def test_no_mark_no_bars_no_fill():
    dec = try_take_profit(mark=None, credit=1.00, contracts=1, cfg=_cfg(),
                          day=D, expiry=EXP)
    assert not dec.filled and dec.via == "no_fill"


def test_print_cost_keeps_numpy_scalar_dtype():
    """Pre-existing, deliberate: the print path's cost is computed from the
    bar frame's numpy scalar, so engine cash goes np.float64 mid-run exactly
    as it did before the seam. Bit-identical values; changing the TYPE is a
    separate decision (it would alter repr/json behavior downstream)."""
    bars = _bars([0.35, 1.10])
    dec = try_take_profit(mark=None, credit=1.00, contracts=3, cfg=_cfg(),
                          day=D, expiry=EXP, bars=bars)
    assert isinstance(dec.cost, np.floating)
    assert isinstance(dec.price, float) and not isinstance(dec.price, np.floating)


def test_filled_contracts_is_full_size_on_both_paths():
    """A17: both current modes are instant-and-whole -- the seam must say so
    explicitly (filled_contracts == contracts), and refusals must leave the
    default so FillDecision equality pins stay valid."""
    quote = try_take_profit(mark=Mark(0.1, 0.2, 0.15), credit=1.00, contracts=7,
                            cfg=_cfg(), day=D, expiry=EXP)
    assert quote.filled and quote.filled_contracts == 7
    printed = try_take_profit(mark=None, credit=1.00, contracts=7, cfg=_cfg(),
                              day=D, expiry=EXP, bars=_bars([0.35, 1.10]))
    assert printed.filled and printed.filled_contracts == 7
    refused = try_take_profit(mark=None, credit=1.00, contracts=7, cfg=_cfg(),
                              day=D, expiry=EXP)
    assert not refused.filled and refused.filled_contracts == 0
