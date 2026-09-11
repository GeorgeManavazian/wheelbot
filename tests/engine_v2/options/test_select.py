import os
import pandas as pd
import pytest
from src.engine_v2.options.select import (
    select_contract, derived_band, option_mark, intrinsic_value, expiry_underlying)
from src.engine_v2.options.chain import Contract

# NOTE: the plan named fixtures/spy_options_small.parquet, but that fixture only
# carries DTE 0-1 expiries and derived_band's floor clamps at 5, so no target_dte
# can ever select from it. spy_wheel_cycle.parquet (also a plan fixture) has a
# 42-DTE expiry on its first date -> target_dte=40 (band 38..43) is satisfiable.
FIX = "fixtures/spy_wheel_cycle.parquet"
pytestmark = pytest.mark.skipif(not os.path.exists(FIX), reason="fixture not built")

def _chain():
    return pd.read_parquet(FIX)

def test_intrinsic_value():
    assert intrinsic_value("P", 100, 90) == 10
    assert intrinsic_value("P", 100, 110) == 0
    assert intrinsic_value("C", 100, 110) == 10
    assert intrinsic_value("C", 100, 90) == 0

def test_derived_band():
    assert derived_band(7) == (5, 11)
    assert derived_band(11) == (9, 15)
    assert derived_band(30) == (28, 34)
    assert derived_band(5) == (5, 9)      # floor clamps at 5


def test_the_band_spans_a_whole_weekly_expiry_cycle():
    """WHY the ceiling is +4 and not +3 (2026-08-09).

    Most names list Friday expiries only -- a 7-day grid. For any given weekday
    the available DTEs are one residue class mod 7 (Mon 4/11/18, Tue 3/10/17,
    Wed 2/9/16, Thu 1/8/15, Fri 0/7/14), so a band of SEVEN consecutive integers
    always contains exactly one. Six does not, and which weekday falls through
    is an accident of the target:

        target 11, old band (9, 14): Thursday sees 8 and 15. Neither fits.
        target  7, old band (5, 10): Monday   sees 4 and 11. Neither fits.

    Measured on 5 years of real chains, KO recorded 8 Thursdays at target 11 and
    9 Mondays at target 7, against ~250 for every other weekday. The bot was
    silently unable to enter most of the universe one day in five, at EVERY
    target DTE, and nothing in the logs said so.

    Widened on the CEILING only: the floor is the expiry-stub guard, and raising
    the ceiling never shortens a trade. In a 15-ticker A/B it added 230 days, all
    Thursdays, and changed ZERO existing selections.
    """
    for target in range(7, 60):
        lo, hi = derived_band(target)
        assert hi - lo + 1 >= 7, f"target {target}: band {(lo, hi)} misses a weekday"

def test_select_contract_expiry_first_then_strike():
    ch = _chain()
    d = sorted(ch["date"].unique())[0]
    c = select_contract(ch, d, "P", 0.30, target_dte=40, root="SPY")
    assert c is not None and c.right == "P" and c.root == "SPY"
    lo, hi = derived_band(40)
    day = ch[(ch["date"] == d) & (ch["right"] == "P")]
    dtes = day.groupby("expiry")["dte"].first()
    eligible = dtes[(dtes >= lo) & (dtes <= hi)]
    # 1) chosen expiry is the eligible one nearest 40
    err = (eligible - 40).abs()
    assert (pd.Timestamp(c.expiry) in eligible[err == err.min()].index)
    # 2) within that expiry, strike is nearest-delta
    e = day[day["expiry"] == c.expiry]
    best = (e["delta"].abs() - 0.30).abs().min()
    got = (e[e["strike"] == c.strike]["delta"].abs() - 0.30).abs().iloc[0]
    assert got == pytest.approx(best, abs=1e-9)

def test_select_contract_deterministic():
    ch = _chain()
    d = sorted(ch["date"].unique())[0]
    a = select_contract(ch, d, "P", 0.30, target_dte=40, root="SPY")
    b = select_contract(ch, d, "P", 0.30, target_dte=40, root="SPY")
    assert a == b

def test_select_returns_none_when_band_misses():
    ch = _chain()
    d = sorted(ch["date"].unique())[0]
    # candidates exist on the date, but no expiry falls in the derived band
    assert select_contract(ch, d, "P", 0.30, target_dte=9998, root="SPY") is None

def test_select_contract_sits_out_when_band_empty():
    ch = _chain()
    d = sorted(ch["date"].unique())[0]
    # target so small the band [5,8] may or may not exist in fixture; force miss:
    ch2 = ch[ch["dte"] > 200]
    assert select_contract(ch2, d, "P", 0.30, target_dte=7, root="SPY") is None

def test_select_contract_root_propagates():
    ch = _chain()
    d = sorted(ch["date"].unique())[0]
    c = select_contract(ch, d, "P", 0.30, target_dte=40, root="GDX")
    assert c.root == "GDX"

def test_option_mark_and_absent():
    ch = _chain()
    d = sorted(ch["date"].unique())[0]
    c = select_contract(ch, d, "P", 0.30, target_dte=40, root="SPY")
    m = option_mark(ch, d, c)
    assert m is not None and m.mid == pytest.approx((m.bid + m.ask) / 2)
    absent = Contract("SPY", c.expiry, 1.0, "P")
    assert option_mark(ch, d, absent) is None

def test_expiry_underlying():
    ch = _chain()
    d = sorted(ch["date"].unique())[0]
    c = select_contract(ch, d, "P", 0.30, target_dte=40, root="SPY")
    u = expiry_underlying(ch, c)
    # the fixture includes the contract's expiry date (2024-02-16), so assert the real value
    assert u is not None
    assert u == pytest.approx(499.51, abs=0.01)

# ---- repair pass: cross-expiry min_strike fallback ----

_COLS = ["date","expiry","dte","strike","right","bid","ask","mid","close","delta","iv","underlying"]

def _rows_chain(rows):
    ch = pd.DataFrame(rows, columns=_COLS)
    ch["date"] = pd.to_datetime(ch["date"]); ch["expiry"] = pd.to_datetime(ch["expiry"])
    return ch

def test_min_strike_falls_back_to_other_inband_expiry():
    # best expiry (dte 7) has no strike >= 470; the dte-9 expiry does. The old
    # code returned None; the fallback must find the 470 in the farther expiry.
    rows = [
        ["2024-01-02","2024-01-09",7,460,"C",1.00,1.10,1.05,1.05,0.30,0.1,465.0],
        ["2024-01-02","2024-01-11",9,470,"C",0.60,0.70,0.65,0.65,0.20,0.1,465.0],
    ]
    ch = _rows_chain(rows)
    c = select_contract(ch, pd.Timestamp("2024-01-02"), "C", 0.30, 7, "SPY", min_strike=470.0)
    assert c is not None and c.strike == 470.0

def test_min_strike_no_inband_expiry_qualifies_returns_none():
    rows = [
        ["2024-01-02","2024-01-09",7,460,"C",1.00,1.10,1.05,1.05,0.30,0.1,465.0],
        ["2024-01-02","2024-01-11",9,465,"C",0.60,0.70,0.65,0.65,0.20,0.1,465.0],
    ]
    ch = _rows_chain(rows)
    assert select_contract(ch, pd.Timestamp("2024-01-02"), "C", 0.30, 7, "SPY",
                           min_strike=470.0) is None

# ---- repair pass amendment 2026-07-13b: roll destination extends time ----

def test_roll_destination_same_strike_beyond_held_expiry():
    from src.engine_v2.options.select import select_roll_contract
    # held 460P exp Jan-09; Jan-08 not beyond; Jan-15 (ext 6) and Jan-17
    # (ext 8) are both inside the tenor band; anchor Jan-16 -> tie on error 1
    # -> longer-dated Jan-17 wins.
    rows = [
        ["2024-01-04","2024-01-08",4,460,"P",3.00,3.10,3.05,3.05,-0.30,0.1,468.0],
        ["2024-01-04","2024-01-15",11,460,"P",3.20,3.30,3.25,3.25,-0.30,0.1,468.0],
        ["2024-01-04","2024-01-17",13,460,"P",4.20,4.30,4.25,4.25,-0.30,0.1,468.0],
    ]
    ch = _rows_chain(rows)
    c = select_roll_contract(ch, pd.Timestamp("2024-01-04"), "P", 460.0,
                             pd.Timestamp("2024-01-09"), 7, "SPY")
    assert c is not None and c.expiry == pd.Timestamp("2024-01-17")
    assert c.strike == 460.0

def test_roll_destination_skips_expiry_missing_the_strike():
    from src.engine_v2.options.select import select_roll_contract
    # nearest-to-anchor in-band expiry (Jan-17) lacks the 460 strike -> fall to
    # Jan-15, which carries it.
    rows = [
        ["2024-01-04","2024-01-15",11,460,"P",3.20,3.30,3.25,3.25,-0.30,0.1,468.0],
        ["2024-01-04","2024-01-17",13,455,"P",4.20,4.30,4.25,4.25,-0.30,0.1,468.0],
    ]
    ch = _rows_chain(rows)
    c = select_roll_contract(ch, pd.Timestamp("2024-01-04"), "P", 460.0,
                             pd.Timestamp("2024-01-09"), 7, "SPY")
    assert c is not None and c.expiry == pd.Timestamp("2024-01-15")

def test_roll_destination_rejects_out_of_band_tenor():
    from src.engine_v2.options.select import select_roll_contract
    # only listing with the strike is 16 days beyond the held expiry — outside
    # derived_band(7)=(5,10) extension -> no roll destination.
    rows = [
        ["2024-01-04","2024-01-25",21,460,"P",5.20,5.30,5.25,5.25,-0.30,0.1,468.0],
    ]
    ch = _rows_chain(rows)
    assert select_roll_contract(ch, pd.Timestamp("2024-01-04"), "P", 460.0,
                                pd.Timestamp("2024-01-09"), 7, "SPY") is None

def test_roll_destination_none_when_nothing_beyond():
    from src.engine_v2.options.select import select_roll_contract
    rows = [
        ["2024-01-04","2024-01-08",4,460,"P",3.00,3.10,3.05,3.05,-0.30,0.1,468.0],
    ]
    ch = _rows_chain(rows)
    assert select_roll_contract(ch, pd.Timestamp("2024-01-04"), "P", 460.0,
                                pd.Timestamp("2024-01-09"), 7, "SPY") is None


def test_select_contract_ignores_mark_only_rows():
    """A row spliced in purely to MARK a held leg must never become tradeable.

    live/held_legs.py adds the legs the bounded chain cannot see, so the
    take-profit can act on them. Those rows are pulled by OCC symbol for a
    position we already hold -- they are not part of the chain the bot
    surveyed, and selecting one means entering a contract the bot never saw.
    Seen live on 2026-07-31 across accounts: account B sold a DOW
    99.5P that existed in its candidate set only because account A held it."""
    import pandas as pd
    from src.engine_v2.options.select import select_contract
    d, exp = pd.Timestamp("2026-07-31"), pd.Timestamp("2026-08-21")
    cols = ["date", "expiry", "strike", "right", "dte", "delta",
            "bid", "ask", "mid", "underlying", "held_only"]
    rows = [
        {"date": d, "expiry": exp, "strike": 99.0, "right": "P", "dte": 21,
         "delta": -0.24, "bid": 1.0, "ask": 1.2, "mid": 1.1, "underlying": 103.0,
         "held_only": False},
        # the spliced leg: a far better delta match to a 0.30 target
        {"date": d, "expiry": exp, "strike": 99.5, "right": "P", "dte": 21,
         "delta": -0.30, "bid": 1.3, "ask": 1.5, "mid": 1.4, "underlying": 103.0,
         "held_only": True},
    ]
    chain = pd.DataFrame(rows, columns=cols)
    c = select_contract(chain, d, "P", 0.30, 21, "DOW")
    assert c is not None
    assert c.strike == 99.0, "selected a mark-only row"


def test_select_contract_still_works_without_the_held_only_column():
    """Every batch chain on disk predates the column; absence means 'tradeable'."""
    import pandas as pd
    from src.engine_v2.options.select import select_contract
    d, exp = pd.Timestamp("2026-07-31"), pd.Timestamp("2026-08-21")
    cols = ["date", "expiry", "strike", "right", "dte", "delta",
            "bid", "ask", "mid", "underlying"]
    chain = pd.DataFrame([{"date": d, "expiry": exp, "strike": 99.0, "right": "P",
                           "dte": 21, "delta": -0.30, "bid": 1.0, "ask": 1.2,
                           "mid": 1.1, "underlying": 103.0}], columns=cols)
    assert select_contract(chain, d, "P", 0.30, 21, "DOW").strike == 99.0


def test_option_mark_still_finds_a_mark_only_row():
    """The whole point: unselectable, but markable."""
    import pandas as pd
    from src.engine_v2.options.select import option_mark
    from src.engine_v2.options.chain import Contract
    d, exp = pd.Timestamp("2026-07-31"), pd.Timestamp("2026-08-21")
    chain = pd.DataFrame([{"date": d, "expiry": exp, "strike": 99.5, "right": "P",
                           "dte": 21, "delta": -0.30, "bid": 1.3, "ask": 1.5,
                           "mid": 1.4, "underlying": 103.0, "held_only": True}])
    mk = option_mark(chain, d, Contract("DOW", exp, 99.5, "P"))
    assert mk is not None and mk.ask == 1.5
