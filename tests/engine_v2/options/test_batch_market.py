import pandas as pd
from src.engine_v2.options.market import BatchMarket


def _chain(dates_strikes):
    # rows: (date, expiry, strike, underlying)
    rows = [{"date": pd.Timestamp(d), "expiry": pd.Timestamp(e), "strike": s,
             "right": "P", "dte": 7, "delta": -0.2, "bid": 1.0, "ask": 1.1,
             "mid": 1.05, "underlying": u} for (d, e, s, u) in dates_strikes]
    return pd.DataFrame(rows)


def _mkt():
    ch = _chain([("2021-01-04", "2021-01-15", 30.0, 33.0),
                 ("2021-01-05", "2021-01-15", 30.0, 34.0)])
    chains = {"GDX": ch}
    # regime_states: a tiny frame indexed by date
    states = {"GDX": pd.DataFrame({"trend": ["chop"], "vol": ["normal"],
              "vol_pctile": [0.5]}, index=[pd.Timestamp("2021-01-04")])}
    return BatchMarket(chains, states, {"XOP": pd.Timestamp("2020-07-01")}, ["GDX"])


def test_chain_returns_days_rows_or_none():
    m = _mkt()
    assert len(m.chain("GDX", pd.Timestamp("2021-01-04"))) == 1
    assert m.chain("GDX", pd.Timestamp("2021-01-06")) is None   # no rows that day


def test_spot_present_and_fallback():
    m = _mkt()
    assert m.spot("GDX", pd.Timestamp("2021-01-05"), 99.0) == 34.0
    assert m.spot("GDX", pd.Timestamp("2021-01-06"), 99.0) == 99.0   # absent -> fallback


def test_settle_price_on_or_before_expiry():
    m = _mkt()
    # last underlying at/before 2021-01-15 is the 2021-01-05 row = 34.0
    assert m.settle_price("GDX", pd.Timestamp("2021-01-15")) == 34.0
    # before any data -> None
    assert m.settle_price("GDX", pd.Timestamp("2020-01-01")) is None


def test_regime_row_strictly_prior_day():
    m = _mkt()
    row = m.regime_row("GDX", pd.Timestamp("2021-01-05"))   # prior day 01-04 exists
    assert row is not None and row["trend"] == "chop"
    assert m.regime_row("GDX", pd.Timestamp("2021-01-04")) is None  # nothing strictly before


def test_default_clean_start_carries_the_a10e_fossil_fences():
    """A10e sweep (2026-08-02): the in-sample chains carry unadjusted
    corporate actions -- AAPL 4:1 2020-08-31, AMZN 20:1 2022-06-06, META
    symbol reuse until 2022-06-09 (a $12 Metaverse ETF, not Facebook), NVDA
    4:1 2021-07-20 + 10:1 2024-06-10, XOP 1:4 2020-03-31 (the original).
    Measured on solo runs: NVDA booked $363k of phantom split losses; META's
    P&L rode a x15 wrong-instrument jump. Every fence must sit AFTER its
    ticker's last fossil; dropping any entry un-fences fiction."""
    import pandas as pd
    from src.engine_v2.options.portfolio import DEFAULT_CLEAN_START
    last_fossil = {"XOP": "2020-03-31", "AAPL": "2020-08-31",
                   "AMZN": "2022-06-06", "META": "2022-06-09",
                   "NVDA": "2024-06-10"}
    for tk, fossil in last_fossil.items():
        assert tk in DEFAULT_CLEAN_START, f"A10e: {tk} fence missing"
        assert DEFAULT_CLEAN_START[tk] > pd.Timestamp(fossil), \
            f"A10e: {tk} fence does not clear its last fossil {fossil}"


def test_eligible_respects_clean_start():
    m = _mkt()
    assert m.eligible("GDX", pd.Timestamp("2021-01-04")) is True     # no clean_start for GDX
    m2 = BatchMarket({"XOP": _chain([("2020-01-02","2020-01-15",30.0,30.0)])},
                     {"XOP": pd.DataFrame()}, {"XOP": pd.Timestamp("2020-07-01")}, ["XOP"])
    assert m2.eligible("XOP", pd.Timestamp("2020-01-02")) is False   # before clean start
    assert m2.eligible("XOP", pd.Timestamp("2020-08-01")) is True


def test_universe_property():
    assert _mkt().universe == ["GDX"]
