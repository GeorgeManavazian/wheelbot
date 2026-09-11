"""Engine-shaped option chains from the month-partitioned store.

Absorbed from `scratchpad/wheel_grid_load.py` (2026-08-09), which was the last
correct loader and the only one aware that `options.data.chain_path()` points at
a layout with no files left in it.

ONE CHANGE FROM THE ORIGINAL, and it matters: the original declared
`open_interest` unavailable, because the flat-file OI endpoint was 403 before
2026-08-02 and the per-expiration one needed ~19s per 5-day range. It has since
been pulled -- `data/options/open_interest/` is 525MB on disk (commit fb196a1) --
and `_join_open_interest` below reads it. So a bench run can now carry ALL THREE
liquidity legs, which means the live FROZEN config is backtestable exactly for
the first time. Where the OI store has no data for a ticker-month the column is
left ABSENT rather than zero-filled, so an OI-gated config fails loudly instead
of silently refusing every entry.

THE COLUMN THAT IS STILL DELIBERATELY MISSING:

  iv -- NOT carried. `IVHistory.from_chains` reads `chain["iv"]`, and the only
        iv in the store is `vendor_iv`. Splicing that in would put a
        vendor-scale number where every other IV in this project is
        solver-scale -- the second unmarked scale the 2026-08-07 provenance
        ruling exists to prevent. The bench builds IV history through the
        solver (`bench/loaders/market.py:load_iv_history`) and passes it as
        `iv_history=`; `from_chains` is never called.

DTE FILTER. `select.derived_band(target) = (max(5, target-2), target+4)`, so the
largest DTE any arm can select is 25 at `target_dte=21`. Rows beyond
`dte <= MAX_DTE` cannot be chosen and are dropped at load -- the difference
between a run that fits in 17GB of RAM and one that does not.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ENGINE_COLUMNS = ["date", "expiry", "dte", "strike", "right", "delta",
                  "bid", "ask", "mid", "underlying", "rate", "div_yield"]

RIGHTS = {"CALL": "C", "PUT": "P"}


def _rate_column(dates, rate):
    """`rate` as one value per observation date.

    A scalar is broadcast. A Series (the published rate curve) is reindexed onto
    the observation dates and carried FORWARD: FRED skips bank holidays on which
    the options market can still trade, and the last published rate is the
    prevailing one on such a day. Before the first published rate the result is
    NaN, deliberately: a missing rate is not a zero rate.
    """
    if not isinstance(rate, pd.Series):
        return float(rate)
    published = pd.Series(rate.values, index=pd.to_datetime(rate.index)).sort_index()
    wanted = pd.DatetimeIndex(pd.unique(dates))
    return dates.map(published.reindex(published.index.union(wanted)).ffill())


def normalize_vendor_frame(df, rate, div_yield):
    """The vendor's rows in the shape select_contract and the solver expect.

    The store's schema is the vendor's, not the engine's: CALL/PUT rights, string
    dates, no dte/mid column, and greeks named vendor_iv/vendor_delta so they can
    never be mistaken for ours. `rate` and `div_yield` are INPUTS, not columns of
    the store (ThetaData ships neither); they are passed explicitly rather than
    defaulted because a silent zero would be a second, unmarked scale.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=ENGINE_COLUMNS)
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"]),
        "expiry": pd.to_datetime(df["expiration"]),
        "strike": df["strike"].astype(float),
        "right": df["right"].map(RIGHTS),
        # The vendor's delta, used ONLY to choose which row to record.
        "delta": df["vendor_delta"].astype(float),
        "bid": df["bid"].astype(float),
        "ask": df["ask"].astype(float),
        "underlying": df["underlying_price"].astype(float),
    })
    out["dte"] = (out["expiry"] - out["date"]).dt.days
    out["mid"] = (out["bid"] + out["ask"]) / 2.0
    out["rate"] = _rate_column(out["date"], rate)
    out["div_yield"] = float(div_yield)
    return out[ENGINE_COLUMNS]

STORE = Path("data/options/chains")
OI_STORE = Path("data/options/open_interest")
OI_SRC = "open_interest_asof_prior_close"

MAX_DTE = 30          # band max is 25 at target_dte=21; 30 leaves mark headroom

# normalize_vendor_frame's output plus the columns the liquidity gate needs.
KEEP = ["date", "expiry", "dte", "strike", "right", "delta",
        "bid", "ask", "mid", "underlying", "rate", "div_yield", "volume"]

# Read only what the normalizer and the gates consume. The store carries 20
# columns; open/high/low/close/count/bid_size/ask_size/timestamp/
# underlying_timestamp/vendor_iv are dead weight.
_SOURCE_COLS = ["date", "expiration", "strike", "right", "bid", "ask",
                "underlying_price", "vendor_delta", "volume"]


def months_in_window(start: pd.Timestamp, end: pd.Timestamp) -> set[str]:
    s, e = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    return {d.strftime("%Y-%m") for d in
            pd.date_range(s, e, freq="MS").union([s, e])}


def load_chain(ticker: str, rates, start, end,
               max_dte: int = MAX_DTE,
               store: Path | None = None) -> pd.DataFrame | None:
    """One ticker's engine-shaped chain over [start, end], or None.

    Month files outside the window are never opened -- the store is 7.1GB and
    reading 61 months to keep 50 is the difference between a run that fits in
    RAM and one that does not."""
    store = Path(store) if store else STORE
    want = months_in_window(start, end)
    paths = [p for p in sorted((store / ticker).glob("*.parquet"))
             if p.stem in want]
    if not paths:
        return None

    frames = []
    for p in paths:
        try:
            raw = pd.read_parquet(p, columns=_SOURCE_COLS)
        except Exception:                                   # noqa: BLE001
            continue
        if raw.empty:
            continue
        vol = raw["volume"].astype("float32").to_numpy()
        out = normalize_vendor_frame(raw, rates, 0.0)
        out["volume"] = vol
        out = out[out["dte"].between(0, max_dte)]
        if not out.empty:
            frames.append(out)

    if not frames:
        return None
    ch = pd.concat(frames, ignore_index=True)
    ch = ch[(ch["date"] >= pd.Timestamp(start)) & (ch["date"] <= pd.Timestamp(end))]
    if ch.empty:
        return None
    ch = _join_open_interest(ch, ticker, want)
    return _shrink(ch)


def _join_open_interest(ch: pd.DataFrame, ticker: str,
                        want: set[str]) -> pd.DataFrame:
    """Annotate the chain with open interest, or leave the column absent.

    The chain store carries `volume` but never `open_interest`; the OI store is
    a SEPARATE tree (`scripts/pull_open_interest.py`) joined here at read time
    rather than by rewriting 32,091 parquets.

    ABSENT, NOT ZERO, when there is no data. `liquidity_ok` and
    `liquidity_size_cap` both treat a missing/NaN field as REFUSE when the
    matching threshold is set, so a half-populated column would silently
    suppress entries; no column at all makes an OI-gated config fail loudly.

    The stored value is OPRA's morning message, which describes the PREVIOUS
    session's close -- correct for a decision taken during this session, and the
    source column name says so. It is renamed to plain `open_interest` because
    that is the name the engine's gate looks for."""
    d = OI_STORE / ticker
    paths = [p for p in sorted(d.glob("*.parquet")) if p.stem in want] \
        if d.is_dir() else []
    if not paths:
        return ch
    oi = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    if oi.empty or OI_SRC not in oi.columns:
        return ch
    oi["expiry"] = pd.to_datetime(oi["expiration"])
    oi["date"] = pd.to_datetime(oi["date"])
    oi["strike"] = oi["strike"].astype("float64")
    # The OI feed spells the side CALL/PUT; the engine's frame uses C/P.
    oi["right"] = oi["right"].astype(str).str[0]
    oi = (oi[["date", "expiry", "strike", "right", OI_SRC]]
          .rename(columns={OI_SRC: "open_interest"})
          .drop_duplicates(subset=["date", "expiry", "strike", "right"]))
    out = ch.merge(oi, on=["date", "expiry", "strike", "right"], how="left")
    out["open_interest"] = pd.to_numeric(out["open_interest"], errors="coerce")
    return out


def _shrink(ch: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32 on the wide numeric columns. Selection compares
    deltas and strikes, and prices feed a P&L in dollars-and-cents; float32
    carries ~7 significant digits, more than a $0.01 tick on a four-figure
    strike needs. `rate` and `div_yield` stay float64 because the solver
    consumes them."""
    for c in ("strike", "delta", "bid", "ask", "mid", "underlying", "volume",
              "open_interest"):
        if c in ch.columns:
            ch[c] = ch[c].astype("float32")
    ch["dte"] = ch["dte"].astype("int16")
    ch["right"] = ch["right"].astype("category")
    # KEEP is an allow-list, so a column added upstream is DROPPED unless named
    # -- which once silently disarmed the open-interest join. Carried only when
    # present, so a run against a store with no OI still fails loudly at the
    # gate rather than quietly here.
    cols = KEEP + (["open_interest"] if "open_interest" in ch.columns else [])
    return ch[cols]


def load_many(tickers, rates, start, end, max_dte: int = MAX_DTE,
              progress=None) -> dict:
    """{ticker: chain} for every ticker that has data. Tickers with no rows in
    the window are omitted, never present-but-empty: `run_portfolio_wheel`
    treats an empty chain and a missing one differently."""
    out: dict = {}
    for i, tk in enumerate(tickers, 1):
        ch = load_chain(tk, rates, start, end, max_dte)
        if ch is not None and not ch.empty:
            out[tk] = ch
        if progress:
            progress(i, len(tickers), tk, ch)
    return out


def mem_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / 1e6


def store_mem_mb(chains: dict) -> float:
    return sum(mem_mb(c) for c in chains.values())
