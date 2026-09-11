"""Everything a wheel run needs besides the chains: closes, regime states,
earnings, rates and solved IV history.

Absorbed from `scratchpad/run_wheel_grid_n1.py` and `wheel_grid_load.py`. Each
function keeps the reason it exists, because in every case the obvious call was
tried first and was silently wrong:

  closes  -- `regime.data.closes_for` resolves only 16 of the 531 names: it
             reads the 16-ETF fixture, and its chain fallback calls the dead
             `chain_path()`. Passing that through runs the whole bench on 16
             tickers while still printing 531.

  iv      -- built by the SOLVER, never from `chain["iv"]`. The only iv in the
             store is `vendor_iv`, on a different scale from every other IV in
             this project (2026-08-07 provenance ruling).

  earnings-- a missing calendar must RAISE, not read as "no prints". Both
             states present as an empty map, and collapsing them is how the
             blackout gate spent weeks reporting protection it was not giving.

MISSING PREREQUISITES RAISE WITH THE COMMAND THAT BUILDS THEM. Falling back
would answer a different question than the one the variant claims to ask, and
would do it while printing a full set of plausible trades.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from bench.loaders.rates import load_rate_series
from src.engine_v2.options.earnings import EarningsCalendar
from src.engine_v2.options.iv_rank import IVHistory
from src.engine_v2.regime.state import regime_series

RATES_CSV = Path("data/rates/dgs1mo.csv")
CLOSES_PARQUET = Path("data/live/grid_closes/closes.parquet")
IV_DIR = Path("data/live/grid_iv")
EARNINGS_DIR = Path("data/earnings")


class PrerequisiteMissing(FileNotFoundError):
    """A required input is absent. Carries the command that builds it."""


def load_rates(path: Path | None = None) -> pd.Series:
    p = Path(path or RATES_CSV)
    if not p.exists():
        raise PrerequisiteMissing(f"{p} missing -- the risk-free series the "
                                  f"solver and the normalizer both consume.")
    return load_rate_series(str(p))


def load_closes(path: Path | None = None) -> dict:
    """ticker -> split-adjusted daily closes.

    Built by `bench prep closes` (formerly `scratchpad/wheel_grid_closes.py`),
    which rebuilds the intended chain fallback off the chains store,
    split-adjusted, and truncates a series at a discontinuity no split explains
    -- FI (FISV->FI, 2023-06-07), META (FB->META, 2022-06-09) and COHR (II-VI
    took the Coherent name, 2022-09-08). Before those dates the ticker is a
    different instrument."""
    p = Path(path or CLOSES_PARQUET)
    if not p.exists():
        raise PrerequisiteMissing(
            f"{p} missing -- run `bench prep closes` first. Without it only 16 "
            f"of 531 names have a regime state and the weather gate silently "
            f"runs on an ETF-only universe.")
    df = pd.read_parquet(p)
    return {tk: pd.Series(g["close"].to_numpy(),
                          index=pd.to_datetime(g["date"].to_numpy())
                          ).sort_index().rename(tk)
            for tk, g in df.groupby("ticker")}


def regime_states(closes: dict, tickers=None) -> dict:
    """ticker -> regime state frame. Built here rather than cached: it is
    cheap, and a cached state frame is one more thing that can be stale
    relative to the closes it came from."""
    want = list(tickers) if tickers is not None else list(closes)
    out = {}
    for tk in want:
        s = closes.get(tk)
        if s is None or len(s) == 0:
            continue
        st = regime_series(s)
        if len(st):
            out[tk] = st
    return out


def load_earnings(directory: Path | None = None) -> EarningsCalendar:
    """The blackout calendar.

    RAISES if absent. `earnings_blackout=True` with no calendar is the zombie
    gate this project already shipped once: `earnings_dates()` returns None for
    every ticker, `in_blackout()` reads that as unknown, unknown allows, and the
    gate reports protection it is not giving."""
    d = Path(directory or EARNINGS_DIR)
    if not (d / "calendar.parquet").exists():
        raise PrerequisiteMissing(
            f"{d}/calendar.parquet missing -- the earnings blackout is ON in "
            f"the master config, and a blackout with no calendar is a gate that "
            f"reports protection it is not giving. Run the earnings puller, or "
            f"set earnings_blackout unset in the variant and declare it as a "
            f"divergence.")
    return EarningsCalendar.load(str(d))


def iv_history_path(put_delta: float, target_dte: int,
                    directory: Path | None = None) -> Path:
    d = Path(directory or IV_DIR)
    return d / f"d{put_delta}_dte{target_dte}.parquet"


def load_iv_history(put_delta: float, target_dte: int, closes: dict,
                    directory: Path | None = None) -> IVHistory:
    """The cell's SOLVED IV history, clipped to each ticker's usable closes.

    Clipping is one rule instead of three special cases: where a ticker's close
    series was truncated at a rename, its implied vol before that date belongs
    to a different instrument too."""
    p = iv_history_path(put_delta, target_dte, directory)
    if not p.exists():
        raise PrerequisiteMissing(
            f"{p} missing -- run `bench prep iv --delta {put_delta} --dte "
            f"{target_dte}` first. Falling back to an unranked run would "
            f"silently answer a different question than the variant asks.")
    df = pd.read_parquet(p)
    by, clipped = {}, 0
    for tk, g in df.groupby("ticker"):
        s = pd.Series(g["iv"].to_numpy(),
                      index=pd.to_datetime(g["date"].to_numpy())).sort_index()
        if tk in closes:
            first = closes[tk].index.min()
            if (s.index < first).any():
                s = s[s.index >= first]
                clipped += 1
        if len(s):
            by[tk] = s
    return IVHistory(by)


def spy_buy_hold(closes: dict, start, end, starting_capital: float) -> pd.Series:
    """SPY buy-and-hold over the run's own window, scaled to its capital.

    THE campaign benchmark (schema 2, 2026-08-17): the gate campaign's done
    condition is "2-yr P&L >= SPY buy-hold over the same window", and no
    scorecard could answer it -- the equal-weight `buy_hold` arm below has
    silently returned None on every card ever written (its try/except swallows
    the TypeError from passing a dict where `buy_hold_curve` wants one chain).

    RAISES when SPY closes are absent or empty in the window, in the house
    style: a benchmark the done-condition depends on must never quietly become
    a missing column three weeks later. Built from closes rather than chains so
    every tier can carry it -- the closes file covers the whole store."""
    s = closes.get("SPY")
    if s is None or len(s) == 0:
        raise PrerequisiteMissing(
            "SPY is not in the closes file -- the campaign benchmark cannot be "
            "computed. Rebuild with `bench prep closes`; the chain store "
            "carries SPY.")
    w = s.loc[pd.Timestamp(start):pd.Timestamp(end)]
    if len(w) < 2:
        raise PrerequisiteMissing(
            f"SPY closes have {len(w)} row(s) inside {start}..{end} -- no "
            f"benchmark can be computed over this window.")
    return starting_capital * w / float(w.iloc[0])


def buy_hold(chains: dict, starting_capital: float) -> pd.Series | None:
    """Equal-weight buy-and-hold over the same names and window -- the
    benchmark rule #3 of the vault manual insists on. None when the chains
    cannot produce one, never a flat line pretending to be a benchmark."""
    from src.engine_v2.options.report import buy_hold_curve
    try:
        return buy_hold_curve(chains, starting_capital)
    except Exception:                                       # noqa: BLE001
        return None
