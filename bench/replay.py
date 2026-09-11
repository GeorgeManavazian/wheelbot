"""`bench replay <variant>` -- re-decide real live days under a variant's rules.

THE GAP THIS FILLS. Backtest chains had no `open_interest` column, so the A2b
finding -- that `liq_max_rel_spread=0.10` refused 25 of 27 good-to-rent names on
CHEAPNESS rather than illiquidity -- could not have been found by any backtest.
It was found by pointing at the bot's own live 2026-08-10 chains and counting.
`bench/policy.toml` therefore makes replay a required gate for any variant that
touches liquidity, fills, spreads or size: a mechanic that has only ever met
backtest chains has not met the thing that broke last time.

WHAT IT COMPARES, AND WHY IT IS A SINGLE FLAT DAY. Each replayed day is stepped
from a FLAT state -- all cash, no positions -- for both configs, against the
exact chain snapshot the bot saw at 15:45 ET that day. That isolates the
variable. A path-dependent replay would answer "how would the last three weeks
have gone", which is a different and much noisier question: two configs that
diverge on day 1 are comparing different portfolios by day 3, and the entry
decision -- the thing a gate variant changes -- gets buried under position
history. The decision diff is the measurement; the backtest is where path
matters.

WHAT IT NEEDS, AND WHY A MISSING DAY IS AN ERROR. The chain snapshots are
written by the live runner at `data/live/chains/<date>.json`. Neither the
snapshots nor the `live` package this module imports to read them ship in this
repository, so `bench replay` needs the live runner alongside. A day with no snapshot is REPORTED AND SKIPPED
with the date named, never silently dropped -- "replayed 10 days" over a window
containing 14 is the same lie as "531 tickers" over a run that kept 16.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
import os
from pathlib import Path

import pandas as pd

from bench import config as bench_config
from bench import fingerprint
from bench import variant as bench_variant

REPLAY_DIR = Path(__file__).resolve().parent / "replays"


class ReplayRefused(RuntimeError):
    """The replay cannot be run honestly. Message says what is missing."""


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def snapshot_dir() -> Path:
    """Where the chain snapshots live. Honours WHEELBOT_STATE_DIR, the same
    variable the live runner uses, so pointing the bench at a copy of its
    state is one env var and not a second convention."""
    return Path(os.environ.get("WHEELBOT_STATE_DIR") or "data/live") / "chains"


def available_days(since=None, until=None) -> list[pd.Timestamp]:
    d = snapshot_dir()
    if not d.is_dir():
        raise ReplayRefused(
            f"no chain snapshots at {d}.\n"
            f"They are written by the live runner, one JSON file per trading "
            f"day, and are not part of this repository.\n"
            f"    WHEELBOT_STATE_DIR=... bench replay  # point at a copy of them")
    out = []
    for p in sorted(d.glob("*.json")):
        try:
            day = pd.Timestamp(p.stem)
        except ValueError:
            continue
        if since is not None and day < pd.Timestamp(since):
            continue
        if until is not None and day > pd.Timestamp(until):
            continue
        out.append(day)
    return out


def _closes_for_replay(tickers, upto: pd.Timestamp) -> dict:
    """Split-adjusted close history per ticker, long enough for a regime state.

    SPLICED FROM TWO SOURCES, deliberately and declared:

      * the bench closes store (built off the options chains) for history, which
        ends when the ThetaData subscription lapsed ~2026-07-25;
      * the underlying price carried on every chain snapshot row, for the days
        after that.

    `regime_series` needs ~273 trading days before it emits its first row (the
    vol percentile lands well past the 200-day warmup), so replaying a recent
    day on snapshot-derived prices alone would leave every name
    entry-ineligible and the replay would report "no entries" as though that
    were a finding."""
    from bench.loaders import market as mkt
    from live.chain_store import load_chain_snapshot
    base = mkt.load_closes()

    tail: dict[str, dict] = {}
    for day in available_days(until=upto):
        chains = load_chain_snapshot(day)
        if not chains:
            continue
        for tk, df in chains.items():
            if df is None or df.empty or "underlying" not in df.columns:
                continue
            px = pd.to_numeric(df["underlying"], errors="coerce").dropna()
            if len(px):
                tail.setdefault(tk, {})[day] = float(px.iloc[0])

    out = {}
    for tk in tickers:
        s = base.get(tk)
        extra = tail.get(tk) or {}
        if extra:
            add = pd.Series(extra).sort_index()
            s = pd.concat([s, add]) if s is not None else add
            s = s[~s.index.duplicated(keep="last")].sort_index()
        if s is not None and len(s):
            out[tk] = s.rename(tk)
    return out


# ---------------------------------------------------------------------------
# The replay
# ---------------------------------------------------------------------------

@dataclass
class DayResult:
    day: str
    entries: dict = field(default_factory=dict)     # arm -> [contract strings]
    refusals: dict = field(default_factory=dict)    # arm -> {reason: [tickers]}
    candidates: int = 0
    dte_coverage: dict = field(default_factory=dict)


def _contract_key(t) -> str:
    c = getattr(t, "contract", None)
    if c is None:
        return str(t)
    return (f"{getattr(c, 'ticker', '?')} {getattr(c, 'right', '?')} "
            f"{getattr(c, 'strike', '?')} {getattr(c, 'expiry', '?')}")


def _refusals(warnings) -> dict:
    out: dict[str, list] = {}
    for w in warnings or []:
        if not isinstance(w, (tuple, list)) or len(w) < 2:
            continue
        reason = str(w[1])
        if not reason.startswith("entry_gated"):
            continue
        tk = str(w[2]) if len(w) > 2 else "?"
        out.setdefault(reason, []).append(tk)
    return {k: sorted(set(v)) for k, v in sorted(out.items())}


def _dte_coverage(chains: dict, cfg) -> dict:
    """Does the snapshot even contain the band this config would select from?

    The snapshot was pulled for the LIVE `target_dte`. A variant that moves it
    may be selecting from rows that were never pulled, and would then look
    conservative for a reason that has nothing to do with its rules."""
    from src.engine_v2.options.select import derived_band
    lo, hi = derived_band(cfg.target_dte)
    have = 0
    for df in chains.values():
        if df is None or df.empty or "dte" not in df.columns:
            continue
        if df["dte"].between(lo, hi).any():
            have += 1
    return {"band": [int(lo), int(hi)], "tickers_with_band": have,
            "tickers_in_snapshot": len(chains)}


def replay_day(day: pd.Timestamp, arms: dict, universe, closes: dict,
               ops: dict, earnings=None) -> DayResult:
    from live.chain_store import load_chain_snapshot
    from live.market_live import LiveMarket
    from src.engine_v2.regime.state import weather_kwargs
    from src.engine_v2.options.portfolio import PortfolioState, step_one_day

    chains = load_chain_snapshot(day)
    if not chains:
        raise ReplayRefused(f"no usable chain snapshot for {day.date()}")

    res = DayResult(day=str(day.date()), candidates=len(chains))
    for arm_name, cfg in arms.items():
        res.dte_coverage[arm_name] = _dte_coverage(chains, cfg)
        market = LiveMarket(
            list(universe), [], day,
            closes_fn=lambda tk: closes[tk],
            chain_fn=lambda tk: chains[tk] if tk in chains
            else (_ for _ in ()).throw(KeyError(f"{tk} not in snapshot")),
            gate=weather_kwargs(cfg),
            earnings=earnings, iv_history=None)
        state = PortfolioState(cash=float(ops.get("capital", 100_000.0)),
                               positions=[])
        step = step_one_day(state, market, day, cfg,
                            selector=ops.get("selector", "chop"),
                            n_slots=int(ops.get("n_slots", 1)))
        res.entries[arm_name] = sorted(_contract_key(t) for t in step.trades
                                       if getattr(t, "action", "") in
                                       ("sell_put", "sell_call", "open"))
        res.refusals[arm_name] = _refusals(step.warnings)
    return res


def run(variant_name: str, since=None, until=None, save: bool = True,
        verbose: bool = True) -> dict:
    v = bench_variant.load(variant_name)
    base_name = v.base or bench_variant.MASTER_VARIANT
    ops = bench_config.ops_for(variant_name)
    capital = float(ops.get("capital", 100_000.0))

    arms = {
        "variant": bench_config.to_wheel_config(variant_name, ticker="SPY",
                                                starting_capital=capital),
        "base": bench_config.to_wheel_config(base_name, ticker="SPY",
                                             starting_capital=capital),
    }
    days = available_days(since, until)
    if not days:
        raise ReplayRefused(
            f"no chain snapshots in {since}..{until} under {snapshot_dir()}. "
            f"Widen the window, or point WHEELBOT_STATE_DIR at a store that has them.")

    from bench.universe import UNIVERSE
    closes = _closes_for_replay(list(UNIVERSE), max(days))
    earnings = None
    if any(c.earnings_blackout for c in arms.values()):
        from bench.loaders import market as mkt
        earnings = mkt.load_earnings()

    day_results, skipped = [], []
    for day in days:
        try:
            day_results.append(replay_day(day, arms, UNIVERSE, closes, ops,
                                          earnings))
        except ReplayRefused as e:
            skipped.append({"day": str(day.date()), "why": str(e)})
        if verbose:
            print(f"  replayed {day.date()}  "
                  f"variant {len(day_results[-1].entries['variant']) if day_results else 0} "
                  f"entries, base "
                  f"{len(day_results[-1].entries['base']) if day_results else 0}")

    n_var = sum(len(d.entries.get("variant", [])) for d in day_results)
    n_base = sum(len(d.entries.get("base", [])) for d in day_results)
    differed = sum(len(set(d.entries.get("variant", [])) ^
                       set(d.entries.get("base", []))) for d in day_results)

    report = {
        "schema": 1,
        "variant": variant_name,
        "base": base_name,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config_hash": fingerprint.config_hash(
            bench_config.knobs_for(variant_name)),
        "base_config_hash": fingerprint.config_hash(
            bench_config.knobs_for(base_name)),
        "engine": fingerprint.git_provenance(),
        "window": {"since": str(since) if since else None,
                   "until": str(until) if until else None,
                   "days_available": [str(d.date()) for d in days]},
        "n_days": len(day_results),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "n_variant_entries": n_var,
        "n_base_entries": n_base,
        "n_differed": differed,
        "days": [d.__dict__ for d in day_results],
        "method": ("flat single-day step per day, both configs against the same "
                   "15:45 ET chain snapshot; isolates the entry decision from "
                   "position history on purpose -- see the module docstring"),
        "closes_source": ("spliced: bench closes store for history + the "
                          "`underlying` column of each snapshot for days after "
                          "the ThetaData lapse (~2026-07-25)"),
    }
    if save:
        d = REPLAY_DIR / variant_name
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{report['config_hash']}.json"
        p.write_text(json.dumps(report, indent=1), encoding="utf-8")
        report["path"] = str(p)
        if verbose:
            print(f"  replay -> {fingerprint.rel(p)}")
    if verbose:
        print(render(report))
    return report


def for_variant(name: str, root: Path | None = None) -> list[dict]:
    d = Path(root or REPLAY_DIR) / name
    if not d.is_dir():
        return []
    out = []
    for p in d.glob("*.json"):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:                                   # noqa: BLE001
            continue
    return sorted(out, key=lambda r: r.get("created", ""), reverse=True)


def render(r: dict, width: int = 84) -> str:
    L = ["=" * width,
         f"  replay: {r['variant']}  vs  {r['base']}",
         "=" * width,
         f"  days replayed        {r['n_days']}"
         + (f"   ({r['n_skipped']} SKIPPED -- see below)" if r["n_skipped"] else ""),
         f"  entries base         {r['n_base_entries']}",
         f"  entries variant      {r['n_variant_entries']}",
         f"  decisions differed   {r['n_differed']}",
         "-" * width]
    agg: dict[str, dict[str, int]] = {}
    for d in r.get("days", []):
        for arm, reasons in (d.get("refusals") or {}).items():
            for reason, tks in reasons.items():
                agg.setdefault(reason, {}).setdefault(arm, 0)
                agg[reason][arm] += len(tks)
    if agg:
        L.append(f"  {'refusal reason':<32}{'base':>10}{'variant':>10}")
        for reason, by in sorted(agg.items()):
            L.append(f"  {reason:<32}{by.get('base',0):>10}"
                     f"{by.get('variant',0):>10}")
        L.append("-" * width)
    for s in r.get("skipped", []):
        L.append(f"  SKIPPED {s['day']}: {s['why']}")
    L.append("  Method: " + r.get("method", ""))
    L.append("  Closes: " + r.get("closes_source", ""))
    L.append("=" * width)
    return "\n".join(L)
