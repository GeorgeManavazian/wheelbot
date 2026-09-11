"""`bench run <variant>` -- the paragraph, tested against the essay.

TWO ARMS, ONE SET OF BYTES. The variant and its base run over the SAME chains,
the same closes, the same regime states, the same earnings calendar, loaded
once. Every previous harness loaded its own, and two of them disagreed about
which loader was current, so their numbers were never actually comparable. Here
the only thing that differs between the arms is the config -- which is the only
thing a variant is.

WHAT IT REFUSES TO DO.

  * Run without a prerequisite. A missing IV history, closes file or earnings
    calendar raises with the command that builds it. The alternative -- running
    unranked, or on 16 of 531 names, or with a blind blackout gate -- produces a
    full set of plausible trades that answer a different question.

  * Hide a divergence. Anything the data cannot support (an OI gate on tickers
    with no OI column, a blackout on names the calendar does not know) is
    counted and written into the scorecard, not silently absorbed.

  * Pretend a solo mechanic works here. `run_portfolio_wheel` supports the
    plain+basis wheel only; roll, stop, regime gates and liquidate are solo
    mechanics. A variant that sets one gets told so by name.

MEMORY. 531 tickers x 4.1y at dte<=15 is ~6GB against 17GB of RAM, and ~8.3GB
at dte<=25. Both arms share one `chains` dict, so the peak is one copy, not two
-- which is what makes a two-arm comparison affordable at all. If a variant
changes `target_dte`, the load covers the wider of the two bands.
"""
from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bench import config as bench_config
from bench import fingerprint, scorecard as sc_mod, tiers
from bench import variant as bench_variant
from bench.loaders import chains as chain_loader
from bench.loaders import market


class RunRefused(RuntimeError):
    """The run cannot answer the question honestly. Message says why."""


#: rank_by values that read the solved IV history (bench prep iv). A module-
#: level constant, not an inline tuple, so a NEW rank_by that reads IV cannot
#: be added to the engine's validation list (portfolio.py) without someone
#: also being able to see -- and test -- that the bench driver still knows to
#: load it. Missing here is exactly the shape of the bug this fixes: the
#: engine refuses loudly ("needs an IV history"), but only at the LIVE tier,
#: because smoke's 12-ETF window is where it was first noticed.
RANK_BY_NEEDS_IV_HISTORY = ("iv_rank", "vrp", "vrp_viable")


@dataclass
class Arm:
    name: str            # "variant" | "base"
    variant: str
    knobs: dict
    cfg: object


def _build_arm(name: str, variant_name: str, capital: float) -> Arm:
    knobs = bench_config.knobs_for(variant_name)
    cfg = bench_config.to_wheel_config(variant_name, ticker="SPY",
                                       starting_capital=capital)
    return Arm(name=name, variant=variant_name, knobs=knobs, cfg=cfg)


def _refuse_solo_mechanics(arm: Arm) -> None:
    cfg = arm.cfg
    solo = []
    if cfg.roll_tested_puts:
        solo.append("roll_tested_puts")
    if cfg.put_stop_mult is not None:
        solo.append("put_stop_mult")
    if cfg.liquidate_assignment:
        solo.append("liquidate_assignment")
    if cfg.any_regime_gate:
        solo.append("a regime gate")
    if solo:
        raise RunRefused(
            f"variant `{arm.variant}` sets {', '.join(solo)}, which "
            f"run_portfolio_wheel does not support -- it runs the plain+basis "
            f"wheel only, and those are solo-wheel mechanics. The live bot is "
            f"the portfolio path, so a knob it cannot run is not a knob the "
            f"master config can ever take. Test it with the solo engine "
            f"(src/engine_v2/options/wheel.run_wheel) and say so in the "
            f"variant's note, or drop it.")


def _divergences(arms, chains: dict, earnings, universe) -> list[str]:
    """What the data could not honestly support. Written into the scorecard.

    A declared divergence is the difference between "this result is not the
    live bot" and "this result quietly was not the live bot"."""
    out: list[str] = []
    wants_oi = any(a.cfg.liq_min_open_interest is not None
                   or a.cfg.liq_max_pct_of_open_interest is not None
                   for a in arms)
    if wants_oi:
        # Per-ROW coverage, not per-ticker. A ticker whose OI store starts in
        # 2024-08 still has the column present for a 2022 window -- full of
        # NaN, which liquidity_ok refuses exactly as hard as an absent column.
        # Counting tickers would report 100% coverage on a run that is blind
        # for half its length. (Measured 2026-08-14: 86 of 534 tickers have OI
        # before 2024-08; the other 443 start there.)
        n_rows = n_have = 0
        no_col = 0
        for c in chains.values():
            n_rows += len(c)
            if "open_interest" not in c.columns:
                no_col += 1
                continue
            n_have += int(c["open_interest"].notna().sum())
        pct = (n_have / n_rows * 100.0) if n_rows else 0.0
        if pct < 99.0:
            out.append(
                f"open_interest is present on {pct:.1f}% of chain rows "
                f"({no_col} of {len(chains)} tickers have no column at all). "
                f"liquidity_ok REFUSES when a threshold is set and the field is "
                f"missing or NaN, so every one of the other {100-pct:.1f}% is "
                f"barred from entry -- on data availability, not on liquidity. "
                f"The OI store reaches near-full coverage only from 2024-08; "
                f"the `live` tier starts there for this reason. Visible in the "
                f"refusal counts as entry_gated_illiquid.")

    if any(a.cfg.earnings_blackout for a in arms) and earnings is not None:
        known = sum(1 for t in universe if earnings.dates(t) is not None)
        if known < len(universe):
            out.append(
                f"earnings calendar knows {known} of {len(universe)} universe "
                f"names; the other {len(universe) - known} read as UNKNOWN and "
                f"in_blackout ALLOWS on unknown. This is identical to what the "
                f"live bot does with the same file, so it is faithful rather "
                f"than a divergence -- but it is not full protection.")

    if any(a.cfg.rank_by == "iv_rank" for a in arms):
        out.append(
            "rank_by='iv_rank' reads a SOLVED IV history (bench prep iv), never "
            "the store's vendor_iv. Ranks the solver could not produce sort at "
            "NEUTRAL_IV_RANK, never last -- see iv_rank.py on why last would "
            "rebuild the veto that was measured and rejected on 2026-08-04.")
    return out


def _load_world(tier: tiers.Tier, max_dte: int, verbose=True):
    """Chains, closes, regime states and the surviving universe.

    Returns the DROP REASONS as well as the survivors. "531 tickers" printed
    over a run that silently kept 16 of them is the exact failure this
    accounting exists to make impossible."""
    say = print if verbose else (lambda *a, **k: None)
    rates = market.load_rates()
    closes = market.load_closes()
    say(f"  closes: {len(closes)} tickers (split-adjusted, from chains)")

    names = tier.tickers()
    dropped = {"no_chain": [], "no_closes": [], "no_state": []}
    chains: dict = {}
    t0 = time.time()
    for i, tk in enumerate(names, 1):
        ch = chain_loader.load_chain(tk, rates, tier.start, tier.end,
                                     max_dte=max_dte)
        if ch is None:
            dropped["no_chain"].append(tk)
            continue
        if tk not in closes:
            dropped["no_closes"].append(tk)
            continue
        chains[tk] = ch
        if verbose and i % 50 == 0:
            say(f"    loaded {i}/{len(names)}  "
                f"{chain_loader.store_mem_mb(chains)/1000:.1f} GB resident")
    say(f"  chains: {len(chains)} tickers, "
        f"{chain_loader.store_mem_mb(chains)/1000:.2f} GB, "
        f"{time.time()-t0:.0f}s")

    states = market.regime_states(closes, list(chains))
    for tk in list(chains):
        if tk not in states:
            dropped["no_state"].append(tk)
            chains.pop(tk, None)

    universe = sorted(chains)
    say(f"  universe: {len(universe)} of {len(names)} requested"
        f"  (dropped: {len(dropped['no_chain'])} no chain, "
        f"{len(dropped['no_closes'])} no closes, "
        f"{len(dropped['no_state'])} no regime state)")
    return chains, closes, states, universe, dropped


def check_arm_fills(res, chains, label: str, out_dir=None, verbose=True):
    """Fill realism for one arm: the per-trade frame, its summary, and a CSV.

    Kept beside the run rather than in a scratchpad script because the trade
    log does not survive the run -- only the scorecard is persisted -- and a
    liquidity question that has to be answered from the trade log otherwise has
    nowhere to be asked from. `bench doctor` refuses one-off harnesses for
    exactly this reason: the answer belongs in the bench."""
    from bench import realism
    df = realism.check_fills(res.trades, chains)
    summary = realism.summarise(df)
    if out_dir is not None and not df.empty:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"fills_{label}.csv"
        df.to_csv(p, index=False)
        if verbose:
            print(f"  fill log -> {fingerprint.rel(p)}")
    if verbose:
        print(realism.render(summary, label))
    return df, summary


def _run_arm(arm: Arm, chains, states, universe, ops, earnings, closes,
             verbose=True):
    from src.engine_v2.options.portfolio import run_portfolio_wheel
    iv = None
    if arm.cfg.rank_by in RANK_BY_NEEDS_IV_HISTORY:
        # `vrp` divides this same solved history by realized vol, and
        # `vrp_viable`'s tier 1 reads the same VRP statistic (batch 5 arm 2),
        # so both need exactly the same artifact -- and would otherwise raise
        # in the engine guard rather than load it.
        iv = market.load_iv_history(arm.cfg.put_delta, arm.cfg.target_dte,
                                    closes)
    t0 = time.time()
    try:
        res = run_portfolio_wheel(
            chains, arm.cfg, states,
            selector=ops.get("selector", "chop"),
            n_slots=int(ops.get("n_slots", 1)),
            universe=universe, earnings=earnings, iv_history=iv)
    except ValueError as e:
        raise RunRefused(
            f"arm `{arm.name}` ({arm.variant}) was refused by the engine: {e}"
        ) from e
    secs = time.time() - t0
    m = sc_mod.metrics_from_result(res, float(ops.get("capital", 100_000.0)),
                                   n_slots=int(ops.get("n_slots", 1)))
    m["secs"] = round(secs, 1)
    if verbose:
        print(f"  {arm.name:<8} {arm.variant:<28} "
              f"tot {m['total_return']*100:+7.2f}%  "
              f"sharpe {m['sharpe'] if m['sharpe'] is not None else float('nan'):5.2f}  "
              f"maxDD {m['max_drawdown']*100:6.2f}%  "
              f"camp {m['campaigns']:5d}  ({secs:.0f}s)")
    return res, m


def run(variant_name: str, tier_name: str | None = None,
        base_override: str | None = None, save: bool = True,
        verbose: bool = True, notes: str = "",
        check_fills: bool = False) -> sc_mod.Scorecard:
    """Run a variant against its base and return the scorecard."""
    v = bench_variant.load(variant_name)
    base_name = base_override or v.base or bench_variant.MASTER_VARIANT
    ops = bench_config.ops_for(variant_name)
    tier = tiers.get(tier_name or ops.get("tier", tiers.DEFAULT_TIER))
    capital = float(ops.get("capital", 100_000.0))

    arms = [_build_arm("variant", variant_name, capital),
            _build_arm("base", base_name, capital)]
    for a in arms:
        _refuse_solo_mechanics(a)

    max_dte = max(tiers.max_dte_for(a.cfg.target_dte) for a in arms)
    if verbose:
        print(f"\n{variant_name}  vs  {base_name}   [{tier.name}] {tier.blurb}")
        print(f"  window {tier.start.date()} -> {tier.end.date()}   "
              f"n_slots {ops.get('n_slots')}   capital {capital:,.0f}   "
              f"max_dte {max_dte}")
        if not tier.is_evidence:
            print("  NOTE: this tier is stamped NOT EVIDENCE -- it cannot "
                  "satisfy the promotion policy.")

    chains, closes, states, universe, dropped = _load_world(tier, max_dte,
                                                            verbose)
    if not universe:
        raise RunRefused(
            f"no ticker in tier `{tier.name}` survived loading "
            f"({len(dropped['no_chain'])} had no chain in "
            f"{tier.start.date()}..{tier.end.date()}). Check "
            f"{chain_loader.STORE} -- the store is month-partitioned and the "
            f"window may fall outside it.")

    earnings = None
    if any(a.cfg.earnings_blackout for a in arms):
        earnings = market.load_earnings()

    t_all = time.time()
    card = sc_mod.new(variant_name, base_name, tier.name, tier.is_evidence)
    fill_check = {}
    for a in arms:
        res, m = _run_arm(a, chains, states, universe, ops, earnings, closes,
                          verbose)
        card.arms[a.name] = m
        if check_fills:
            out_dir = Path("results/bench") / variant_name / tier.name
            _, fs = check_arm_fills(res, chains, a.name, out_dir, verbose)
            fill_check[a.name] = fs
        del res

    bh = market.buy_hold(chains, capital)
    if bh is not None and len(bh) > 1:
        card.arms["buy_hold"] = sc_mod.metrics_from_equity(bh, capital)
    # THE campaign benchmark (schema 2): the done condition is "P&L >= SPY
    # buy-hold, same window, same capital", and the equal-weight arm above has
    # silently been None on every card ever written. Report-only -- delta()
    # never reads it -- and loud on absence, like every other prerequisite.
    spy = market.spy_buy_hold(closes, tier.start, tier.end, capital)
    card.arms["spy_buy_hold"] = sc_mod.metrics_from_equity(spy, capital)

    card.wall_secs = time.time() - t_all
    card.notes = notes
    card.divergences = _divergences(arms, chains, earnings, universe)
    card.conditions = {
        "start": str(tier.start.date()), "end": str(tier.end.date()),
        "n_slots": int(ops.get("n_slots", 1)), "capital": capital,
        "selector": ops.get("selector", "chop"), "max_dte": max_dte,
        "n_tickers_requested": len(tier.tickers()),
        "n_tickers_loaded": len(universe),
        "dropped": {k: len(x) for k, x in dropped.items()},
    }
    if fill_check:
        # Onto the CARD, not just the console: a liquidity claim that cannot be
        # re-read from the committed evidence is an anecdote again. AFTER the
        # conditions dict is built -- until 2026-08-17 this line ran before it
        # and was clobbered two statements later, so no card ever carried the
        # check it printed.
        card.conditions["fill_check"] = fill_check
    card.provenance = {
        "engine": fingerprint.git_provenance(),
        "config_hash": {"variant": fingerprint.config_hash(arms[0].knobs),
                        "base": fingerprint.config_hash(arms[1].knobs)},
        "data": fingerprint.data_fingerprint(universe, tier.start, tier.end),
        "env": fingerprint.environment(),
    }

    del chains
    gc.collect()

    if save:
        p = card.save()
        if verbose:
            print(f"\n  scorecard -> {fingerprint.rel(p)}")
    if verbose:
        print()
        print(sc_mod.render(card))
    return card
