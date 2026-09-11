"""The scorecard: one run's verdict, in a schema, committed to git.

WHY THIS FILE IS COMMITTED AND `results/` IS NOT. `results/` is in `.gitignore`.
Every verdict this project has ever computed -- the ten-arm grid, the ranker
width sweep, the gate ablation -- lives there and is one `rm -rf` from gone,
with no record of which engine or which data produced it. A scorecard is small
(a few KB of JSON), so it lives in the repo beside the variant it judges. The
heavy artifacts -- trade logs, equity curves -- stay in `results/`, where being
disposable is correct.

WHAT MAKES IT EVIDENCE RATHER THAN A NUMBER. Three stamps and one flag:

  config_hash  the variant's knobs at run time. If the variant has been edited
               since, this scorecard describes a config that no longer exists,
               and `is_stale_for()` says so. Stale evidence is not evidence --
               that is the whole mechanism by which `bench promote` cannot be
               satisfied with an old, favourable run.
  engine sha   plus `dirty`. A scorecard from a dirty tree cannot be re-run
               by anyone, so the default policy refuses it.
  data_fp      the parquet files in scope, by path and size.
  is_evidence  false for the smoke tier. A 12-ETF six-month run is a smoke test
               and is stamped as one, permanently, in the file.

JUDGED ON WHAT. `sharpe` and `ret_per_maxdd` lead, deliberately, and
`total_return` is printed but never sorted on. Delta behaves as a leverage dial
in this engine -- the 0.40-delta arms top the return column by construction --
which is exactly how the original sweep table became unreadable.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 2 (2026-08-17, gate-campaign batch 1): adds the `spy_buy_hold` arm (the
# campaign's done condition is P&L >= SPY buy-hold and schema 1 could not
# answer it) and the below-basis covered-call counters (the tier2-deep arms'
# auto-reject is unverifiable without them). Schema-1 cards still load; they
# simply lack the new keys, which is itself the honest record.
SCHEMA = 2
SCORECARD_DIR = Path(__file__).resolve().parent / "scorecards"

#: Printed in this order. `total_return` sits below the risk-adjusted pair on
#: purpose -- see the module docstring.
METRIC_ORDER = ("sharpe", "ret_per_maxdd", "total_return", "cagr",
                "max_drawdown", "worst_year", "pnl", "campaigns",
                "assignments", "assignment_rate",
                "days_flat", "days_uncovered", "n_trades",
                "shares_sold", "sold_per_assignment",
                # Below-basis covered-call writes, added 2026-08-17 -- see
                # below_basis_call_writes().
                "calls_written_below_basis", "campaigns_with_below_basis_call",
                # Capacity, added 2026-08-16 -- see slot_economics().
                "slot_fill_pct", "slotdays_assigned_pct",
                "pnl_per_slotday_clean", "pnl_per_slotday_assigned",
                "idle_slot_days", "pool_depth_mean", "pool_starved_pct",
                # The funnel above the instrumented gates -- see slot_economics.
                "scan_no_chain_mean", "scan_weather_mean",
                "scan_no_contract_mean",
                # rank_by="vrp_viable"'s tier-1 share -- see slot_economics.
                "rank_viable_share_mean")

#: Metrics where a LARGER number is better. Used only for arrow rendering --
#: the bench does not decide what wins, because the promotion bar is the
#: owner's to set (see bench/policy.toml).
HIGHER_IS_BETTER = {"sharpe", "ret_per_maxdd", "total_return", "cagr",
                    "pnl", "campaigns", "worst_year",
                    "slot_fill_pct", "pnl_per_slotday_clean",
                    "pnl_per_slotday_assigned", "pool_depth_mean"}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _finite(x) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def year_by_year(equity) -> dict:
    out = {}
    for y, g in equity.groupby(equity.index.year):
        if len(g) > 1:
            out[str(int(y))] = float(g.iloc[-1] / g.iloc[0] - 1)
    return out


def metrics_from_equity(equity, starting_capital: float) -> dict:
    """The metric set every arm reports. Computed from the equity curve alone,
    so a buy-hold benchmark and a wheel run are measured identically."""
    from src.engine_v2.backtest import metrics_simple as m
    ppy = m.infer_periods_per_year(equity.index)
    rets = equity.pct_change().fillna(0.0)
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    dd = float(m.max_drawdown(equity))
    by_year = year_by_year(equity)
    return {
        "total_return": total,
        "cagr": _finite(m.cagr(equity, ppy)),
        "sharpe": _finite(m.sharpe(rets, ppy)),
        "max_drawdown": dd,
        "ret_per_maxdd": _finite(total / abs(dd)) if dd else None,
        "pnl": float(equity.iloc[-1] - starting_capital),
        "final_equity": float(equity.iloc[-1]),
        "n_days": int(len(equity)),
        "by_year": by_year,
        "worst_year": min(by_year.values()) if by_year else None,
    }


#: Actions that leave the campaign still holding something. A campaign whose
#: LAST action is one of these never finished, so its P&L is not yet knowable
#: and it is excluded from the per-slot-day figures rather than counted as a
#: zero. `CLOSE_CALL` and `CALL_EXPIRED` are both in here for the reason the
#: take-profit block warns about: ending the call leaves you holding the STOCK.
_HOLDING_AFTER = {"SELL_PUT", "SELL_CALL", "ASSIGNED", "ROLL_OPEN",
                  "CLOSE_CALL", "CALL_EXPIRED"}


def _campaign_table(trades, equity_index, starting_cash: float | None = None) -> list:
    """One row per campaign: when it held a slot, whether it was assigned, and
    the cash it moved.

    Cash comes from DIFFERENCES in `cash_after` down the trade log rather than
    from price x size, because only the difference carries the friction
    ($0.65 commission + $0.05 fees per contract per side) that decides whether
    a short-dated trade was worth doing. It is exact while `cash_yield == 0`,
    which is settled policy on this bot -- idle collateral earns nothing, so
    nothing else moves cash between trades.
    """
    if not trades or equity_index is None or len(equity_index) == 0:
        return []
    import pandas as pd
    idx = pd.DatetimeIndex(equity_index)
    rows: dict = {}
    # Seeded with the account's opening cash, so the FIRST campaign's credit is
    # counted like every other. Left as None the first trade has nothing to
    # difference against and campaign 1 silently books zero -- which is exactly
    # the kind of quiet wrong number this file exists to stop.
    prev_cash = _finite(starting_cash)
    for t in trades:
        cid = int(getattr(t, "campaign_id", 0) or 0)
        d = pd.Timestamp(getattr(t, "date"))
        action = str(getattr(t, "action", ""))
        cash_after = _finite(getattr(t, "cash_after", None))
        contract = getattr(t, "contract", None)
        ticker = str(getattr(contract, "root", "") or "")
        r = rows.setdefault(cid, {"campaign": cid, "ticker": ticker,
                                  "opened": d, "closed": d, "assigned": False,
                                  "cash": 0.0, "last_action": action})
        r["closed"] = max(r["closed"], d)
        r["last_action"] = action
        if action == "ASSIGNED":
            r["assigned"] = True
        if cash_after is not None:
            if prev_cash is not None:
                r["cash"] += cash_after - prev_cash
            prev_cash = cash_after
    for r in rows.values():
        # Half-open [opened, closed): the closing session hands the slot back,
        # and a same-session round trip still used one slot-day.
        lo = int(idx.searchsorted(r["opened"], side="left"))
        hi = int(idx.searchsorted(r["closed"], side="left"))
        # Unfinished is decided by the last ACTION, never by the last date: a
        # campaign holding shares stops trading, so its final trade can sit
        # months before the end of the run.
        r["open_at_end"] = r["last_action"] in _HOLDING_AFTER
        if r["open_at_end"]:
            hi = len(idx)
        r["slot_days"] = max(1, hi - lo)
    return sorted(rows.values(), key=lambda r: r["campaign"])


def slot_economics(result, n_slots: int | None,
                   starting_capital: float | None = None) -> dict:
    """How much of the machine each kind of campaign occupies, and what it
    earns per day of that occupancy.

    WHY THIS EXISTS. Until 2026-08-16 the scorecard could say how many
    campaigns opened and never how much of the bot was BUSY, so the standing
    claim that "a gate costs turnover by leaving slots idle" had no number
    under it -- and it turned out to be the wrong mechanism. The entry loop
    rescans all 530 names for every slot, so a refusal promotes the next name;
    it cannot idle a slot unless the whole pool is empty, which is what
    `pool_starved_pct` now counts. What refusals really do is change WHICH
    names get assigned, and an assigned campaign holds its slot for months.

    On the exported best_run: assigned campaigns were 20% of the count, 77% of
    all slot-time, and earned -$1.02 per slot-day against +$25.11 for the
    campaigns that stayed clean. That is the economics of the strategy and it
    was invisible on every scorecard ever written.
    """
    out: dict = {
        "idle_slot_days": int(getattr(result, "idle_slot_days", 0) or 0),
        "pool_starved_days": int(getattr(result, "pool_starved_days", 0) or 0),
    }
    sessions = int(getattr(result, "pool_depth_sessions", 0) or 0)
    # Per ROUTING SESSION, not per run: a run with more free slots scans more
    # often, and a total would confound how thin the funnel is with how often
    # the bot looked down it.
    funnel = (("scan_names_mean", "scan_names"),
              ("scan_no_chain_mean", "drop_no_chain"),
              ("scan_weather_mean", "drop_weather"),
              ("scan_no_contract_mean", "drop_no_contract"))
    if sessions:
        pool_depth_sum = getattr(result, "pool_depth_sum", 0) or 0
        out["pool_depth_mean"] = float(pool_depth_sum / sessions)
        out["pool_starved_pct"] = float(out["pool_starved_days"] / sessions)
        for key, attr in funnel:
            out[key] = float((getattr(result, attr, 0) or 0) / sessions)
        # Batch 5 Arm 2 (2026-08-22): rank_by="vrp_viable"'s tier-1 share of
        # the pool -- the mechanism clause ("non-trivially populated,
        # REPORTED") reads this field. Ratio of sums across sessions, not a
        # mean of daily ratios, same style as pool_depth_mean above. Zero for
        # every other rank_by (rank_viable_sum stays 0), so this is safe to
        # always compute.
        rank_viable_sum = getattr(result, "rank_viable_sum", 0) or 0
        out["rank_viable_share_mean"] = (
            float(rank_viable_sum / pool_depth_sum) if pool_depth_sum else None)
    else:
        out["pool_depth_mean"] = None
        out["pool_starved_pct"] = None
        out["rank_viable_share_mean"] = None
        for key, _ in funnel:
            out[key] = None

    equity = getattr(result, "equity", None)
    camps = _campaign_table(list(result.trades or []),
                            None if equity is None else equity.index,
                            starting_capital)
    if not camps:
        return out
    n_days = int(len(result.equity))
    occupied = sum(c["slot_days"] for c in camps)
    assigned_days = sum(c["slot_days"] for c in camps if c["assigned"])
    out.update({
        "campaigns_open_at_end": sum(1 for c in camps if c["open_at_end"]),
        "slotdays_occupied": occupied,
        "slotdays_assigned": assigned_days,
        "slotdays_assigned_pct": (assigned_days / occupied) if occupied else None,
    })
    if n_slots:
        out["slot_fill_pct"] = occupied / float(n_days * n_slots)
    # Unfinished campaigns are excluded from BOTH sides of the ratio: their
    # shares are still on the book, so their P&L is not yet a number. The count
    # above is reported so the size of the exclusion is visible.
    for label, want in (("clean", False), ("assigned", True)):
        done = [c for c in camps if c["assigned"] is want and not c["open_at_end"]]
        days = sum(c["slot_days"] for c in done)
        out[f"campaigns_{label}"] = len(done)
        out[f"pnl_per_slotday_{label}"] = (
            sum(c["cash"] for c in done) / days if days else None)
    return out


#: Actions whose cash_after difference is a PREMIUM flow -- what the engine
#: accumulates into pos["premium"] (sell proceeds and buy-back costs, friction
#: included). Assignment/call-away/share-sale cash is stock, not premium.
_PREMIUM_ACTIONS = {"SELL_PUT", "SELL_CALL", "CLOSE_PUT", "CLOSE_CALL"}


def below_basis_call_writes(trades, starting_cash: float,
                            mult: int = 100) -> tuple[int, int]:
    """(covered calls written below net basis, campaigns that had one).

    WHY. `exit-ladder-on` was rejected 2026-08-17 because tier 2 wrote fat
    0.80-delta calls BELOW net basis on the routine case -- selling away the
    recovery the wheel depends on. The tier2-deep arms' pre-registered
    auto-reject is "below-basis writes on more than a third of assigned
    campaigns", and no card could count the writes. This makes the reject
    checkable from the card instead of from a trade log that `results/` can
    lose.

    RECONSTRUCTED FROM THE TRADE LOG, deliberately: the engine at the pinned
    batch sha stays byte-identical. The definition is the engine's own
    (portfolio._net_basis): assignment strike minus premium banked this
    campaign per share, with premium followed through `cash_after` differences
    exactly as `_campaign_table` does (exact while cash_yield == 0, settled
    policy). The floor check in portfolio.py runs BEFORE the write books its
    own credit, so the strike is tested against the PRE-write premium here too.

    `mult` is the contract multiplier -- 100 on every config this repo has
    ever run (WheelConfig default; the account block pins it). A trade without
    `cash_after` leaves the premium ledger unchanged rather than guessing.

    KNOWN EDGE CASE (measured batch 1, 2026-08-17): the FROZEN base arm
    prints calls_written_below_basis = 1 on the live tier even though the
    engine's floor forbids the write. It is a net-basis-ACCRUAL artifact of
    this reconstruction, not the mechanic firing: the engine's floor is
    checked against pos["premium"] at write time inside the step, while this
    ledger accrues the same premium from cash_after differences that include
    each fill's friction -- a cents-level divergence that can push a
    written-at-the-floor strike epsilon below the reconstructed net basis.
    Read a base-arm count of ~1 as zero; the counter's job is the tier-2
    family's per-campaign RATE, where the signal is 3-9 campaigns, not 1.
    """
    prev_cash = _finite(starting_cash)
    st: dict[int, dict] = {}
    n_calls = 0
    camps: set[int] = set()
    for t in trades:
        action = str(getattr(t, "action", ""))
        cid = int(getattr(t, "campaign_id", 0) or 0)
        s = st.setdefault(cid, {"basis": None, "shares": 0, "premium": 0.0})
        cash_after = _finite(getattr(t, "cash_after", None))
        delta = None
        if cash_after is not None:
            if prev_cash is not None:
                delta = cash_after - prev_cash
            prev_cash = cash_after
        c = getattr(t, "contract", None)
        n = int(getattr(t, "contracts", 0) or 0)
        strike = _finite(getattr(c, "strike", None))
        if action == "ASSIGNED":
            if strike is not None:
                s["basis"] = strike
            s["shares"] += mult * n
        elif action == "CALLED_AWAY":
            s["shares"] -= mult * n
            if s["shares"] <= 0:
                s["shares"], s["basis"] = 0, None
        elif action == "SOLD_SHARES":
            # tier 3 sells every share; `contracts` carries the share count and
            # `contract` is the bare ticker string.
            s["shares"], s["basis"] = 0, None
        elif action == "SELL_CALL":
            # Check FIRST, book the credit after -- see the docstring.
            if s["basis"] is not None and s["shares"] > 0 and strike is not None:
                nb = s["basis"] - s["premium"] / s["shares"]
                if strike < nb:
                    n_calls += 1
                    camps.add(cid)
            if delta is not None:
                s["premium"] += delta
            continue
        if action in _PREMIUM_ACTIONS and delta is not None:
            s["premium"] += delta
    return n_calls, len(camps)


def metrics_from_result(result, starting_capital: float,
                        n_slots: int | None = None) -> dict:
    """Equity metrics plus the portfolio-specific counters that explain them.

    `days_uncovered` is here because it is the n=1 diagnosis: it counts days
    holding assigned shares with NO covered call against them. With
    call_min_strike="basis" a crashed name has no strike at or above net basis,
    so it cannot be called away, and at n_slots=1 it occupies the only slot for
    the rest of the run -- which predicts few campaigns AND a deep drawdown at
    once. Without this counter that shape reads as a bad strategy rather than a
    jammed slot."""
    out = metrics_from_equity(result.equity, starting_capital)
    warnings = list(result.warnings or [])
    trades = list(result.trades or [])
    # ASSIGNMENT RATE is the entry gate's actual job, and until 2026-08-15 no
    # scorecard reported it. The weather-gate brief judges on mechanism counts
    # BEFORE P&L for a reason that is arithmetic, not taste: 272 campaigns
    # carried 54 assignments, so the assignment rate has real power over this
    # sample where a blended return does not. Counted off the trade log rather
    # than the result object so it needs no engine change.
    assigns = sum(1 for t in trades if getattr(t, "action", "") == "ASSIGNED")
    camps = int(getattr(result, "n_campaigns_opened", 0) or 0)
    # How often the assigned-stock exit ladder reached its last rung and SOLD.
    # Counted because the case for that ladder is that it fires deep and
    # rarely: `put_stop_mult` was falsified at every threshold in 2026-07 and
    # the -40% stop fired three times and did not survive its neighbours, so
    # "how often did it cut" is the first question about any exit that
    # realises a loss -- and until now no scorecard could answer it.
    sold = sum(1 for t in trades if getattr(t, "action", "") == "SOLD_SHARES")
    # Below-basis covered-call writes (schema 2): the tier2-deep auto-reject
    # is a per-campaign rate, so both the write count and the campaign count
    # go on the card. See below_basis_call_writes() for the definition.
    bb_calls, bb_camps = below_basis_call_writes(trades, starting_capital)
    out.update({
        "campaigns": camps,
        "assignments": assigns,
        "assignment_rate": (assigns / camps) if camps else None,
        "shares_sold": sold,
        "sold_per_assignment": (sold / assigns) if assigns else None,
        "calls_written_below_basis": bb_calls,
        "campaigns_with_below_basis_call": bb_camps,
        "days_flat": int(getattr(result, "days_flat", 0) or 0),
        "days_uncovered": int(getattr(result, "days_shares_uncovered", 0) or 0),
        "n_trades": int(len(result.trades or [])),
        "n_warnings": len(warnings),
        "warning_counts": _warning_counts(warnings),
        "final_shares": {k: float(v) for k, v in
                         (result.final_shares or {}).items() if v},
    })
    out.update(slot_economics(result, n_slots, starting_capital))
    return out


def _warning_counts(warnings) -> dict:
    """{reason: count}, top 12. The engine's refusal reasons are the second
    half of every result: an arm that scores worse because a gate refused 114
    of 145 entries is a different finding from one that scored worse on the
    entries it took."""
    counts: dict[str, int] = {}
    for w in warnings:
        reason = w[1] if isinstance(w, (tuple, list)) and len(w) > 1 else str(w)
        counts[str(reason)] = counts.get(str(reason), 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1])[:12])


# ---------------------------------------------------------------------------
# The scorecard
# ---------------------------------------------------------------------------

@dataclass
class Scorecard:
    variant: str
    base: str
    tier: str
    is_evidence: bool
    created: str
    arms: dict = field(default_factory=dict)        # arm name -> metrics
    provenance: dict = field(default_factory=dict)
    conditions: dict = field(default_factory=dict)
    divergences: list = field(default_factory=list)
    wall_secs: float = 0.0
    notes: str = ""
    schema: int = SCHEMA
    path: Path | None = None

    # -- derived ------------------------------------------------------------

    @property
    def config_hash(self) -> str:
        return (self.provenance.get("config_hash") or {}).get("variant", "")

    @property
    def base_config_hash(self) -> str:
        return (self.provenance.get("config_hash") or {}).get("base", "")

    @property
    def engine_sha(self) -> str:
        return (self.provenance.get("engine") or {}).get("sha", "")

    @property
    def dirty(self) -> bool:
        return bool((self.provenance.get("engine") or {}).get("dirty"))

    def delta(self) -> dict:
        """variant minus base, for every metric both arms report as a number."""
        a, b = self.arms.get("variant", {}), self.arms.get("base", {})
        out = {}
        for k in METRIC_ORDER:
            va, vb = _finite(a.get(k)), _finite(b.get(k))
            if va is not None and vb is not None:
                out[k] = va - vb
        return out

    # The [ops] fields that define the EXPERIMENT rather than the strategy. A
    # scorecard produced under different values does not describe the current
    # variant even when every knob matches. `tier` is excluded on purpose: it
    # is already in the filename, and one card per tier is the intended shape.
    OPS_FIELDS = ("n_slots", "capital", "selector")

    def is_stale_for(self, resolved_config_hash: str,
                     base_config_hash: str | None = None,
                     ops: dict | None = None) -> str | None:
        """Return why this scorecard no longer describes the current variant,
        or None if it still does.

        This is the anti-cherry-pick mechanism. Without it, a variant could be
        tuned after a favourable run and promoted on the old number.

        `ops` is the variant's RESOLVED ops dict. Pass it and the conditions of
        the experiment are compared too; omit it and the check is exactly what
        it was before, so this is additive. It matters because a variant may
        override the shared [ops] -- call-otm-25 moved n_slots 1 -> 5 on
        2026-08-16, because a mechanic that only fires on assigned stock cannot
        be resolved at n=1 -- and the knobs do not move when it does. Before
        this, the n=1 card stayed FRESH for the n=5 variant, kept the filename
        the new run would claim, and could have satisfied the promotion gate:
        the anti-cherry-pick mechanism failing in its own direction."""
        # NO HASH AT ALL is stale, not fresh. Imported scorecards (salvaged from
        # the old `results/` tree) carry no config hash because the harness that
        # produced them never recorded one -- so there is no way to know which
        # config they describe. Treating an absent hash as "matches" would make
        # every unverifiable old number the easiest evidence in the system to
        # satisfy the gate with, which is precisely backwards.
        if not self.config_hash:
            return ("this scorecard records no config hash, so there is no way "
                    "to tell which config produced it (imported from the old "
                    "results/ tree). Re-run it: `bench run <variant>`")
        if self.config_hash != resolved_config_hash:
            return (f"the variant changed since this run "
                    f"(scorecard {self.config_hash}, now {resolved_config_hash})")
        if base_config_hash and self.base_config_hash and \
                self.base_config_hash != base_config_hash:
            return (f"the BASE changed since this run (scorecard "
                    f"{self.base_config_hash}, now {base_config_hash}) -- the "
                    f"comparison is against a config that no longer exists")
        # A card with NO conditions block at all predates their being recorded.
        # It is left to the hash rules above rather than blanket-failed here:
        # every card `bench run` writes carries conditions (bench/run.py), and
        # the only cards that lack them are salvaged imports, which the no-hash
        # rule already treats as stale. Blanket-failing here would buy no real
        # safety and would retire the whole scorecard tree in one commit.
        #
        # A card that records conditions but is missing a field being compared
        # is a different animal -- a partial record from a writer that knew
        # about conditions -- and IS stale, because absence there is not
        # agreement.
        if ops and self.conditions:
            for f in self.OPS_FIELDS:
                want = ops.get(f)
                if want is None:
                    continue
                got = self.conditions.get(f)
                if got is None:
                    return (f"this scorecard records its run conditions but "
                            f"not {f}, so there is no way to tell whether it "
                            f"ran under the current {f}={want!r}. Re-run it: "
                            f"`bench run {self.variant}`")
                if got != want:
                    return (f"the run conditions changed since this scorecard: "
                            f"{f} was {got!r}, now {want!r}. The knobs are "
                            f"unchanged, so nothing else flags this -- but a "
                            f"mechanic measured at {f}={got!r} has not been "
                            f"measured at {f}={want!r}")
        return None

    # -- io -----------------------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("path", None)
        d["delta"] = self.delta()
        return d

    def filename(self) -> str:
        return f"{self.tier}__{self.config_hash or 'nohash'}.json"

    def save(self, root: Path | None = None) -> Path:
        root = Path(root or SCORECARD_DIR) / self.variant
        root.mkdir(parents=True, exist_ok=True)
        p = root / self.filename()
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=False),
                       encoding="utf-8")
        tmp.replace(p)
        self.path = p
        return p


def load(path: str | Path) -> Scorecard:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    d.pop("delta", None)
    sc = Scorecard(**{k: v for k, v in d.items()
                      if k in Scorecard.__dataclass_fields__})
    sc.path = Path(path)
    return sc


def for_variant(name: str, root: Path | None = None) -> list[Scorecard]:
    """Every scorecard on disk for a variant, newest first."""
    d = Path(root or SCORECARD_DIR) / name
    if not d.is_dir():
        return []
    cards = []
    for p in d.glob("*.json"):
        try:
            cards.append(load(p))
        except Exception:                                   # noqa: BLE001
            continue
    return sorted(cards, key=lambda c: c.created, reverse=True)


def new(variant: str, base: str, tier: str, is_evidence: bool) -> Scorecard:
    return Scorecard(variant=variant, base=base, tier=tier,
                     is_evidence=is_evidence,
                     created=datetime.now(timezone.utc)
                     .isoformat(timespec="seconds"))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_FMT = {
    "total_return": lambda v: f"{v:+.2%}",
    "cagr": lambda v: f"{v:+.2%}",
    "max_drawdown": lambda v: f"{v:.2%}",
    "worst_year": lambda v: f"{v:+.2%}",
    "sharpe": lambda v: f"{v:6.2f}",
    "ret_per_maxdd": lambda v: f"{v:6.2f}",
    "pnl": lambda v: f"{v:+,.0f}",
}


def _cell(metric: str, v) -> str:
    f = _finite(v)
    if f is None:
        return "n/a"
    fmt = _FMT.get(metric)
    return fmt(f) if fmt else (f"{int(f):,}" if float(f).is_integer() else f"{f:.3f}")


def render(sc: Scorecard, width: int = 84) -> str:
    """A terminal scorecard. Deliberately verbose about conditions: a table of
    numbers with no window, no universe count and no divergence list is how the
    old grid outputs became unreadable three weeks later."""
    L: list[str] = []
    ev = "EVIDENCE" if sc.is_evidence else "SMOKE -- not evidence"
    L.append("=" * width)
    L.append(f"  {sc.variant}   vs   {sc.base}        [{sc.tier} tier: {ev}]")
    L.append("=" * width)

    c = sc.conditions or {}
    L.append(f"  window     {c.get('start','?')} -> {c.get('end','?')}"
             f"    n_slots {c.get('n_slots','?')}"
             f"    capital {c.get('capital', 0):,.0f}")
    L.append(f"  universe   {c.get('n_tickers_loaded','?')} loaded "
             f"of {c.get('n_tickers_requested','?')} requested"
             f"    selector {c.get('selector','?')}")
    p = sc.provenance or {}
    eng = p.get("engine", {})
    dat = p.get("data", {})
    L.append(f"  engine     {eng.get('sha','?')} on {eng.get('branch','?')}"
             + ("   *** DIRTY TREE -- not reproducible ***" if eng.get("dirty") else ""))
    L.append(f"  data       {dat.get('hash','?')}  "
             f"{dat.get('n_files',0):,} files  {dat.get('bytes',0)/1e9:.2f} GB")
    L.append(f"  config     variant {sc.config_hash}   base {sc.base_config_hash}")
    L.append(f"  ran in     {sc.wall_secs:,.0f}s   at {sc.created}")
    L.append("-" * width)

    arms = [a for a in ("variant", "base", "buy_hold", "spy_buy_hold")
            if a in sc.arms]
    head = f"  {'metric':<16}" + "".join(f"{a:>14}" for a in arms) + f"{'delta':>14}"
    L.append(head)
    L.append("-" * width)
    delta = sc.delta()
    for met in METRIC_ORDER:
        if not any(met in sc.arms[a] for a in arms):
            continue
        row = f"  {met:<16}"
        for a in arms:
            row += f"{_cell(met, sc.arms[a].get(met)):>14}"
        if met in delta:
            arrow = ""
            if met in HIGHER_IS_BETTER:
                arrow = " +" if delta[met] > 0 else (" -" if delta[met] < 0 else "  ")
            elif met == "max_drawdown":
                arrow = " +" if delta[met] > 0 else (" -" if delta[met] < 0 else "  ")
            row += f"{_cell(met, delta[met]):>12}{arrow}"
        L.append(row)

    by = {a: sc.arms[a].get("by_year") or {} for a in arms}
    years = sorted({y for m in by.values() for y in m})
    if years:
        L.append("-" * width)
        L.append(f"  {'year':<16}" + "".join(f"{a:>14}" for a in arms))
        for y in years:
            L.append(f"  {y:<16}" + "".join(
                f"{_cell('total_return', by[a].get(y)):>14}" for a in arms))

    wc = (sc.arms.get("variant") or {}).get("warning_counts") or {}
    if wc:
        L.append("-" * width)
        L.append("  refusals / warnings (variant arm)")
        for k, v in wc.items():
            L.append(f"    {k:<40} {v:>8,}")

    if sc.divergences:
        L.append("-" * width)
        L.append("  DECLARED DIVERGENCES from the live bot")
        for d in sc.divergences:
            for i, line in enumerate(str(d).split("\n")):
                L.append(f"    {'* ' if i == 0 else '  '}{line}")

    L.append("-" * width)
    L.append("  Read this as a RANKING, not a forecast: no held-out split, every")
    L.append("  arm sees the whole window (owner call 2026-08-09). The forward")
    L.append("  paper run is the exam.")
    L.append("=" * width)
    return "\n".join(L)
