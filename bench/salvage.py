"""Rescue the verdicts that are one `rm -rf results/` from gone.

`results/` is in `.gitignore`. Everything this project has ever measured lives
there and nowhere else:

    results/wheel_grid_n1/       10-arm config grid at n=1  (2026-08-09)
    results/wheel_factorial_n1/  60-arm delta x DTE x TP x floor factorial
    results/ranker_width/        the IV ranker at n = 1, 3, 5
    results/gate_audit/          gate ablation, basis-floor A/B, path noise

None of it records which engine, which data, or which exact config produced it,
so none of it can be re-run or verified. That is why this module imports it
as ARCHIVE, not as evidence:

  * the raw JSON is copied verbatim into `bench/imported/`, which IS committed,
    so the numbers stop being one command from oblivion;
  * the 10-arm grid additionally becomes real variant files, because each arm is
    a genuine paragraph -- a candidate live config with a stated question;
  * every scorecard it writes is stamped `is_evidence = false` AND carries no
    config hash, so `bench promote` refuses it twice over
    (`scorecard.is_stale_for` reads a missing hash as stale, by design).

The point is not to launder old numbers into promotions. It is that "we already
tried delta 0.40 at DTE 7 and here is what happened" should be answerable by
`bench show`, not by remembering which scratchpad file to open.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from bench import scorecard as S
from bench import toml_io
from bench import variant as V

RESULTS = Path("results")
IMPORTED = Path(__file__).resolve().parent / "imported"

#: The ten arms of the 2026-08-09 grid, from `scratchpad/run_wheel_grid_n1.py`.
#: (row, put_delta, target_dte, take_profit, rank_by, label)
GRID_ARMS = [
    (1, 0.30, 11, 0.60, "iv_rank", "live FROZEN + ranker"),
    (2, 0.20, 11, 0.60, "iv_rank", "delta 0.20"),
    (3, 0.40, 11, 0.60, "iv_rank", "delta 0.40"),
    (4, 0.30, 7, 0.60, "iv_rank", "DTE 7"),
    (5, 0.30, 21, 0.60, "iv_rank", "DTE 21"),
    (6, 0.30, 11, 0.25, "iv_rank", "TP 25%"),
    (7, 0.30, 11, 0.90, "iv_rank", "TP 90%"),
    (8, 0.40, 7, 0.25, "iv_rank", "old sweep arm"),
    (9, 0.20, 21, 0.60, "iv_rank", "conservative corner"),
    (10, 0.30, 11, 0.60, "vol_pctile", "CONTROL (no ranker)"),
]

GRID_SLUG = {1: "iv-ranker", 2: "iv-ranker-delta-020", 3: "iv-ranker-delta-040",
             4: "iv-ranker-dte-7", 5: "iv-ranker-dte-21",
             6: "iv-ranker-tp-25", 7: "iv-ranker-tp-90",
             8: "iv-ranker-old-sweep-arm", 9: "iv-ranker-conservative-corner"}

#: Every divergence the grid harness declared in its own docstring. Carried onto
#: every imported scorecard so the numbers are never read as live-faithful.
GRID_DIVERGENCES = [
    "liq_min_open_interest was None: the chain store had no open_interest "
    "column when this ran, and the flat-file OI endpoint was 403 before "
    "2026-08-02. The live config sets 250. (The OI store has since been pulled "
    "-- commit fb196a1 -- so a re-run under `bench run` can carry all three "
    "liquidity legs.)",
    "liq_max_rel_spread was 0.10, the value A2b turned OFF on 2026-08-11 after "
    "measuring that it refused 25 of 27 good-to-rent names on cheapness.",
    "No held-out split (owner call 2026-08-09): every arm saw the whole window, "
    "so the winner is a RANKING selected in-sample over ten tries, not a "
    "forecast.",
    "n_slots = 1. At n=1 the held-exclusion rule removes exactly one name from "
    "the pool, so the self-starvation that killed rank_by='iv_rank' at n=3 and "
    "n=5 in the 2026-08-05 A/B cannot fire. A win here is a win at n=1.",
    "No engine sha, no data fingerprint, no config hash were recorded by the "
    "harness -- which is why this scorecard is stamped is_evidence = false.",
]

_WINDOW = {"start": "2022-06-01", "end": "2026-07-01", "n_slots": 1,
           "capital": 100000.0, "selector": "chop", "n_tickers_loaded": 530,
           "n_tickers_requested": 531}


def _metrics(raw: dict) -> dict:
    """Old harness row -> the bench metric names. Renames only; nothing is
    recomputed, because recomputing from a summary would invent precision."""
    by_year = {str(k): v for k, v in (raw.get("by_year") or {}).items()}
    out = {
        "total_return": raw.get("total"),
        "cagr": raw.get("cagr"),
        "sharpe": raw.get("sharpe"),
        "max_drawdown": raw.get("maxdd"),
        "ret_per_maxdd": raw.get("ret_per_dd"),
        "pnl": raw.get("pnl"),
        "campaigns": raw.get("campaigns"),
        "days_flat": raw.get("days_flat"),
        "days_uncovered": raw.get("days_uncovered"),
        "n_days": raw.get("n_days"),
        "by_year": by_year,
        "worst_year": min(by_year.values()) if by_year else None,
    }
    return {k: v for k, v in out.items() if v is not None}


def copy_raw(verbose=True) -> list[str]:
    """Copy every study JSON into `bench/imported/`, verbatim.

    Verbatim matters: a transformed copy is a second thing that can be wrong,
    and the whole reason these files are worth keeping is that they are the
    original measurement."""
    written = []
    for sub in ("wheel_grid_n1", "wheel_factorial_n1", "ranker_width",
                "gate_audit"):
        src = RESULTS / sub
        if not src.is_dir():
            continue
        dst = IMPORTED / sub
        dst.mkdir(parents=True, exist_ok=True)
        for p in sorted(src.glob("*.json")):
            shutil.copy2(p, dst / p.name)
            written.append(str((dst / p.name).relative_to(IMPORTED.parent)))
    if verbose:
        print(f"  copied {len(written)} study file(s) into bench/imported/")
    return written


def import_grid(verbose=True) -> list[str]:
    """The 2026-08-09 ten-arm grid -> archived variants + scorecards."""
    src = RESULTS / "wheel_grid_n1"
    if not src.is_dir():
        if verbose:
            print("  no results/wheel_grid_n1 -- nothing to import")
        return []

    rows = {}
    for row, *_ in GRID_ARMS:
        p = src / f"row{row:02d}.json"
        if p.exists():
            rows[row] = json.loads(p.read_text())
    control = rows.get(10)
    buyhold = None
    bh = src / "buyhold.json"
    if bh.exists():
        buyhold = _metrics(json.loads(bh.read_text()))

    made = []
    V.ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    for row, delta, dte, tp, rank_by, label in GRID_ARMS:
        if row == 10 or row not in rows:
            continue                    # row 10 IS frozen; it is the control
        name = GRID_SLUG[row]
        knobs: dict[str, V.Knob] = {}
        if row == 1:
            knobs["rank_by"] = V.Knob(
                "rank_by", "iv_rank",
                why=_RANKER_WHY, since="2026-08-05",
                source="grid row 1, 2026-08-09")
        else:
            for field, val, live in (("put_delta", delta, 0.30),
                                     ("target_dte", dte, 11),
                                     ("take_profit_pct", tp, 0.60)):
                if val != live:
                    knobs[field] = V.Knob(
                        field, val,
                        why=f"Grid row {row} ({label}): one knob moved off the "
                            f"live value ({live} -> {val}) with everything else "
                            f"held, so a cliff shows up as one bad neighbour "
                            f"rather than as a bad corner of a sweep.",
                        since="2026-08-09", source="grid row %d" % row)

        v = V.Variant(
            name=name,
            question=_QUESTION.get(row, f"Grid row {row}: does {label} beat the "
                                        f"live config under the IV ranker?"),
            base="frozen" if row == 1 else "iv-ranker",
            status="testing" if row == 1 else "superseded",
            created="2026-08-09", updated="2026-08-14",
            touches=["ranking"] if row == 1 else ["dte" if dte != 11 else
                                                  ("delta" if delta != 0.30
                                                   else "take-profit")],
            knobs=knobs,
            ops={"tier": "full", "n_slots": 1},
            note=_GRID_NOTE.format(row=row, label=label, name=name))
        p = v.save(V.ARCHIVE_DIR / f"{name}.toml")
        made.append(str(p.relative_to(V.REPO_ROOT)))

        card = S.new(name, "frozen", "full", is_evidence=False)
        card.arms["variant"] = _metrics(rows[row])
        if control:
            card.arms["base"] = _metrics(control)
        if buyhold:
            card.arms["buy_hold"] = buyhold
        card.created = "2026-08-09T00:00:00+00:00"
        card.conditions = dict(_WINDOW)
        card.divergences = list(GRID_DIVERGENCES)
        card.wall_secs = float(rows[row].get("secs", 0.0))
        card.notes = (f"IMPORTED from results/wheel_grid_n1/row{row:02d}.json "
                      f"by bench/salvage.py. Not evidence: no engine sha, no "
                      f"data fingerprint, no config hash was ever recorded.")
        card.provenance = {"engine": {"sha": "", "branch": "", "dirty": True},
                           "config_hash": {"variant": "", "base": ""},
                           "data": {"hash": ""},
                           "imported_from": f"results/wheel_grid_n1/row{row:02d}.json"}
        made.append(str(card.save().relative_to(V.REPO_ROOT)))

    if verbose:
        print(f"  imported {len(rows)-1} grid arm(s) as archived variants")
    return made


_RANKER_WHY = """
Rank the candidate pool by IV RANK rather than by realized-vol percentile.

The live sort (`vol_pctile`) picks the riskiest name and never reads the price
it is paid for that risk. Worse, the chop weather gate has already capped
realized vol at the 75th percentile, so the sort is choosing inside a 70-75th
percentile sliver on the wrong variable.

A sort is the right instrument where a veto was the wrong one. `min_iv_rank` was
measured as a VETO on 2026-08-04 and rejected: it raised P&L per campaign ~14%
but HALVED campaign count, costing 18 points of total return (+43.0% -> +24.9%
at 0.88) and 0.26 of Sharpe, because the wheel's return comes from capital
turnover. A sort changes WHICH names fill the slots, never HOW MANY, so it does
not pay that cost.

Owner ruling 2026-08-08: ship it in the live engine, stop proposing another A/B.
The read side landed in commit 6272640 and it is still OFF in the master config
as of 2026-08-14.
""".strip()

_QUESTION = {
    1: "Does ranking the pool by IV rank instead of realized-vol percentile "
       "beat the live config?",
    2: "Is delta 0.20 a better place on the leverage dial than the live 0.30?",
    3: "Is delta 0.40 a better place on the leverage dial than the live 0.30?",
    4: "Does DTE 7 beat the live 11 under the ranker?",
    5: "Does DTE 21 beat the live 11 under the ranker?",
    6: "Does taking profit at 25% beat the live 60% under the ranker?",
    7: "Does taking profit at 90% beat the live 60% under the ranker?",
    8: "Does the old sweep's +95.4% corner (0.40 / 7 / 25%) survive the live "
       "gates?",
    9: "Does the conservative corner (0.20 / 21 / 60%) beat the live config?",
}

_GRID_NOTE = """
IMPORTED by bench/salvage.py from the 2026-08-09 ten-arm grid
(`scratchpad/run_wheel_grid_n1.py`, row {row}: {label}).

Its scorecard is stamped **is_evidence = false** and carries no config hash, so
`bench promote` refuses it twice over. That is not a judgement on the number --
it is that the harness recorded no engine sha, no data fingerprint and no config
hash, so nobody can tell what produced it. To make it evidence, re-run it:
`bench run {name}`.

JUDGE ON Sharpe and return-per-drawdown, not total return. Delta behaves as a
leverage dial in this engine, so the 0.40-delta arms top the return column by
construction -- which is exactly how the original sweep table became unreadable.
""".strip()


def index_markdown() -> str:
    """A generated index of every salvaged study. Never hand-maintained."""
    L = ["# Imported studies — rescued from the gitignored `results/` tree",
         "",
         "GENERATED by `bench salvage`. Do not edit by hand.",
         "",
         "`results/` is in `.gitignore`, so every verdict this project had ever "
         "computed was one `rm -rf` from gone. The raw JSON below is copied "
         "here verbatim and committed.",
         "",
         "**None of it is evidence.** No study recorded an engine sha, a data "
         "fingerprint or a config hash, so no number here can be tied to the "
         "code and data that produced it. `bench promote` refuses them. They "
         "are kept so that *\"we already tried that\"* is answerable.",
         ""]
    for sub, blurb in (
            ("wheel_grid_n1", "10-arm config grid at n=1 (2026-08-09). Also "
                              "imported as archived variants — see "
                              "`variants/_archive/iv-ranker*.toml`."),
            ("wheel_factorial_n1", "60-arm factorial: delta x DTE x "
                                   "take-profit x basis-floor."),
            ("ranker_width", "the IV ranker at n = 1, 3 and 5 — the "
                             "self-starvation question."),
            ("gate_audit", "gate ablation, basis-floor A/B, jam depth, path "
                           "noise, baseline replication.")):
        d = IMPORTED / sub
        if not d.is_dir():
            continue
        files = sorted(p.name for p in d.glob("*.json"))
        L += [f"## `{sub}` — {len(files)} file(s)", "", blurb, "",
              "<details><summary>files</summary>", ""]
        L += [f"- `{f}`" for f in files]
        L += ["", "</details>", ""]
    return "\n".join(L)


def run(verbose=True) -> list[str]:
    written = copy_raw(verbose)
    written += import_grid(verbose)
    IMPORTED.mkdir(parents=True, exist_ok=True)
    (IMPORTED / "README.md").write_text(index_markdown(), encoding="utf-8")
    written.append("bench/imported/README.md")
    from bench import registry
    written += registry.regenerate()
    return written
