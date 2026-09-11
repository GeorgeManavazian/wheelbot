"""`bench promote <variant>` -- the only sanctioned way to change the live bot.

WHAT IT WRITES. For every knob the variant actually moves, it writes into
`variants/frozen.toml`: the new value, the variant's own `why` prose, and a
provenance footer naming the date, the variant, the human verdict and the
evidence behind it. Then it updates `variants/frozen.lock.json`, whose git diff
is the dated record of what changed about the bot, and regenerates the
changelog.

WHAT IT REFUSES. Everything `bench/policy.py` says to refuse -- and it runs all
the gates before reporting, so you get the whole list rather than the first
failure. `--dry-run` shows exactly what would be written without touching a
file, which is the mode to use while the evidence is still coming together.

WHY THE PROSE IS CARRIED FORWARD AND NOT SUMMARISED. The most valuable text in
this project is the 25-line explanation of the A2b liquidity finding sitting
beside the knob it justifies. That survived only because someone kept
hand-writing it into a comment. Carrying `why` mechanically means the next one
survives without anybody remembering to.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

from bench import config as bench_config
from bench import fingerprint, lock, policy
from bench import variant as bench_variant


@dataclass
class PromoteResult:
    variant: str
    changed: dict = field(default_factory=dict)     # knob -> (old, new)
    report: policy.PolicyReport | None = None
    applied: bool = False
    files: list = field(default_factory=list)
    message: str = ""

    def render(self) -> str:
        L = []
        if self.report:
            L.append(self.report.render())
            L.append("")
        if not self.changed:
            L.append(f"`{self.variant}` changes nothing about the master "
                     f"config -- there is nothing to promote.")
            return "\n".join(L)
        L.append(f"{'APPLIED' if self.applied else 'WOULD CHANGE'} in "
                 f"variants/frozen.toml:")
        L.append("")
        w = max(len(k) for k in self.changed)
        for k, (old, new) in sorted(self.changed.items()):
            L.append(f"  {k:<{w}}   {old!r}  ->  {new!r}")
        L.append("")
        if self.files:
            L.append("files written:")
            for f in self.files:
                L.append(f"  {f}")
        if self.message:
            L.append("")
            L.append(self.message)
        return "\n".join(L)


def diff_for(variant_name: str) -> dict:
    """{knob: (master_value, variant_value)} -- what promoting would change."""
    master = bench_variant.resolve(bench_variant.MASTER_VARIANT)
    cand = bench_variant.resolve(variant_name)
    return cand.diff_against(master)


_FOOTER = """

--- promoted {date} from variant `{variant}` ---
Verdict: {verdict}
Evidence: {evidence}"""


def _promotion_why(base_why: str, *, date: str, variant: str, verdict: str,
                   evidence: str) -> str:
    """The variant's own reasoning, plus a footer nobody has to remember to
    write. Idempotent-ish: promoting the same variant twice appends a second
    footer, which is correct -- it happened twice."""
    return (base_why.rstrip() + _FOOTER.format(
        date=date, variant=variant, verdict=verdict, evidence=evidence)).strip()


def promote(variant_name: str, verdict: str = "", *, dry_run: bool = False,
            skip_tests: bool = False, today: str | None = None,
            force: bool = False) -> PromoteResult:
    today = today or dt.date.today().isoformat()
    changed = diff_for(variant_name)
    res = PromoteResult(variant=variant_name, changed=changed)

    report = policy.check(variant_name, verdict=verdict, skip_tests=skip_tests)
    res.report = report
    if not changed:
        return res
    if not report.passed and not force:
        res.message = ("Nothing was written. Fix the FAIL lines above, or "
                       "re-run with --force to promote anyway (the override is "
                       "recorded in the changelog).")
        return res
    if dry_run:
        res.message = "--dry-run: nothing was written."
        return res

    card = report.scorecard()
    ev: dict = {}
    if card is not None:
        d = card.delta()
        ev = {"scorecard": f"{card.tier}/{card.config_hash}",
              "engine": card.engine_sha,
              "data": (card.provenance.get("data") or {}).get("hash", ""),
              "sharpe_delta": f"{d.get('sharpe', float('nan')):+.3f}",
              "ret_per_maxdd_delta": f"{d.get('ret_per_maxdd', float('nan')):+.3f}"}
    if not report.passed and force:
        ev["override"] = "promoted with --force despite failing gates"
    ev_text = " · ".join(f"{k}={v}" for k, v in ev.items()) or "none recorded"

    cand = bench_variant.resolve(variant_name)
    master = bench_variant.load(bench_variant.MASTER_VARIANT)

    for knob in changed:
        src = cand.provenance.get(knob)
        base_why = src.why if src else ""
        if not base_why.strip():
            base_why = (f"Promoted from `{variant_name}` with no `why` on the "
                        f"knob. This should not have passed validation -- fix it.")
        master.knobs[knob] = bench_variant.Knob(
            name=knob, value=cand.knobs[knob],
            why=_promotion_why(base_why, date=today, variant=variant_name,
                               verdict=verdict or "(none)", evidence=ev_text),
            since=today, source=f"variant:{variant_name}")

    master.updated = today
    master_path = master.save()
    res.files.append(fingerprint.rel(master_path))

    new_knobs = bench_config.load_frozen()
    lock.append(new_knobs, date=today, variant=variant_name, verdict=verdict,
                evidence=ev, changed=changed,
                note=bench_variant.load(variant_name).question)
    res.files.append(fingerprint.rel(lock.LOCK_PATH))

    v = bench_variant.load(variant_name)
    v.status = "promoted"
    v.updated = today
    v.evidence.append({"kind": "promotion", "date": today, "verdict": verdict,
                       **{k: str(x) for k, x in ev.items()}})
    v.save()
    res.files.append(fingerprint.rel(v.path))

    from bench import registry
    res.files.extend(registry.regenerate())

    res.applied = True
    res.message = (
        "The live bot's config changed. `git diff variants/` is the review: "
        "frozen.lock.json shows exactly which knobs moved, frozen.toml shows "
        "the reasoning that moved with them. Nothing reaches the live bot until you "
        "deploy.")
    return res
