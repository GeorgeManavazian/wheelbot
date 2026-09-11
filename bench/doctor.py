"""`bench doctor` -- did anyone go around the bench?

WHY THIS EXISTS. The owner's question, 2026-08-15: "how do I know if our future
Claude sessions are using this?" The honest answer at the time was "partly" --
some rules were enforced by tests, and some were only written down in a file
that may or may not be read. This closes the gap by making the question
answerable in one command instead of by reading a diff.

THE DISTINCTION IT DRAWS. There are two ways the discipline gets lost:

  DRIFT      someone edits the live config by hand, writes a variant with no
             reasoning, promotes on a stale run, or drops another one-off
             harness in scratchpad/. All of this is accidental, and all of it is
             detectable.
  FORGERY    someone edits the live config, regenerates the lock, AND writes a
             convincing fake promotion entry into the history. Nothing here can
             stop that, and nothing should pretend to. What it does is make the
             cheapest path (drift) fail loudly, so that going around the bench
             takes deliberate effort and leaves a git diff that says so.

THE LOCK-HISTORY CHECK IS THE LOAD-BEARING ONE. `frozen.lock.json` carries both
the current config and an append-only history. If someone hand-edits the master
and regenerates the lock, `current` no longer matches the last history entry --
which is exactly the signature of a change that did not come from `bench
promote`. That is the hole a value-equality check alone could not close.

Exit code is 1 when anything is wrong, so the pre-push hook can refuse a push.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from bench import blocks as B
from bench import config as bench_config
from bench import fingerprint, lock, scorecard as S
from bench import variant as V

REPO_ROOT = fingerprint.REPO_ROOT

#: A finding's weight. FAIL means the discipline was broken and the push should
#: stop; WARN means something needs a human decision but is not itself wrong.
FAIL, WARN, OK = "FAIL", "WARN", "OK"


@dataclass
class Finding:
    level: str
    check: str
    detail: str
    fix: str = ""


@dataclass
class Report:
    findings: list = field(default_factory=list)

    @property
    def failures(self) -> list:
        return [f for f in self.findings if f.level == FAIL]

    @property
    def warnings(self) -> list:
        return [f for f in self.findings if f.level == WARN]

    @property
    def healthy(self) -> bool:
        return not self.failures

    def add(self, level, check, detail, fix=""):
        self.findings.append(Finding(level, check, detail, fix))

    def render(self, width: int = 84) -> str:
        L = ["=" * width]
        head = ("the bench is being used" if self.healthy
                else "SOMETHING WENT AROUND THE BENCH")
        L += [f"  bench doctor -- {head}", "=" * width]
        checks = [f for f in self.findings if f.level != OK]
        if not checks:
            L.append("  Every check passed.")
        for f in checks:
            L.append("")
            L.append(f"  [{f.level}] {f.check}")
            for line in f.detail.split("\n"):
                L.append(f"        {line}")
            if f.fix:
                L.append(f"        -> {f.fix}")
        passed = [f for f in self.findings if f.level == OK]
        if passed:
            L += ["", "  passed: " + ", ".join(f.check for f in passed)]
        L.append("=" * width)
        return "\n".join(L)


def _git(*args: str) -> str:
    try:
        return subprocess.run(("git", "-C", str(REPO_ROOT)) + args,
                              capture_output=True, text=True,
                              timeout=15).stdout.strip()
    except Exception:                                       # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_lock_matches_master(r: Report) -> None:
    """The live config file and its lock must agree."""
    try:
        locked = lock.current()
    except FileNotFoundError as e:
        r.add(FAIL, "lock present", str(e),
              "restore variants/frozen.lock.json from git")
        return
    try:
        live = bench_config.load_frozen()
    except Exception as e:                                  # noqa: BLE001
        r.add(FAIL, "master loads", str(e))
        return
    if live != locked:
        only_master = {k: v for k, v in live.items() if locked.get(k, ...) != v}
        only_lock = {k: v for k, v in locked.items() if live.get(k, ...) != v}
        r.add(FAIL, "lock matches master",
              f"variants/frozen.toml and frozen.lock.json disagree.\n"
              f"in the master, not the lock: {only_master}\n"
              f"in the lock, not the master: {only_lock}",
              "the master is only meant to change via `bench promote` -- "
              "revert the hand-edit, or promote a variant")
    else:
        r.add(OK, "lock matches master", "")


def check_history_explains_the_current_config(r: Report) -> None:
    """THE ONE THAT CLOSES THE HOLE.

    The golden test only proves the master equals the lock. Someone who edits
    the master and regenerates the lock passes it. But `current` must also equal
    the knobs recorded in the LAST history entry -- and a regenerate that does
    not append history breaks exactly that."""
    try:
        doc = lock.load()
    except FileNotFoundError:
        return                                  # already reported above
    hist = doc.get("history") or []
    if not hist:
        r.add(FAIL, "history explains current",
              "the lock has no history at all, so nothing records how the live "
              "config got the way it is",
              "restore variants/frozen.lock.json from git")
        return
    last = hist[-1]
    if doc.get("current") != last.get("knobs"):
        r.add(FAIL, "history explains current",
              "the live config does not match the last recorded change. Either "
              "the master was edited and the lock regenerated without recording "
              "a promotion, or the history was truncated.\n"
              f"last entry: {last.get('date','?')} by {last.get('by','?')}"
              + (f" (variant {last['variant']})" if last.get("variant") else ""),
              "`bench promote` is what writes both. Revert, then promote.")
        return
    unexplained = [h for h in hist
                   if h.get("by") == "promote" and not (h.get("verdict") or "").strip()]
    if unexplained:
        r.add(WARN, "history explains current",
              f"{len(unexplained)} promotion(s) recorded with no verdict",
              "promotions are meant to carry the sentence you would want to "
              "read in six months")
    else:
        r.add(OK, "history explains current", "")


def check_master_reasoning(r: Report) -> None:
    """Every live knob keeps a real reason and a date."""
    master = V.load(V.MASTER_VARIANT)
    thin = {k: len(kn.why.split()) for k, kn in master.knobs.items()
            if len(kn.why.split()) < 12}
    undated = [k for k, kn in master.knobs.items() if not kn.since]
    if thin:
        r.add(FAIL, "master reasoning",
              f"knobs whose reasoning is under 12 words: {thin}",
              "a knob without its reason is how a setting ends up live with "
              "nothing behind it")
    elif undated:
        r.add(WARN, "master reasoning", f"knobs with no `since` date: {undated}")
    else:
        r.add(OK, "master reasoning", "")


def check_variants_valid(r: Report) -> None:
    known = V.wheel_fields()
    bad = {}
    for name, v in V.all_variants().items():
        probs = [p for p in V.validate(v, known_fields=known)
                 if "check out" not in p]     # an unchecked-out branch is fine
        if probs:
            bad[name] = probs
    if bad:
        lines = [f"{n}: {'; '.join(p[:110] for p in ps[:2])}"
                 for n, ps in bad.items()]
        r.add(FAIL, "variants valid", "\n".join(lines),
              "`bench validate` for the full list")
    else:
        r.add(OK, "variants valid", "")


def check_block_coverage(r: Report) -> None:
    probs = B.coverage_problems()
    if probs:
        r.add(FAIL, "block coverage", "\n".join(probs[:4]),
              "add the field to bench/blocks.toml -- an engine setting with no "
              "block is invisible in every readable view of the bot")
    else:
        r.add(OK, "block coverage", "")


def check_claims_have_evidence(r: Report) -> None:
    """A variant claiming to be tested or promoted should have a scorecard, and
    that scorecard should still describe it."""
    stale, unbacked = [], []
    for name, v in V.all_variants().items():
        if name in (V.ROOT_VARIANT, V.MASTER_VARIANT):
            continue
        if v.status not in ("testing", "promoted"):
            continue
        cards = S.for_variant(name)
        if not cards:
            unbacked.append(name)
            continue
        try:
            now = fingerprint.config_hash(bench_config.knobs_for(name))
            # Ops too: a card run at a different slot width does not back a
            # claim made at this one, and the knob hash cannot see that.
            now_ops = bench_config.ops_for(name)
        except Exception:                                   # noqa: BLE001
            continue
        if all(c.is_stale_for(now, ops=now_ops) for c in cards):
            stale.append(name)
    if unbacked:
        r.add(WARN, "claims have evidence",
              f"status says tested but no scorecard exists: {', '.join(unbacked)}",
              "`bench run <name>`")
    if stale:
        r.add(WARN, "claims have evidence",
              f"every scorecard predates the current settings: {', '.join(stale)}",
              "`bench run <name>` again -- the old number describes a config "
              "that no longer exists")
    if not unbacked and not stale:
        r.add(OK, "claims have evidence", "")


def check_no_new_one_off_harnesses(r: Report) -> None:
    """The old habit: a new untracked script in scratchpad/ that hand-mirrors
    the live config. That is what the bench replaced."""
    untracked = [l for l in _git("ls-files", "--others", "--exclude-standard",
                                 "scratchpad").split("\n") if l.strip()]
    py = [p for p in untracked if p.endswith(".py")]
    mirrors = []
    for p in py:
        try:
            txt = (REPO_ROOT / p).read_text(errors="replace")
        except OSError:
            continue
        if "WheelConfig(" in txt and "bench" not in txt:
            mirrors.append(p)
    if mirrors:
        r.add(WARN, "no one-off harnesses",
              f"{len(mirrors)} untracked script(s) build a WheelConfig without "
              f"going through the bench:\n" + "\n".join(f"  {m}" for m in mirrors[:5]),
              "these drift from the live config -- run_wheel_grid_n1.py said it "
              "mirrored FROZEN and did not. Use `bench new` + `bench run`.")
    else:
        r.add(OK, "no one-off harnesses", "")


def check_branches(r: Report) -> None:
    branches = [b.strip().lstrip("* ").strip()
                for b in _git("branch", "--format=%(refname:short)").split("\n")
                if b.strip()]
    stale = []
    for b in branches:
        if b in ("main",):
            continue
        behind_ahead = _git("rev-list", "--left-right", "--count", f"main...{b}")
        parts = behind_ahead.split()
        if len(parts) == 2 and parts[1] == "0":
            stale.append(b)
    if stale:
        r.add(WARN, "branches", f"fully merged, safe to delete: {', '.join(stale)}",
              "git branch -d " + " ".join(stale))
    elif len(branches) > 3:
        r.add(WARN, "branches",
              f"{len(branches)} branches open: {', '.join(branches)}",
              "a mechanic belongs in a variant file, not a long-lived branch")
    else:
        r.add(OK, "branches", "")


def check_uncommitted_config(r: Report) -> None:
    """An uncommitted change to variants/ is a change to the bot nobody has
    reviewed."""
    dirty = [l for l in _git("status", "--porcelain", "variants").split("\n")
             if l.strip()]
    if dirty:
        r.add(WARN, "config committed",
              "uncommitted changes under variants/:\n"
              + "\n".join(f"  {d}" for d in dirty[:6]),
              "commit them -- `git diff variants/` is the review of what changed "
              "about the bot")
    else:
        r.add(OK, "config committed", "")


CHECKS = (check_lock_matches_master, check_history_explains_the_current_config,
          check_master_reasoning, check_variants_valid, check_block_coverage,
          check_claims_have_evidence, check_no_new_one_off_harnesses,
          check_branches, check_uncommitted_config)

#: The two checks that answer "did the live config change without a promotion?".
#: Pure file reads, no git subprocess -- fast enough to run on every Stop hook,
#: where the full set's ~1s would be felt on every single turn.
QUICK = (check_lock_matches_master, check_history_explains_the_current_config)


def run(only: list | None = None, quick: bool = False) -> Report:
    r = Report()
    for fn in (QUICK if quick else CHECKS):
        if only and fn.__name__ not in only:
            continue
        try:
            fn(r)
        except Exception as e:                              # noqa: BLE001
            r.add(FAIL, fn.__name__.replace("check_", "").replace("_", " "),
                  f"the check itself failed: {type(e).__name__}: {e}")
    return r
