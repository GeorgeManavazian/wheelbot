"""The promotion gate. Reads `bench/policy.toml`; decides nothing on its own.

The split this module exists to hold:

  THE MACHINE checks that evidence EXISTS, is FRESH, and is of a kind that
  counts. Freshness is the load-bearing half -- a scorecard whose config hash
  no longer matches the variant is describing a config that no longer exists,
  and accepting it is exactly how you promote on a number you tuned away from.

  THE HUMAN decides whether the numbers are good. That is `--verdict`, recorded
  verbatim in the master config's provenance. The owner pinned the numeric bar
  on 2026-08-14; when it arrives it goes in `[thresholds]` and the machine
  starts checking it too, with no code change here.

Every gate returns the same shape, so `bench promote --dry-run` can print the
full picture instead of stopping at the first failure. Being told all four
reasons at once is the difference between one more run and four.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from bench import fingerprint, scorecard as sc_mod, toml_io
from bench import variant as bench_variant

POLICY_PATH = Path(__file__).resolve().parent / "policy.toml"


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str
    evidence: dict = field(default_factory=dict)
    skipped: bool = False

    def line(self) -> str:
        mark = "SKIP" if self.skipped else ("PASS" if self.passed else "FAIL")
        return f"  [{mark}] {self.name:<10} {self.detail}"


@dataclass
class PolicyReport:
    variant: str
    gates: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(g.passed or g.skipped for g in self.gates)

    def render(self) -> str:
        head = f"promotion gates for `{self.variant}` -- " + \
               ("ALL PASS" if self.passed else "BLOCKED")
        return "\n".join([head] + [g.line() for g in self.gates])

    def scorecard(self) -> sc_mod.Scorecard | None:
        for g in self.gates:
            if g.name == "scorecard" and g.passed:
                return g.evidence.get("card")
        return None


def load_policy(path: Path | None = None) -> dict:
    return toml_io.load(path or POLICY_PATH)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def _gate_tests(v, pol, ctx) -> GateResult:
    """Run the fast suite NOW. A pass recorded yesterday says nothing about the
    tree today, and a stored test result is the easiest evidence to fake by
    accident."""
    cmd = list(pol.get("tests", {}).get("command",
                                        ["-m", "pytest", "-q", "-m", "not slow"]))
    timeout = int(pol.get("tests", {}).get("timeout_secs", 900))
    if ctx.get("skip_tests"):
        return GateResult("tests", False, "skipped by --skip-tests "
                          "(recorded in the changelog)", skipped=True)
    try:
        proc = subprocess.run([sys.executable, *cmd],
                              cwd=str(fingerprint.REPO_ROOT),
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return GateResult("tests", False, f"timed out after {timeout}s")
    tail = (proc.stdout or proc.stderr or "").strip().split("\n")[-1][:120]
    ok = proc.returncode == 0
    return GateResult("tests", ok, tail if ok else f"FAILED: {tail}",
                      {"returncode": proc.returncode, "summary": tail})


def _gate_scorecard(v, pol, ctx) -> GateResult:
    ev = pol.get("evidence", {})
    accept = set(ev.get("accept_tiers", ["full"]))
    want_fresh = bool(ev.get("require_fresh", True))
    allow_dirty = bool(ev.get("allow_dirty_tree", False))

    cards = sc_mod.for_variant(v.name)
    if not cards:
        return GateResult("scorecard", False,
                          f"no scorecard on disk. Run `bench run {v.name}`.")

    now_hash = ctx["config_hash"]
    base_hash = ctx.get("base_config_hash")
    rejected: list[str] = []
    for c in cards:
        if c.tier not in accept:
            rejected.append(f"{c.tier} tier not in accept_tiers {sorted(accept)}")
            continue
        if not c.is_evidence:
            rejected.append(f"{c.tier} run is stamped is_evidence=false")
            continue
        if c.dirty and not allow_dirty:
            rejected.append(f"{c.tier} run came from a dirty tree "
                            f"({c.engine_sha})")
            continue
        if want_fresh:
            why = c.is_stale_for(now_hash, base_hash,
                                 ops=ctx.get("ops"))
            if why:
                rejected.append(f"{c.tier} run is stale: {why}")
                continue
        d = c.delta()
        summary = ", ".join(f"{k} {d[k]:+.3f}" for k in
                            ("sharpe", "ret_per_maxdd") if k in d)
        return GateResult("scorecard", True,
                          f"{c.tier} vs {c.base} ({c.engine_sha}) -- {summary}",
                          {"card": c})

    return GateResult("scorecard", False,
                      f"{len(cards)} scorecard(s) on disk, none usable: "
                      + "; ".join(rejected[:4]))


def _gate_replay(v, pol, ctx) -> GateResult:
    from bench import replay as replay_mod
    reps = replay_mod.for_variant(v.name)
    if not reps:
        return GateResult("replay", False,
                          f"no replay on disk. Run `bench replay {v.name}` -- "
                          f"required because this variant touches "
                          f"{', '.join(ctx.get('replay_reasons', ['liquidity']))}.")
    r = reps[0]
    if r.get("config_hash") and r["config_hash"] != ctx["config_hash"]:
        return GateResult("replay", False,
                          "the newest replay predates the variant's current "
                          "knobs -- re-run it")
    return GateResult("replay", True,
                      f"{r.get('n_days', 0)} live day(s) replayed, decisions "
                      f"differed on {r.get('n_differed', 0)} of "
                      f"{r.get('n_base_entries', 0)} base entries",
                      {"replay": r})


def _gate_verdict(v, pol, ctx) -> GateResult:
    text = (ctx.get("verdict") or "").strip()
    if not text:
        return GateResult("verdict", False,
                          "no --verdict given. The machine checks that evidence "
                          "exists and is fresh; whether the numbers are GOOD is "
                          "a human call, and it gets recorded in the master "
                          "config beside the knob.")
    if len(text.split()) < 4:
        return GateResult("verdict", False,
                          f"--verdict is {len(text.split())} words. Write the "
                          f"sentence you would want to read in six months.")
    return GateResult("verdict", True, f"\"{text[:70]}\"", {"verdict": text})


def _gate_thresholds(v, pol, ctx) -> GateResult:
    """Numeric bars. Inert until `[thresholds]` in policy.toml is filled in."""
    th = pol.get("thresholds") or {}
    if not th:
        return GateResult("thresholds", True,
                          "no numeric bar set -- [thresholds] in "
                          "bench/policy.toml is empty by design (owner pinned "
                          "the bar 2026-08-14)")
    card = ctx.get("card")
    if card is None:
        return GateResult("thresholds", False,
                          "thresholds are set but there is no usable scorecard "
                          "to check them against")
    d = card.delta()
    fails = [f"{k}: {d.get(k)} vs required > {req}"
             for k, req in th.items()
             if d.get(k) is None or d[k] <= float(req)]
    if fails:
        return GateResult("thresholds", False, "; ".join(fails))
    return GateResult("thresholds", True,
                      ", ".join(f"{k} {d[k]:+.3f} > {th[k]}" for k in th))


GATES = {
    "tests": _gate_tests,
    "scorecard": _gate_scorecard,
    "replay": _gate_replay,
    "verdict": _gate_verdict,
    "thresholds": _gate_thresholds,
}


# ---------------------------------------------------------------------------

def required_gates(v, pol: dict) -> tuple[list[str], list[str]]:
    """(gate names, reasons the conditional ones fired)."""
    ev = pol.get("evidence", {})
    names = list(ev.get("required", ["tests", "scorecard", "verdict"]))
    reasons: list[str] = []
    touches = {str(t).lower() for t in (v.touches or [])}
    for key, block in (pol.get("conditional") or {}).items():
        hit = touches & {str(t).lower() for t in block.get("when_touches", [])}
        if hit:
            reasons.extend(sorted(hit))
            for g in block.get("require", []):
                if g not in names:
                    names.append(g)
    if "thresholds" not in names and (pol.get("thresholds") or {}):
        names.append("thresholds")
    return names, reasons


def check(variant_name: str, verdict: str = "", skip_tests: bool = False,
          policy_path: Path | None = None) -> PolicyReport:
    """Run every required gate and return the full picture."""
    pol = load_policy(policy_path)
    v = bench_variant.load(variant_name)
    from bench import config as bench_config

    ctx = {
        "verdict": verdict,
        "skip_tests": skip_tests,
        "config_hash": fingerprint.config_hash(
            bench_config.knobs_for(variant_name)),
        "base_config_hash": fingerprint.config_hash(
            bench_config.knobs_for(v.base or bench_variant.MASTER_VARIANT)),
        # The conditions of the experiment, not the strategy. A variant may
        # override the shared [ops]; when it does, the knobs do not move, so
        # without this a scorecard run at the OLD width stays "fresh" and can
        # satisfy this very gate. See Scorecard.is_stale_for.
        "ops": bench_config.ops_for(variant_name),
    }
    names, reasons = required_gates(v, pol)
    ctx["replay_reasons"] = reasons

    report = PolicyReport(variant=variant_name)
    for name in names:
        fn = GATES.get(name)
        if fn is None:
            report.gates.append(GateResult(
                name, False, f"unknown gate `{name}` in policy.toml. Known: "
                             f"{', '.join(sorted(GATES))}"))
            continue
        g = fn(v, pol, ctx)
        report.gates.append(g)
        if name == "scorecard" and g.passed:
            ctx["card"] = g.evidence.get("card")
    return report
