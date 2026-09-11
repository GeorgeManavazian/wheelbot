"""`variants/frozen.lock.json` -- the machine-readable history of the live bot.

WHY A LOCK FILE EXISTS BESIDE THE MASTER TOML. `variants/frozen.toml` is
readable and hand-editable, and hand-editable is precisely the risk: the whole
point of the bench is that the live config changes only through `bench promote`,
with evidence. A lock file makes that enforceable rather than aspirational --
`bench doctor` asserts the two agree, so a hand-edit to `frozen.toml` is caught
by the pre-push hook before it ships.

It is also the changelog. Every promotion appends an entry with the date, the
variant, the verdict, the evidence, and the knobs that actually moved. The git
diff of this file is therefore a complete, dated record of what changed about
the bot and why -- the thing that currently has to be reconstructed from
comment archaeology in the live runner.

The first history entry is the migration itself, and it never changes: it is
the proof that moving the config out of a Python literal was a no-op.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LOCK_PATH = Path(__file__).resolve().parent.parent / "variants" / "frozen.lock.json"
SCHEMA = 1


def load(path: Path | None = None) -> dict:
    p = Path(path or LOCK_PATH)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing. It is the record of what the live bot runs and how it "
            f"got that way -- regenerate with `bench lock --init` ONLY if you "
            f"accept losing the history, which includes the migration proof.")
    return json.loads(p.read_text(encoding="utf-8"))


def current(path: Path | None = None) -> dict[str, Any]:
    return load(path).get("current", {})


def save(doc: dict, path: Path | None = None) -> Path:
    p = Path(path or LOCK_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    tmp.replace(p)
    return p


def new_doc(knobs: dict, note: str, by: str = "migration",
            date: str = "") -> dict:
    return {"schema": SCHEMA,
            "current": dict(knobs),
            "history": [{"date": date, "by": by, "variant": None,
                         "note": note, "knobs": dict(knobs), "changed": {}}]}


def append(knobs: dict, *, date: str, variant: str, verdict: str,
           evidence: dict, changed: dict, note: str = "",
           path: Path | None = None) -> dict:
    """Record a promotion. `changed` is {knob: [old, new]} -- what actually
    moved, which is what a reader wants first."""
    doc = load(path)
    doc["current"] = dict(knobs)
    doc.setdefault("history", []).append({
        "date": date, "by": "promote", "variant": variant,
        "verdict": verdict, "evidence": evidence,
        "changed": {k: list(v) for k, v in changed.items()},
        "note": note, "knobs": dict(knobs)})
    save(doc, path)
    return doc


def changelog_markdown(path: Path | None = None) -> str:
    """The human-readable changelog, GENERATED from the lock.

    Generated, never hand-maintained -- the vault's standing rule ("never
    hand-maintain a catalog") applies to a repo just as well."""
    doc = load(path)
    L = ["# The live wheel bot -- what changed, and why",
         "",
         "GENERATED from `variants/frozen.lock.json` by `bench registry`.",
         "Do not edit by hand. Every entry is one `bench promote`.",
         ""]
    for e in reversed(doc.get("history", [])):
        head = f"## {e.get('date','?')} -- " + (
            f"`{e['variant']}`" if e.get("variant") else e.get("by", "?"))
        L.append(head)
        L.append("")
        if e.get("note"):
            L.append(e["note"])
            L.append("")
        changed = e.get("changed") or {}
        if changed:
            L.append("| knob | from | to |")
            L.append("|---|---|---|")
            for k, (old, new) in changed.items():
                L.append(f"| `{k}` | `{old!r}` | `{new!r}` |")
            L.append("")
        if e.get("verdict"):
            L.append(f"**Verdict.** {e['verdict']}")
            L.append("")
        ev = e.get("evidence") or {}
        if ev:
            bits = [f"{k} `{v}`" for k, v in ev.items()]
            L.append("**Evidence.** " + " · ".join(bits))
            L.append("")
    return "\n".join(L).rstrip() + "\n"
