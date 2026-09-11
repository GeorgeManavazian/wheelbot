"""A variant is one paragraph of the essay: a named, committed, reviewable
change to the wheel bot, with the reason for every knob written beside it.

THE PROBLEM THIS EXISTS TO SOLVE. Before this module, a "variant" was a dict
literal inside an untracked file in `scratchpad/`. Ten of them lived in
`run_wheel_grid_n1.py`; their verdicts were written to `results/`, which is
gitignored. Every experiment therefore had to be re-derived from prose, from
memory, or not at all -- and the reasoning that survived did so only because
someone hand-wrote it into a comment in `live/run_daily.FROZEN`.

THE SHAPE. One file per variant under `variants/`. It carries:

  * `question`   -- what this paragraph must PROVE. Not a description. If you
                    cannot state the question, the variant is not ready.
  * `base`       -- what it is a diff AGAINST. Usually `frozen` (the live bot).
  * `knobs`      -- the diff itself, one table per knob, each with a `why`.
  * `code`       -- optional: the branch and the new flags it introduces, for a
                    variant that needs engine code that does not exist yet.
  * `evidence`   -- appended by the bench, never hand-written.

THE ONE RULE THIS MODULE ENFORCES HARDEST: **every knob must have a `why`.**
A knob without a reason is how `min_ann_yield_on_collateral` sat in the engine
from commit 682823e while `FROZEN` never set it, and `_STATUS` reported the
config work as done. `validate()` refuses a variant whose knob has an empty
`why`, and the promotion gate refuses to merge one.

INHERITANCE. `plain` is the root: the WheelConfig dataclass defaults, every
flag off, byte-identical to the original engine. `frozen` derives from `plain`
and is the live bot. A variant normally derives from `frozen`, but may derive
from another variant to stack two paragraphs. Resolution walks the lineage
root-first and records, per knob, which ancestor last set it -- so
`bench frozen` can print not just the live value but where it came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from bench import toml_io

# ---------------------------------------------------------------------------
# Locations. Resolved from THIS file, never from the working directory: the
# live bot imports bench.config from a systemd unit whose cwd is not the repo.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
VARIANTS_DIR = REPO_ROOT / "variants"
ARCHIVE_DIR = VARIANTS_DIR / "_archive"

ROOT_VARIANT = "plain"      # the all-defaults baseline; has no base
MASTER_VARIANT = "frozen"   # the live bot -- the essay being submitted

STATUSES = ("draft", "testing", "rejected", "promoted", "superseded")

#: A knob whose value is None is written `unset = true`, because TOML has no
#: null. This is not a workaround dressed as a feature: `liq_max_rel_spread`
#: is deliberately None in the live config and carries 25 lines of measurement
#: explaining why, so "explicitly off, and here is the reason" is a state the
#: schema must be able to express distinctly from "never mentioned".
UNSET = object()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Knob:
    """One knob in one variant: the value, and the reason it has that value."""
    name: str
    value: Any                  # None when `unset = true`
    why: str
    since: str | None = None    # date this value was decided
    source: str | None = None   # "owner ruling" / a scorecard id / a vault note

    @property
    def is_unset(self) -> bool:
        return self.value is None


@dataclass
class Variant:
    name: str
    question: str
    base: str | None
    status: str
    knobs: dict[str, Knob] = field(default_factory=dict)
    ops: dict = field(default_factory=dict)
    code: dict = field(default_factory=dict)
    evidence: list[dict] = field(default_factory=list)
    created: str = ""
    updated: str = ""
    touches: list[str] = field(default_factory=list)
    note: str = ""
    path: Path | None = None

    @property
    def new_flags(self) -> list[str]:
        """Flags this variant's own branch introduces to WheelConfig. Empty for
        a config-only variant."""
        return list(self.code.get("new_flags", []))

    @property
    def branch(self) -> str | None:
        return self.code.get("branch") or None

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict:
        """Round-trippable dict in a deliberate key order: identity, then the
        question, then the diff. A variant file is read by a human first."""
        d: dict = {"name": self.name, "question": self.question}
        if self.base is not None:
            d["base"] = self.base
        d["status"] = self.status
        if self.created:
            d["created"] = self.created
        if self.updated:
            d["updated"] = self.updated
        if self.touches:
            d["touches"] = list(self.touches)
        if self.note:
            d["note"] = self.note
        if self.code:
            d["code"] = dict(self.code)
        if self.ops:
            d["ops"] = dict(self.ops)
        if self.knobs:
            knobs: dict = {}
            for name, k in self.knobs.items():
                entry: dict = {}
                if k.is_unset:
                    entry["unset"] = True
                else:
                    entry["value"] = k.value
                entry["why"] = k.why
                if k.since:
                    entry["since"] = k.since
                if k.source:
                    entry["source"] = k.source
                knobs[name] = entry
            d["knobs"] = knobs
        if self.evidence:
            d["evidence"] = [dict(e) for e in self.evidence]
        return d

    def save(self, path: str | Path | None = None, header: str = "") -> Path:
        target = Path(path) if path else (self.path or VARIANTS_DIR / f"{self.name}.toml")
        toml_io.dump(self.to_dict(), target, header=header)
        self.path = target
        return target


def from_dict(d: dict, path: Path | None = None) -> Variant:
    """Build a Variant from parsed TOML. Structural problems raise here;
    semantic ones (unknown knob, missing `why`) are reported by validate(),
    so a broken file can still be loaded and explained rather than just
    exploding on read."""
    knobs: dict[str, Knob] = {}
    raw_knobs = d.get("knobs") or {}
    if not isinstance(raw_knobs, dict):
        raise ValueError("`knobs` must be a table of tables")
    for name, spec in raw_knobs.items():
        if not isinstance(spec, dict):
            raise ValueError(
                f"knob `{name}`: expected a table with `value` and `why`, got "
                f"{type(spec).__name__}. Write it as:\n"
                f"  [knobs.{name}]\n  value = ...\n  why = \"\"\"...\"\"\"")
        if spec.get("unset"):
            value = None
        elif "value" in spec:
            value = spec["value"]
        else:
            raise ValueError(f"knob `{name}`: needs `value` or `unset = true`")
        knobs[name] = Knob(name=name, value=value, why=str(spec.get("why", "")),
                           since=spec.get("since"), source=spec.get("source"))

    return Variant(
        name=str(d.get("name", "")),
        question=str(d.get("question", "")),
        base=d.get("base"),
        status=str(d.get("status", "draft")),
        knobs=knobs,
        ops=dict(d.get("ops") or {}),
        code=dict(d.get("code") or {}),
        evidence=list(d.get("evidence") or []),
        created=str(d.get("created", "")),
        updated=str(d.get("updated", "")),
        touches=list(d.get("touches") or []),
        note=str(d.get("note", "")),
        path=path,
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def variant_path(name: str, root: Path | None = None) -> Path:
    root = root or VARIANTS_DIR
    live = root / f"{name}.toml"
    if live.exists():
        return live
    archived = root / "_archive" / f"{name}.toml"
    if archived.exists():
        return archived
    return live      # canonical location, for the error message


def load(name: str, root: Path | None = None) -> Variant:
    p = variant_path(name, root)
    if not p.exists():
        known = ", ".join(sorted(all_variants(root))) or "(none)"
        raise FileNotFoundError(
            f"no variant named `{name}` (looked in {p}). Known: {known}")
    v = from_dict(toml_io.load(p), path=p)
    if v.name != name:
        raise ValueError(
            f"{p}: `name = \"{v.name}\"` does not match the filename `{name}`. "
            f"The filename is the identity -- rename one of them.")
    return v


def all_variants(root: Path | None = None) -> dict[str, Variant]:
    """Every variant, live and archived, keyed by name. Archived ones are
    included on purpose: a rejected experiment is knowledge, and the registry
    that lists it is the whole point of committing it."""
    root = root or VARIANTS_DIR
    out: dict[str, Variant] = {}
    for d in (root, root / "_archive"):
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.toml")):
            try:
                v = from_dict(toml_io.load(p), path=p)
            except Exception as e:                      # noqa: BLE001
                raise ValueError(f"{p}: {e}") from e
            out.setdefault(v.name or p.stem, v)
    return out


# ---------------------------------------------------------------------------
# Lineage and resolution
# ---------------------------------------------------------------------------

def lineage(name: str, root: Path | None = None) -> list[Variant]:
    """[root_ancestor, ..., name]. Raises on a cycle -- a variant whose base
    chain loops would otherwise resolve to whichever value the walk happened
    to see last, which is a silent wrong answer."""
    chain: list[Variant] = []
    seen: list[str] = []
    cur: str | None = name
    while cur:
        if cur in seen:
            loop = " -> ".join(seen + [cur])
            raise ValueError(f"variant base cycle: {loop}")
        seen.append(cur)
        v = load(cur, root)
        chain.append(v)
        cur = v.base or None
    chain.reverse()
    return chain


@dataclass
class Resolved:
    """The flattened result of walking a lineage."""
    name: str
    knobs: dict[str, Any]                 # field -> value (None means None)
    provenance: dict[str, Knob]           # field -> the Knob that won
    origin: dict[str, str]                # field -> variant that set it
    ops: dict
    lineage: list[str]
    new_flags: list[str]

    def diff_against(self, other: "Resolved") -> dict[str, tuple]:
        """{field: (other_value, self_value)} for every field they disagree on.
        This is what `bench diff` prints and what `promote` writes."""
        out: dict[str, tuple] = {}
        for k in sorted(set(self.knobs) | set(other.knobs)):
            a, b = other.knobs.get(k, UNSET), self.knobs.get(k, UNSET)
            if a is UNSET and b is UNSET:
                continue
            if a is UNSET or b is UNSET or a != b:
                out[k] = (None if a is UNSET else a, None if b is UNSET else b)
        return out


def resolve(name: str, root: Path | None = None) -> Resolved:
    """Flatten a variant's lineage into one config. Later ancestors override
    earlier ones; `origin` remembers who won each knob so the master config can
    be printed with its provenance intact."""
    chain = lineage(name, root)
    knobs: dict[str, Any] = {}
    prov: dict[str, Knob] = {}
    origin: dict[str, str] = {}
    ops: dict = {}
    flags: list[str] = []
    for v in chain:
        for kname, k in v.knobs.items():
            knobs[kname] = k.value
            prov[kname] = k
            origin[kname] = v.name
        ops.update(v.ops)
        for f in v.new_flags:
            if f not in flags:
                flags.append(f)
    return Resolved(name=name, knobs=knobs, provenance=prov, origin=origin,
                    ops=ops, lineage=[v.name for v in chain], new_flags=flags)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def wheel_fields() -> dict[str, Any]:
    """{field_name: dataclasses.Field} for WheelConfig, imported lazily so this
    module stays usable (for `bench validate`) even from a checkout where the
    engine is mid-edit."""
    from src.engine_v2.options.wheel import WheelConfig
    return {f.name: f for f in fields(WheelConfig)}


_TYPE_MAP = {
    "bool": (bool,),
    "int": (int,),
    "float": (int, float),      # an int literal is a fine float in TOML
    "str": (str,),
    # A TOML array. Added 2026-08-15 for the weather-gate BAND knobs
    # (`chop_drawdown_band`, `chop_dip_z_band`), which are [lo, hi] pairs
    # rather than thresholds -- deliberately, because a band's neighbourhood
    # can be checked for a plateau and a threshold's cannot. The declared type
    # is `list | None`; the engine checks length and ordering where it is used.
    "list": (list,),
}


def _type_ok(declared: str, value: Any) -> bool:
    """Loose structural check against the dataclass annotation string (the
    module uses `from __future__ import annotations`, so field.type is text).
    Catches the real mistakes -- a string where a float belongs, `true` where a
    threshold belongs -- without reimplementing typing."""
    parts = [p.strip() for p in str(declared).split("|")]
    if value is None:
        return "None" in parts
    for p in parts:
        allowed = _TYPE_MAP.get(p)
        if not allowed:
            continue
        if p == "int" and isinstance(value, bool):
            continue            # bool is an int subclass; never accept it as one
        if p == "float" and isinstance(value, bool):
            continue
        if isinstance(value, allowed):
            return True
    return False


def validate(v: Variant, root: Path | None = None,
             known_fields: dict | None = None) -> list[str]:
    """Return a list of problems, empty if the variant is sound.

    Reports EVERY problem rather than raising on the first: a half-written
    variant should get one complete list of what to fix, not a game of
    whack-a-mole."""
    problems: list[str] = []
    known = known_fields if known_fields is not None else wheel_fields()

    if not v.name:
        problems.append("`name` is empty")
    if v.path and v.path.stem != v.name:
        problems.append(f"`name` ({v.name}) does not match filename ({v.path.stem})")
    if v.status not in STATUSES:
        problems.append(f"status `{v.status}` is not one of {', '.join(STATUSES)}")

    # `plain` is the control arm and `frozen` is the essay itself; neither is a
    # paragraph, so neither has a question to prove. Every other variant does.
    if v.name not in (ROOT_VARIANT, MASTER_VARIANT) and not v.question.strip():
        problems.append(
            "`question` is empty. A variant states what it must PROVE -- if you "
            "cannot write the question, the paragraph is not ready to write.")

    if v.name == ROOT_VARIANT:
        if v.base:
            problems.append(f"`{ROOT_VARIANT}` is the root and must have no base")
    elif not v.base:
        problems.append("`base` is required (usually `frozen`)")
    else:
        try:
            lineage(v.name, root)
        except (FileNotFoundError, ValueError) as e:
            problems.append(str(e))

    declared_new = set(v.new_flags)
    for name, k in v.knobs.items():
        if not k.why.strip():
            problems.append(
                f"knob `{name}` has no `why`. Every knob carries its reason -- "
                f"that discipline is why FROZEN is readable and scratchpad/ is not.")
        if name in known:
            if not _type_ok(known[name].type, k.value):
                problems.append(
                    f"knob `{name}`: {k.value!r} does not fit the declared type "
                    f"`{known[name].type}`")
        elif name in declared_new:
            problems.append(
                f"knob `{name}` is declared in [code].new_flags but is not yet a "
                f"WheelConfig field -- check out `{v.branch}` before running this "
                f"variant. (Not an error in the variant file; a state of the world.)")
        else:
            close = _suggest(name, known)
            problems.append(
                f"knob `{name}` is not a WheelConfig field"
                + (f" -- did you mean `{close}`?" if close else "")
                + ". If its code is on a branch, declare it in [code].new_flags.")

    for f in declared_new:
        if f not in v.knobs:
            problems.append(
                f"[code].new_flags lists `{f}` but no [knobs.{f}] sets it -- "
                f"a flag introduced and never exercised proves nothing.")
    if declared_new and not v.branch:
        problems.append("[code].new_flags is set but [code].branch is missing")

    return problems


def _suggest(name: str, known: dict) -> str | None:
    """Cheap nearest-name hint. A typo'd knob would otherwise fail as
    'unknown field' with no clue which of 41 fields was meant."""
    import difflib
    hits = difflib.get_close_matches(name, list(known), n=1, cutoff=0.6)
    return hits[0] if hits else None
