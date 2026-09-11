"""Blocks: the 41 WheelConfig fields, grouped into 18 mechanics a human can read.

WHY THIS LAYER EXISTS. `bench frozen` prints 16 knobs and their reasoning, which
is honest and still does not answer "what does my bot DO". A knob is not a
mechanic. `liq_min_open_interest` is a number; "refuse contracts nobody trades,
so the backtest stops booking fills that could never have happened" is a thing
you can hold in your head. The catalog (`bench/blocks.toml`) is that translation,
and this module is the code that reads it and asks it questions.

THE INVARIANT THAT MAKES IT WORTH LOOKING AT: **every WheelConfig field belongs
to exactly one block.** `coverage_problems()` returns a non-empty list the moment
a field is unclaimed, double-claimed, or names something the dataclass does not
have, and a test fails on it. Without that rule the block view degrades into a
partial picture, which is worse than no picture -- you would be reading a
complete-looking screen that quietly omits a mechanic the bot is running.

ON / OFF is computed from the EFFECTIVE config, not from what a variant happens
to mention. A variant that sets nothing still runs every engine default, so
asking the resolved knob dict would report half the bot as absent. The state is
read off a real `WheelConfig`; only the *reasoning* comes from the variant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bench import toml_io
from bench import variant as _v

CATALOG_PATH = Path(__file__).resolve().parent / "blocks.toml"

#: How a block decides it is "on". Kept to four kinds on purpose -- an
#: expression language here would be a second config format nobody can review.
ON_KINDS = ("always", "flag", "any_set", "not_value")


@dataclass(frozen=True)
class Category:
    id: str
    name: str
    order: int
    blurb: str


@dataclass(frozen=True)
class Block:
    id: str
    category: str
    order: int
    name: str
    knobs: tuple
    does: str
    why: str
    watch_out: str
    on_when: dict

    def is_on(self, cfg) -> bool:
        """Is this mechanic active in the given WheelConfig?"""
        kind = self.on_when.get("kind", "any_set")
        if kind == "always":
            return True
        if kind == "flag":
            return bool(getattr(cfg, self.on_when["knob"], False))
        if kind == "not_value":
            return getattr(cfg, self.on_when["knob"], None) != self.on_when["value"]
        # any_set: on when any knob it owns carries a real value. NOT
        # "differs from the dataclass default" -- take_profit_pct=0.5 IS the
        # default and the mechanic is plainly on. None and False mean off;
        # everything else means on.
        for k in self.knobs:
            v = getattr(cfg, k, None)
            if v is None or v is False:
                continue
            return True
        return False


@dataclass
class Catalog:
    categories: list
    blocks: list

    def by_id(self, block_id: str) -> Block:
        for b in self.blocks:
            if b.id == block_id:
                return b
        raise KeyError(f"no block `{block_id}`. Known: "
                       f"{', '.join(b.id for b in self.blocks)}")

    def category(self, cat_id: str) -> Category:
        for c in self.categories:
            if c.id == cat_id:
                return c
        raise KeyError(f"no category `{cat_id}`")

    def ordered(self) -> list:
        """[(Category, [Block, ...]), ...] in display order."""
        out = []
        for c in sorted(self.categories, key=lambda c: c.order):
            bs = sorted((b for b in self.blocks if b.category == c.id),
                        key=lambda b: b.order)
            out.append((c, bs))
        return out

    def owner_of(self, knob: str) -> Block | None:
        for b in self.blocks:
            if knob in b.knobs:
                return b
        return None


def load_catalog(path: Path | None = None) -> Catalog:
    raw = toml_io.load(path or CATALOG_PATH)
    cats = [Category(id=c["id"], name=c["name"], order=int(c.get("order", 0)),
                     blurb=c.get("blurb", ""))
            for c in raw.get("categories", [])]
    blocks = []
    for b in raw.get("blocks", []):
        blocks.append(Block(
            id=b["id"], category=b["category"], order=int(b.get("order", 0)),
            name=b["name"], knobs=tuple(b.get("knobs", [])),
            does=b.get("does", ""), why=b.get("why", "").strip(),
            watch_out=b.get("watch_out", ""),
            on_when=dict(b.get("on_when") or {"kind": "any_set"})))
    return Catalog(categories=cats, blocks=blocks)


# ---------------------------------------------------------------------------
# The invariant
# ---------------------------------------------------------------------------

def coverage_problems(catalog: Catalog | None = None) -> list[str]:
    """Every WheelConfig field claimed by exactly one block, and nothing else.

    This is the whole trustworthiness argument for the block view. If it can
    return an empty list, the screen the owner reads is complete."""
    catalog = catalog or load_catalog()
    fields = set(_v.wheel_fields())
    problems: list[str] = []

    seen: dict[str, str] = {}
    for b in catalog.blocks:
        for k in b.knobs:
            if k in seen:
                problems.append(
                    f"field `{k}` is claimed by both `{seen[k]}` and `{b.id}` -- "
                    f"a field in two blocks shows up twice on screen and is "
                    f"toggled by whichever the reader clicks last")
            seen[k] = b.id
            if k not in fields:
                problems.append(
                    f"block `{b.id}` claims `{k}`, which is not a WheelConfig "
                    f"field (renamed? removed?)")

    for k in sorted(fields - set(seen)):
        problems.append(
            f"field `{k}` belongs to NO block, so it is invisible in the block "
            f"view while the bot still runs it. Add it to bench/blocks.toml.")

    known_cats = {c.id for c in catalog.categories}
    for b in catalog.blocks:
        if b.category not in known_cats:
            problems.append(f"block `{b.id}` has unknown category `{b.category}`")
        if b.on_when.get("kind") not in ON_KINDS:
            problems.append(
                f"block `{b.id}`: on_when.kind `{b.on_when.get('kind')}` is not "
                f"one of {', '.join(ON_KINDS)}")
        if b.on_when.get("kind") in ("flag", "not_value"):
            knob = b.on_when.get("knob")
            if knob not in b.knobs:
                problems.append(
                    f"block `{b.id}`: on_when reads `{knob}`, which it does not "
                    f"own -- a block must decide its own state from its own knobs")
        if not b.does.strip():
            problems.append(f"block `{b.id}` has no `does`")
        if not b.why.strip():
            problems.append(f"block `{b.id}` has no `why`")
        if b.does.count(".") > 1:
            problems.append(
                f"block `{b.id}`: `does` is more than one sentence -- it is the "
                f"line read at a glance, so it stays one")

    ids = [b.id for b in catalog.blocks]
    for dup in {i for i in ids if ids.count(i) > 1}:
        problems.append(f"duplicate block id `{dup}`")
    return problems


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class KnobState:
    name: str
    value: Any
    why: str
    since: str
    origin: str          # which variant set it, or "" for the engine default
    is_default: bool


@dataclass
class BlockState:
    block: Block
    on: bool
    knobs: list = field(default_factory=list)   # [KnobState]

    @property
    def settings(self) -> dict:
        return {k.name: k.value for k in self.knobs}

    @property
    def touched(self) -> list:
        """Knobs this variant (or its lineage) actually set, as opposed to
        inherited from the engine defaults."""
        return [k for k in self.knobs if not k.is_default]


def state_for(variant_name: str, catalog: Catalog | None = None) -> list:
    """Every block's on/off state and settings under one variant.

    Values come from a real WheelConfig, so engine defaults are included --
    reading the variant's own knob dict would report the untouched half of the
    bot as if it were absent."""
    from dataclasses import fields as dc_fields
    from src.engine_v2.options.wheel import WheelConfig
    from bench import config as bench_config

    catalog = catalog or load_catalog()
    cfg = bench_config.to_wheel_config(variant_name)
    resolved = _v.resolve(variant_name)
    defaults = {f.name: f.default for f in dc_fields(WheelConfig)}

    out = []
    for c, blocks in catalog.ordered():
        for b in blocks:
            ks = []
            for k in b.knobs:
                prov = resolved.provenance.get(k)
                ks.append(KnobState(
                    name=k, value=getattr(cfg, k, None),
                    why=(prov.why if prov else ""),
                    since=(prov.since or "" if prov else ""),
                    origin=resolved.origin.get(k, ""),
                    is_default=(k not in resolved.knobs
                                and getattr(cfg, k, None) == defaults.get(k))))
            out.append(BlockState(block=b, on=b.is_on(cfg), knobs=ks))
    return out


def diff(variant_a: str, variant_b: str, catalog: Catalog | None = None) -> list:
    """[(Block, state_a, state_b), ...] for blocks that differ in ANY way --
    turned on/off, or the same mechanic tuned differently. Both matter: a
    changed threshold is as much a change to the bot as a flipped flag."""
    catalog = catalog or load_catalog()
    a = {s.block.id: s for s in state_for(variant_a, catalog)}
    b = {s.block.id: s for s in state_for(variant_b, catalog)}
    out = []
    for blk in catalog.blocks:
        sa, sb = a[blk.id], b[blk.id]
        if sa.on != sb.on or sa.settings != sb.settings:
            out.append((blk, sa, sb))
    return out


# ---------------------------------------------------------------------------
# Terminal rendering
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    if v is None:
        return "off"
    if v is True:
        return "on"
    if v is False:
        return "off"
    return str(v)


def render(variant_name: str, verbose: bool = False, width: int = 84) -> str:
    states = state_for(variant_name)
    catalog = load_catalog()
    on = sum(1 for s in states if s.on)
    L = ["=" * width,
         f"  {variant_name}  --  {on} of {len(states)} blocks on",
         "=" * width]
    by_cat: dict[str, list] = {}
    for s in states:
        by_cat.setdefault(s.block.category, []).append(s)
    for c in sorted(catalog.categories, key=lambda c: c.order):
        rows = by_cat.get(c.id) or []
        if not rows:
            continue
        L.append("")
        L.append(f"  {c.name.upper()}")
        L.append(f"  {c.blurb}")
        L.append("  " + "-" * (width - 4))
        for s in rows:
            mark = "[on] " if s.on else "[  ] "
            L.append(f"  {mark}{s.block.name}")
            L.append(f"        {s.block.does}")
            sett = ", ".join(f"{k.name}={_fmt(k.value)}" for k in s.knobs)
            L.append(f"        {sett}")
            if verbose:
                L.append("")
                for line in s.block.why.split("\n"):
                    L.append(f"        {line}" if line else "")
                if s.block.watch_out:
                    L.append(f"        WATCH OUT: {s.block.watch_out}")
            L.append("")
    L.append("=" * width)
    return "\n".join(L)


def render_diff(variant_a: str, variant_b: str, width: int = 84) -> str:
    rows = diff(variant_a, variant_b)
    if not rows:
        return f"`{variant_b}` changes no block versus `{variant_a}`."
    L = ["=" * width,
         f"  {variant_b}  vs  {variant_a}  --  {len(rows)} block(s) differ",
         "=" * width]
    for blk, sa, sb in rows:
        flip = ""
        if sa.on != sb.on:
            flip = "  TURNED ON" if sb.on else "  TURNED OFF"
        L.append("")
        L.append(f"  {blk.name}   [{blk.category}]{flip}")
        L.append(f"    {blk.does}")
        for k in blk.knobs:
            va, vb = sa.settings.get(k), sb.settings.get(k)
            if va != vb:
                L.append(f"      {k:<32} {_fmt(va)}  ->  {_fmt(vb)}")
    L.append("=" * width)
    return "\n".join(L)
