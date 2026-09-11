"""The block catalog, and the invariant that makes it worth reading.

The load-bearing test in this file is `test_every_wheelconfig_field_belongs_to_
exactly_one_block`. The block view's entire value is the promise that it shows
EVERYTHING the bot does. A view that is 95% complete is worse than no view,
because it looks complete -- the owner would read a full-looking screen and not
know a mechanic was missing from it. That test is what turns the promise into
something the fast suite enforces.
"""
from __future__ import annotations

import pytest

from bench import blocks, toml_io
from bench import variant as V


@pytest.fixture(scope="module")
def catalog():
    return blocks.load_catalog()


# -- the invariant -----------------------------------------------------------

def test_every_wheelconfig_field_belongs_to_exactly_one_block(catalog):
    problems = blocks.coverage_problems(catalog)
    assert problems == [], (
        "the block catalog no longer describes the whole bot:\n  - "
        + "\n  - ".join(problems))


def test_the_catalog_claims_exactly_as_many_fields_as_wheelconfig_has(catalog):
    claimed = [k for b in catalog.blocks for k in b.knobs]
    assert len(claimed) == len(set(claimed)), "a field is claimed twice"
    assert set(claimed) == set(V.wheel_fields())


def test_a_field_left_out_is_caught(catalog, tmp_path):
    """Proof the guard actually fires -- a coverage test that cannot fail is
    decoration."""
    raw = toml_io.load(blocks.CATALOG_PATH)
    raw["blocks"] = [b for b in raw["blocks"] if b["id"] != "earnings-blackout"]
    p = tmp_path / "blocks.toml"
    toml_io.dump(raw, p)
    problems = blocks.coverage_problems(blocks.load_catalog(p))
    assert any("earnings_blackout" in x and "NO block" in x for x in problems)


def test_a_field_claimed_twice_is_caught(tmp_path):
    raw = toml_io.load(blocks.CATALOG_PATH)
    for b in raw["blocks"]:
        if b["id"] == "take-profit":
            b["knobs"] = list(b["knobs"]) + ["put_delta"]
    p = tmp_path / "blocks.toml"
    toml_io.dump(raw, p)
    problems = blocks.coverage_problems(blocks.load_catalog(p))
    assert any("claimed by both" in x for x in problems)


def test_a_renamed_field_is_caught(tmp_path):
    raw = toml_io.load(blocks.CATALOG_PATH)
    for b in raw["blocks"]:
        if b["id"] == "take-profit":
            b["knobs"] = ["take_profit_percentage"]
    p = tmp_path / "blocks.toml"
    toml_io.dump(raw, p)
    problems = blocks.coverage_problems(blocks.load_catalog(p))
    assert any("not a WheelConfig field" in x for x in problems)


# -- the writing -------------------------------------------------------------

def test_every_block_says_what_it_does_in_one_sentence(catalog):
    for b in catalog.blocks:
        assert b.does.strip(), f"{b.id} has no `does`"
        assert b.does.count(".") == 1, (
            f"{b.id}: `does` must be exactly one sentence -- it is the line read "
            f"at a glance. Got: {b.does!r}")


def test_every_block_explains_why_it_exists(catalog):
    for b in catalog.blocks:
        words = len(b.why.split())
        assert words >= 40, (
            f"{b.id}: `why` is {words} words. This field is the answer to "
            f'"why did we even build this" -- a one-liner does not answer it.')


def test_no_block_repeats_a_field_name_in_its_plain_sentence(catalog):
    """`does` is read by someone who does not know the code. A field name in it
    means they have to go look something up, which is the thing this catalog
    exists to stop."""
    names = set(V.wheel_fields())
    for b in catalog.blocks:
        repeated = [n for n in names if n in b.does]
        assert not repeated, f"{b.id}: `does` mentions {repeated}"


def test_block_names_are_short_and_human(catalog):
    for b in catalog.blocks:
        assert 1 <= len(b.name.split()) <= 6, f"{b.id}: name {b.name!r}"
        assert "_" not in b.name, f"{b.id}: name looks like an identifier"


def test_categories_are_ordered_and_unique(catalog):
    ids = [c.id for c in catalog.categories]
    assert len(ids) == len(set(ids))
    orders = [c.order for c in catalog.categories]
    assert len(orders) == len(set(orders)), "two categories share an order"


def test_every_category_has_at_least_one_block(catalog):
    used = {b.category for b in catalog.blocks}
    empty = [c.id for c in catalog.categories if c.id not in used]
    assert not empty, f"categories with no blocks: {empty}"


def test_blocks_are_uniquely_ordered_within_a_category(catalog):
    for c, bs in catalog.ordered():
        orders = [b.order for b in bs]
        assert len(orders) == len(set(orders)), \
            f"category {c.id} has two blocks at the same order"


# -- state -------------------------------------------------------------------

def test_the_live_config_turns_on_the_blocks_we_expect():
    on = {s.block.id for s in blocks.state_for("frozen") if s.on}
    # Every one of these is verifiable against variants/frozen.toml.
    assert {"short-put-terms", "take-profit", "chop-weather-gate",
            "earnings-blackout", "yield-floor", "intrinsic-filter",
            "liquidity-floors", "order-size-cap", "covered-call",
            "trading-costs", "account"} <= on
    # And every one of these is off in production, on purpose.
    assert on.isdisjoint({"ranking", "iv-rank-floor", "regime-gates",
                          "assigned-stock-exit", "put-stop",
                          "roll-tested-puts", "liquidate-on-assignment"})


def test_plain_turns_off_every_optional_block():
    """The control arm. Anything still on under `plain` is a mechanic the
    engine does unconditionally, which should only ever be the four core ones."""
    on = {s.block.id for s in blocks.state_for("plain") if s.on}
    assert on == {"short-put-terms", "take-profit", "covered-call",
                  "trading-costs", "account"}


def test_state_reports_engine_defaults_not_only_what_the_variant_sets():
    """`frozen` sets 16 of 41 fields. Reading the variant's knob dict would
    report the other 25 as absent, so most of the bot would be invisible."""
    states = {s.block.id: s for s in blocks.state_for("frozen")}
    acct = states["account"]
    assert acct.settings["contract_multiplier"] == 100     # never set by frozen
    assert acct.on


def test_a_knob_carries_the_reason_the_variant_gave_it():
    states = {s.block.id: s for s in blocks.state_for("frozen")}
    liq = states["liquidity-floors"]
    spread = [k for k in liq.knobs if k.name == "liq_max_rel_spread"][0]
    assert "CHEAPNESS" in spread.why
    assert spread.origin == "frozen"
    assert spread.since == "2026-08-11"


def test_touched_separates_deliberate_settings_from_inherited_ones():
    states = {s.block.id: s for s in blocks.state_for("frozen")}
    costs = states["trading-costs"]
    touched = {k.name for k in costs.touched}
    assert "fees_per_contract" in touched              # frozen sets it
    assert "commission_per_contract" not in touched    # engine default


# -- on/off rules ------------------------------------------------------------

def test_any_set_treats_a_real_value_as_on_even_when_it_equals_the_default():
    """`take_profit_pct` defaults to 0.5 and the mechanic is plainly ON at 0.5.
    An "is it different from the default" rule would report it off."""
    from src.engine_v2.options.wheel import WheelConfig
    blk = blocks.load_catalog().by_id("take-profit")
    assert blk.is_on(WheelConfig(take_profit_pct=0.5)) is True
    assert blk.is_on(WheelConfig(take_profit_pct=None)) is False


def test_any_set_treats_false_as_off():
    from src.engine_v2.options.wheel import WheelConfig
    blk = blocks.load_catalog().by_id("regime-gates")
    assert blk.is_on(WheelConfig()) is False
    assert blk.is_on(WheelConfig(regime_roll_gate=True)) is True


def test_not_value_rule_reads_the_sort_key():
    from src.engine_v2.options.wheel import WheelConfig
    blk = blocks.load_catalog().by_id("ranking")
    assert blk.is_on(WheelConfig(rank_by="vol_pctile")) is False
    assert blk.is_on(WheelConfig(rank_by="iv_rank")) is True


def test_a_block_may_only_decide_its_state_from_a_knob_it_owns(catalog):
    for b in catalog.blocks:
        if b.on_when.get("kind") in ("flag", "not_value"):
            assert b.on_when["knob"] in b.knobs, b.id


# -- diff --------------------------------------------------------------------

def test_diff_reports_a_retuned_block_not_only_a_flipped_one():
    """dte-7 turns nothing on or off -- it retunes one block. A diff that only
    watched on/off would call the two configs identical."""
    rows = blocks.diff("frozen", "dte-7")
    assert len(rows) == 1
    blk, a, b = rows[0]
    assert blk.id == "short-put-terms"
    assert a.on and b.on
    assert a.settings["target_dte"] == 11 and b.settings["target_dte"] == 7


def test_diff_of_a_variant_against_itself_is_empty():
    assert blocks.diff("frozen", "frozen") == []


def test_render_shows_the_on_count_and_every_category(catalog):
    text = blocks.render("frozen")
    # Derived from the catalog, not hardcoded: the number is a property of
    # bench/blocks.toml, and pinning it here only ever means editing this line
    # whenever a mechanic is added -- which teaches "update the number", not
    # "check the catalog". The invariant worth pinning is coverage, and
    # test_every_field_belongs_to_exactly_one_block already pins it.
    assert f"of {len(catalog.blocks)} blocks on" in text
    for c in catalog.categories:
        assert c.name.upper() in text


def test_render_diff_names_the_change():
    text = blocks.render_diff("frozen", "dte-7")
    assert "1 block(s) differ" in text
    assert "11  ->  7" in text
