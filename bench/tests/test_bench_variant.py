"""The variant schema, its inheritance, and -- most of all -- what it refuses.

The validation rules here are the whole discipline of the bench. A variant that
loads but says nothing about WHY is exactly the artifact the scratchpad
harnesses already produce, and the bench is only worth having if it will not
accept one.
"""
from __future__ import annotations

import pytest

from bench import toml_io
from bench import variant as V


@pytest.fixture
def root(tmp_path):
    """A throwaway variants/ tree. Every test writes its own so nothing here
    can touch the real master config."""
    d = tmp_path / "variants"
    d.mkdir()
    toml_io.dump({"name": "plain", "question": "", "status": "promoted",
                  "ops": {"tier": "smoke", "n_slots": 1}}, d / "plain.toml")
    toml_io.dump({"name": "frozen", "question": "", "base": "plain",
                  "status": "promoted",
                  "knobs": {"target_dte": {"value": 11, "why": "live value"},
                            "put_delta": {"value": 0.3, "why": "live value"}}},
                 d / "frozen.toml")
    return d


def write(root, name, **kw):
    d = {"name": name, "question": "does it?", "base": "frozen",
         "status": "testing"}
    d.update(kw)
    toml_io.dump(d, root / f"{name}.toml")
    return name


# -- loading and inheritance ------------------------------------------------

def test_lineage_is_root_first(root):
    write(root, "x", knobs={"target_dte": {"value": 7, "why": "shorter"}})
    assert V.lineage("x", root)[0].name == "plain"
    assert V.resolve("x", root).lineage == ["plain", "frozen", "x"]


def test_a_variant_inherits_its_base_and_overrides_one_knob(root):
    write(root, "x", knobs={"target_dte": {"value": 7, "why": "shorter"}})
    r = V.resolve("x", root)
    assert r.knobs == {"target_dte": 7, "put_delta": 0.3}
    assert r.origin["target_dte"] == "x"
    assert r.origin["put_delta"] == "frozen"


def test_variants_can_stack(root):
    write(root, "a", knobs={"target_dte": {"value": 7, "why": "shorter"}})
    write(root, "b", base="a",
          knobs={"put_delta": {"value": 0.4, "why": "fatter"}})
    r = V.resolve("b", root)
    assert r.knobs == {"target_dte": 7, "put_delta": 0.4}
    assert r.lineage == ["plain", "frozen", "a", "b"]


def test_ops_are_inherited_too(root):
    write(root, "x", knobs={"target_dte": {"value": 7, "why": "s"}})
    assert V.resolve("x", root).ops["tier"] == "smoke"
    write(root, "y", knobs={"target_dte": {"value": 7, "why": "s"}},
          ops={"tier": "live"})
    assert V.resolve("y", root).ops["tier"] == "live"


def test_a_base_cycle_raises_instead_of_resolving_to_whatever_came_last(root):
    write(root, "a", base="b", knobs={"target_dte": {"value": 7, "why": "s"}})
    write(root, "b", base="a", knobs={"target_dte": {"value": 9, "why": "s"}})
    with pytest.raises(ValueError, match="cycle"):
        V.lineage("a", root)


def test_unknown_variant_names_the_ones_that_exist(root):
    with pytest.raises(FileNotFoundError, match="frozen"):
        V.load("nope", root)


def test_diff_against_reports_only_real_differences(root):
    write(root, "x", knobs={"target_dte": {"value": 7, "why": "s"}})
    d = V.resolve("x", root).diff_against(V.resolve("frozen", root))
    assert d == {"target_dte": (11, 7)}


# -- what it refuses --------------------------------------------------------

def test_a_knob_with_no_why_is_rejected(root):
    write(root, "x", knobs={"target_dte": {"value": 7, "why": "  "}})
    problems = V.validate(V.load("x", root), root)
    assert any("has no `why`" in p for p in problems)


def test_a_variant_with_no_question_is_rejected(root):
    write(root, "x", question="",
          knobs={"target_dte": {"value": 7, "why": "s"}})
    problems = V.validate(V.load("x", root), root)
    assert any("`question` is empty" in p for p in problems)


def test_plain_and_frozen_are_exempt_from_the_question_rule(root):
    for n in ("plain", "frozen"):
        assert not any("question" in p
                       for p in V.validate(V.load(n, root), root))


def test_an_unknown_knob_is_rejected_with_a_suggestion(root):
    write(root, "x", knobs={"target_dtee": {"value": 7, "why": "typo"}})
    problems = V.validate(V.load("x", root), root)
    assert any("not a WheelConfig field" in p for p in problems)
    assert any("did you mean `target_dte`" in p for p in problems)


def test_a_wrong_typed_value_is_rejected(root):
    write(root, "x", knobs={"target_dte": {"value": 7.5, "why": "s"}})
    problems = V.validate(V.load("x", root), root)
    assert any("does not fit the declared type" in p for p in problems)


def test_true_is_not_accepted_where_a_number_belongs(root):
    """`bool` is an `int` subclass, so a naive isinstance check would let
    `target_dte = true` through and the engine would select an expiry 1 day
    out."""
    write(root, "x", knobs={"target_dte": {"value": True, "why": "s"}})
    assert any("does not fit" in p for p in V.validate(V.load("x", root), root))


def test_an_unset_knob_is_only_valid_where_none_is(root):
    write(root, "ok", knobs={"take_profit_pct": {"unset": True, "why": "off"}})
    assert V.validate(V.load("ok", root), root) == []
    write(root, "bad", knobs={"target_dte": {"unset": True, "why": "off"}})
    assert any("does not fit" in p for p in V.validate(V.load("bad", root), root))


def test_a_new_flag_must_be_exercised_by_a_knob(root):
    write(root, "x", code={"branch": "feat/y", "new_flags": ["made_up_flag"]},
          knobs={"target_dte": {"value": 7, "why": "s"}})
    problems = V.validate(V.load("x", root), root)
    assert any("never exercised" in p for p in problems)


def test_new_flags_require_a_branch(root):
    write(root, "x", code={"new_flags": ["made_up_flag"]},
          knobs={"made_up_flag": {"value": True, "why": "s"}})
    assert any("branch" in p for p in V.validate(V.load("x", root), root))


def test_a_declared_new_flag_reports_as_a_state_of_the_world(root):
    """A knob whose code is on a branch you have not checked out is not a
    broken variant -- it is a checkout you have not done. The message must say
    which branch."""
    write(root, "x", code={"branch": "feat/exit", "new_flags": ["made_up_flag"]},
          knobs={"made_up_flag": {"value": True, "why": "s"}})
    problems = V.validate(V.load("x", root), root)
    assert any("check out `feat/exit`" in p for p in problems)


def test_a_bad_status_is_rejected(root):
    write(root, "x", status="probably-fine",
          knobs={"target_dte": {"value": 7, "why": "s"}})
    assert any("is not one of" in p for p in V.validate(V.load("x", root), root))


def test_name_must_match_filename(root):
    toml_io.dump({"name": "other", "question": "q", "base": "frozen",
                  "status": "testing"}, root / "x.toml")
    with pytest.raises(ValueError, match="does not match the filename"):
        V.load("x", root)


def test_a_knob_table_written_as_a_bare_value_explains_the_shape(root):
    toml_io.dump({"name": "x", "question": "q", "base": "frozen",
                  "status": "testing", "knobs": {"target_dte": 7}},
                 root / "x.toml")
    with pytest.raises(ValueError, match=r"\[knobs.target_dte\]"):
        V.load("x", root)


# -- the real tree ----------------------------------------------------------

def test_every_variant_in_the_repo_is_valid():
    known = V.wheel_fields()
    bad = {}
    for name, v in V.all_variants().items():
        problems = V.validate(v, known_fields=known)
        if problems:
            bad[name] = problems
    assert not bad, f"invalid variants in variants/: {bad}"


def test_every_frozen_knob_has_a_why():
    """The migration's real claim: no reasoning was dropped when the FROZEN
    literal's comments became data."""
    frozen = V.load("frozen")
    thin = {k: len(kn.why.split()) for k, kn in frozen.knobs.items()
            if len(kn.why.split()) < 12}
    assert not thin, (
        f"knobs whose `why` is under 12 words: {thin}. Every one of these had a "
        "comment beside it in run_daily.py; a short one means prose was lost in "
        "the move.")


def test_the_master_records_when_each_knob_was_decided():
    frozen = V.load("frozen")
    missing = [k for k, kn in frozen.knobs.items() if not kn.since]
    assert not missing, f"master knobs with no `since` date: {missing}"


def test_round_trip_through_disk_preserves_every_why(tmp_path):
    frozen = V.load("frozen")
    p = frozen.save(tmp_path / "frozen.toml")
    again = V.from_dict(toml_io.load(p))
    assert {k: kn.why for k, kn in again.knobs.items()} == \
           {k: kn.why for k, kn in frozen.knobs.items()}
    assert {k: kn.value for k, kn in again.knobs.items()} == \
           {k: kn.value for k, kn in frozen.knobs.items()}
