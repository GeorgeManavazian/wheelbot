"""A scorecard run under different OPS does not describe the current variant.

THE DEFECT (found 2026-08-16). `config_hash` hashes the resolved KNOB dict only,
and `is_stale_for` compares nothing else. `[ops]` -- n_slots, capital, selector,
tier -- is the conditions of the experiment, and a variant may override it. So:

    call-otm-25/full__b86d5296919daac8.json   n_slots=1,  18 campaigns  (old)
    call-otm-25/live__b86d5296919daac8.json   n_slots=5,  99 campaigns  (new)

Same hash. The variant declared `[ops] n_slots = 5` between the two runs
because the mechanic only fires on assigned stock and cannot be resolved at
n=1 -- and the old card was still judged FRESH for it, still sat at the
filename a new full-tier run would claim, and could still have satisfied
`bench promote`'s evidence gate.

That is the anti-cherry-pick mechanism failing in the direction it exists to
prevent: promote a mechanic on a run made under conditions where it provably
cannot fire. The knobs did not change, so nothing looked wrong.

WHY THE FIX IS A STALENESS CHECK AND NOT A WIDER HASH. Folding ops into
config_hash would be the tidier-looking change and it would invalidate every
scorecard in the tree at once, including the ones whose ops never moved. Every
card already RECORDS its ops in `conditions` (n_slots, capital, selector), so
the comparison can be made directly, non-destructively, and with a reason
string that names the field that moved. A card that predates the recording is
handled by the existing no-hash rule.

`tier` is deliberately NOT compared here: it is already in the filename, and
`bench run --tier` legitimately produces one card per tier for the same config.
"""
import pytest

from bench.scorecard import Scorecard

HASH = {"config_hash": {"variant": "aaaa", "base": "cccc"}}


def _card(**conditions):
    base = {"start": "2024-08-01", "end": "2026-07-01", "n_slots": 1,
            "capital": 100000.0, "selector": "chop"}
    base.update(conditions)
    return Scorecard(variant="v", base="frozen", tier="full",
                     is_evidence=True, created="2026-08-15T00:00:00+00:00",
                     provenance=dict(HASH), conditions=base,
                     arms={"variant": {}, "base": {}})


def test_matching_ops_is_still_fresh():
    """The change must not make every existing scorecard stale."""
    assert _card().is_stale_for("aaaa", "cccc",
                                ops={"n_slots": 1, "capital": 100000.0,
                                     "selector": "chop"}) is None


def test_omitting_ops_entirely_keeps_the_old_behaviour():
    """Callers that do not pass ops get exactly the old comparison, so this is
    additive and nothing that does not opt in can break."""
    assert _card().is_stale_for("aaaa", "cccc") is None


def test_a_run_at_a_different_slot_width_is_stale():
    """THE SPEC. The call-otm-25 case, reduced."""
    why = _card(n_slots=1).is_stale_for(
        "aaaa", "cccc", ops={"n_slots": 5, "capital": 100000.0,
                             "selector": "chop"})
    assert why is not None, \
        "an n_slots=1 scorecard was judged fresh for an n_slots=5 variant"
    assert "n_slots" in why
    assert "1" in why and "5" in why, \
        "the reason must name both values -- 'ops changed' is not actionable"


@pytest.mark.parametrize("field,old,new", [
    ("n_slots", 1, 5),
    ("capital", 100000.0, 25000.0),
    ("selector", "chop", "plain"),
])
def test_every_experiment_defining_ops_field_is_compared(field, old, new):
    why = _card(**{field: old}).is_stale_for(
        "aaaa", "cccc", ops={"n_slots": 1, "capital": 100000.0,
                             "selector": "chop", field: new})
    assert why is not None and field in why


def test_a_card_with_no_conditions_block_is_left_to_the_hash_rules():
    """Cards predating the conditions block are NOT blanket-failed here.

    Every card `bench run` writes records its conditions, and the only cards
    that lack them are salvaged imports, which the no-hash rule already treats
    as stale. Failing them here as well would buy no real safety and would
    retire the entire existing scorecard tree in a single commit -- the kind of
    collateral that gets a correctness fix reverted rather than kept."""
    c = _card()
    c.conditions = {}
    assert c.is_stale_for("aaaa", "cccc", ops={"n_slots": 5}) is None


def test_a_partial_conditions_block_is_stale():
    """Different animal: a writer that knew about conditions and recorded some
    of them. Absence of the compared field there is not agreement."""
    c = _card()
    c.conditions = {"start": "2024-08-01", "capital": 100000.0}   # no n_slots
    why = c.is_stale_for("aaaa", "cccc", ops={"n_slots": 5})
    assert why is not None and "n_slots" in why


def test_the_knob_and_base_checks_still_fire_first():
    """Ordering: a changed knob is the more fundamental staleness and should
    still be the reason given, so existing messages do not regress."""
    why = _card(n_slots=1).is_stale_for(
        "zzzz", "cccc", ops={"n_slots": 5})
    assert "the variant changed since this run" in why
