"""`bench doctor` -- the check that answers "is anyone going around the bench?"

The important test in this file is
`test_a_hand_edit_with_a_regenerated_lock_is_caught`. The golden test only
proves the master config equals its lock, so someone who edits the master AND
regenerates the lock slips past it. The history check is what closes that, and a
guard that has never been shown to fire is decoration.
"""
from __future__ import annotations

import pytest

from bench import doctor, lock, toml_io
from bench import variant as V


@pytest.fixture
def world(tmp_path, monkeypatch):
    """An isolated bench with a healthy master and one variant."""
    vdir = tmp_path / "variants"
    vdir.mkdir()
    toml_io.dump({"name": "plain", "question": "", "status": "promoted",
                  "ops": {"tier": "live", "n_slots": 1}}, vdir / "plain.toml")
    toml_io.dump(
        {"name": "frozen", "question": "", "base": "plain", "status": "promoted",
         "knobs": {
             "target_dte": {
                 "value": 11, "since": "2026-07-17",
                 "why": "Target days to expiry. Eleven days is the inherited "
                        "live value and has never been tested as part of a set "
                        "against the alternatives, so treat it as unproven."},
             "put_delta": {
                 "value": 0.3, "since": "2026-07-17",
                 "why": "Short-put delta. Roughly the market's odds the option "
                        "finishes in the money, so 0.30 aims at strikes with "
                        "about a thirty percent chance of assignment."}}},
        vdir / "frozen.toml")

    monkeypatch.setattr(V, "VARIANTS_DIR", vdir)
    monkeypatch.setattr(lock, "LOCK_PATH", vdir / "frozen.lock.json")
    from bench import scorecard as S
    monkeypatch.setattr(S, "SCORECARD_DIR", tmp_path / "scorecards")
    lock.save(lock.new_doc(V.resolve("frozen").knobs, "test baseline",
                           date="2026-08-14"))
    return tmp_path


def run(*checks) -> doctor.Report:
    return doctor.run(only=[c.__name__ for c in checks])


# -- the hole this closes ----------------------------------------------------

def test_a_clean_bench_passes_both_config_checks(world):
    r = run(doctor.check_lock_matches_master,
            doctor.check_history_explains_the_current_config)
    assert r.healthy, r.render()


def test_a_hand_edit_of_the_master_is_caught(world):
    """The obvious case: the golden test catches this too."""
    v = V.load("frozen")
    v.knobs["target_dte"] = V.Knob("target_dte", 7, "changed by hand")
    v.save()
    r = run(doctor.check_lock_matches_master)
    assert not r.healthy
    assert "disagree" in r.failures[0].detail


def test_a_hand_edit_with_a_regenerated_lock_is_caught(world):
    """THE ONE THAT MATTERS.

    Edit the master, then regenerate the lock so the two agree again. Every
    other guard in the repo now passes. The history is what still does not: the
    current config no longer matches the last recorded change."""
    v = V.load("frozen")
    v.knobs["target_dte"] = V.Knob("target_dte", 7, "changed by hand")
    v.save()
    doc = lock.load()
    doc["current"] = V.resolve("frozen").knobs          # regenerate, no history
    lock.save(doc)

    assert run(doctor.check_lock_matches_master).healthy, \
        "precondition: the naive check is fooled by this"

    r = run(doctor.check_history_explains_the_current_config)
    assert not r.healthy
    assert "does not match the last recorded change" in r.failures[0].detail


def test_a_real_promotion_leaves_the_history_consistent(world):
    """The legitimate path must not trip the guard it exists to enforce."""
    v = V.load("frozen")
    v.knobs["target_dte"] = V.Knob("target_dte", 7, "promoted properly")
    v.save()
    lock.append(V.resolve("frozen").knobs, date="2026-08-20", variant="cand",
                verdict="Seven beat eleven on Sharpe and drawdown.",
                evidence={"scorecard": "live/abc"},
                changed={"target_dte": (11, 7)})
    r = run(doctor.check_lock_matches_master,
            doctor.check_history_explains_the_current_config)
    assert r.healthy, r.render()


def test_a_truncated_history_is_caught(world):
    doc = lock.load()
    doc["history"] = []
    lock.save(doc)
    r = run(doctor.check_history_explains_the_current_config)
    assert not r.healthy
    assert "no history" in r.failures[0].detail


def test_a_promotion_with_no_verdict_warns(world):
    lock.append(V.resolve("frozen").knobs, date="2026-08-20", variant="cand",
                verdict="   ", evidence={}, changed={})
    r = run(doctor.check_history_explains_the_current_config)
    assert r.healthy                      # not a failure
    assert any("no verdict" in w.detail for w in r.warnings)


def test_a_missing_lock_fails_rather_than_passing_quietly(world):
    lock.LOCK_PATH.unlink()
    r = run(doctor.check_lock_matches_master)
    assert not r.healthy


# -- the reasoning discipline ------------------------------------------------

def test_a_master_knob_stripped_of_its_reasoning_is_caught(world):
    v = V.load("frozen")
    v.knobs["target_dte"] = V.Knob("target_dte", 11, "because")
    v.save()
    r = run(doctor.check_master_reasoning)
    assert not r.healthy
    assert "target_dte" in r.failures[0].detail


def test_an_undated_master_knob_warns(world):
    v = V.load("frozen")
    kn = v.knobs["target_dte"]
    v.knobs["target_dte"] = V.Knob(kn.name, kn.value, kn.why, since=None)
    v.save()
    r = run(doctor.check_master_reasoning)
    assert r.healthy
    assert any("since" in w.detail for w in r.warnings)


def test_an_invalid_variant_is_caught(world):
    toml_io.dump({"name": "bad", "question": "q", "base": "frozen",
                  "status": "testing",
                  "knobs": {"target_dte": {"value": 7, "why": ""}}},
                 V.VARIANTS_DIR / "bad.toml")
    r = run(doctor.check_variants_valid)
    assert not r.healthy
    assert "bad" in r.failures[0].detail


def test_a_variant_whose_code_is_on_a_branch_is_not_a_failure(world):
    """An unchecked-out branch is a state of the world, not a broken variant."""
    toml_io.dump({"name": "pending", "question": "does it?", "base": "frozen",
                  "status": "draft",
                  "code": {"branch": "feat/x", "new_flags": ["not_a_field_yet"]},
                  "knobs": {"not_a_field_yet": {
                      "value": True,
                      "why": "A flag this branch introduces, default off so "
                             "the plain path stays byte identical."}}},
                 V.VARIANTS_DIR / "pending.toml")
    assert run(doctor.check_variants_valid).healthy


# -- evidence ---------------------------------------------------------------

def test_a_variant_claiming_to_be_tested_without_a_scorecard_warns(world):
    toml_io.dump({"name": "cand", "question": "does it?", "base": "frozen",
                  "status": "testing",
                  "knobs": {"target_dte": {"value": 7, "why": "shorter window "
                                                              "means more turns"}}},
                 V.VARIANTS_DIR / "cand.toml")
    r = run(doctor.check_claims_have_evidence)
    assert any("no scorecard exists" in w.detail for w in r.warnings)


def test_a_draft_variant_is_not_expected_to_have_evidence(world):
    toml_io.dump({"name": "cand", "question": "does it?", "base": "frozen",
                  "status": "draft",
                  "knobs": {"target_dte": {"value": 7, "why": "shorter window "
                                                              "means more turns"}}},
                 V.VARIANTS_DIR / "cand.toml")
    r = run(doctor.check_claims_have_evidence)
    assert not r.warnings


# -- the real repo -----------------------------------------------------------

def test_this_repo_is_healthy_right_now():
    """Runs against the actual bench. Failures here are real."""
    r = doctor.run()
    assert r.healthy, r.render()


def test_a_check_that_raises_becomes_a_failure_not_a_silent_pass(monkeypatch):
    def boom(r):
        raise RuntimeError("nope")
    boom.__name__ = "check_boom"
    monkeypatch.setattr(doctor, "CHECKS", (boom,))
    r = doctor.run()
    assert not r.healthy
    assert "the check itself failed" in r.failures[0].detail
