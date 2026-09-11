"""The promotion gate and the write it guards.

These tests run against a throwaway `variants/` tree. Nothing here can touch the
real master config -- which matters more here than anywhere else in the repo,
because the real one is what a live bot loads at 17:00 ET.

The test gate is always skipped in this file (`skip_tests=True`): letting it run
would spawn pytest from inside pytest.
"""
from __future__ import annotations

import json

import pytest

from bench import lock, policy, promote, registry, scorecard as S
from bench import toml_io
from bench import variant as V


@pytest.fixture
def world(tmp_path, monkeypatch):
    """An isolated bench: variants/, scorecards/, replays/ and the lock."""
    vdir = tmp_path / "variants"
    vdir.mkdir()
    toml_io.dump({"name": "plain", "question": "", "status": "promoted",
                  "ops": {"tier": "live", "n_slots": 1, "capital": 100000.0}},
                 vdir / "plain.toml")
    toml_io.dump({"name": "frozen", "question": "", "base": "plain",
                  "status": "promoted",
                  "knobs": {"target_dte": {"value": 11, "why": "the live value"},
                            "put_delta": {"value": 0.3, "why": "the live value"}}},
                 vdir / "frozen.toml")
    toml_io.dump({"name": "cand", "question": "does 7 beat 11?",
                  "base": "frozen", "status": "testing",
                  "knobs": {"target_dte": {"value": 7, "why": "half the theta "
                                                              "window; more turns"}}},
                 vdir / "cand.toml")

    monkeypatch.setattr(V, "VARIANTS_DIR", vdir)
    monkeypatch.setattr(S, "SCORECARD_DIR", tmp_path / "scorecards")
    monkeypatch.setattr(lock, "LOCK_PATH", vdir / "frozen.lock.json")
    monkeypatch.setattr(registry, "REGISTRY_PATH", vdir / "README.md")
    monkeypatch.setattr(registry, "CHANGELOG_PATH", vdir / "CHANGELOG.md")
    from bench import replay as replay_mod
    monkeypatch.setattr(replay_mod, "REPLAY_DIR", tmp_path / "replays")

    lock.save(lock.new_doc(V.resolve("frozen").knobs, "test baseline",
                           date="2026-08-14"))
    return tmp_path


def good_card(variant="cand", tier="live", dirty=False, vhash=None,
              bhash=None):
    from bench import config as bench_config, fingerprint
    c = S.new(variant, "frozen", tier, tier in ("live", "full"))
    import pandas as pd
    idx = pd.bdate_range("2025-01-01", periods=3)
    c.arms["variant"] = S.metrics_from_equity(
        pd.Series([100.0, 110.0, 125.0], index=idx), 100.0)
    c.arms["base"] = S.metrics_from_equity(
        pd.Series([100.0, 105.0, 110.0], index=idx), 100.0)
    c.provenance = {
        "engine": {"sha": "abc123abc123", "dirty": dirty},
        "config_hash": {
            "variant": vhash or fingerprint.config_hash(
                bench_config.knobs_for(variant)),
            "base": bhash or fingerprint.config_hash(
                bench_config.knobs_for("frozen"))},
        "data": {"hash": "d00dfeed"}}
    return c


VERDICT = "Seven beats eleven on Sharpe and drawdown across the live window."


# -- the diff ---------------------------------------------------------------

def test_diff_reports_only_what_would_actually_change(world):
    assert promote.diff_for("cand") == {"target_dte": (11, 7)}


def test_a_variant_identical_to_master_promotes_nothing(world):
    toml_io.dump({"name": "same", "question": "q", "base": "frozen",
                  "status": "testing",
                  "knobs": {"target_dte": {"value": 11, "why": "unchanged"}}},
                 V.VARIANTS_DIR / "same.toml")
    res = promote.promote("same", VERDICT, skip_tests=True)
    assert res.changed == {}
    assert not res.applied
    assert "changes nothing" in res.render()


# -- what the gate refuses --------------------------------------------------

def test_promotion_is_blocked_without_a_scorecard(world):
    res = promote.promote("cand", VERDICT, skip_tests=True)
    assert not res.applied
    assert any(g.name == "scorecard" and not g.passed for g in res.report.gates)
    assert "no scorecard on disk" in res.report.render()


def test_promotion_is_blocked_without_a_verdict(world):
    good_card().save()
    res = promote.promote("cand", "", skip_tests=True)
    assert not res.applied
    assert any(g.name == "verdict" and not g.passed for g in res.report.gates)


def test_a_one_word_verdict_is_refused(world):
    good_card().save()
    res = promote.promote("cand", "good", skip_tests=True)
    assert not res.applied
    assert "words" in res.report.render()


def test_a_smoke_scorecard_cannot_satisfy_the_gate(world):
    good_card(tier="smoke").save()
    res = promote.promote("cand", VERDICT, skip_tests=True)
    assert not res.applied
    assert "not in accept_tiers" in res.report.render()


def test_a_dirty_tree_scorecard_is_refused(world):
    good_card(dirty=True).save()
    res = promote.promote("cand", VERDICT, skip_tests=True)
    assert not res.applied
    assert "dirty tree" in res.report.render()


def test_a_stale_scorecard_is_refused(world):
    """The anti-cherry-pick rule: tune the variant after a good run and the
    run stops counting."""
    good_card().save()
    v = V.load("cand")
    v.knobs["target_dte"] = V.Knob("target_dte", 5, "changed my mind")
    v.save()
    res = promote.promote("cand", VERDICT, skip_tests=True)
    assert not res.applied
    assert "stale" in res.report.render()


def test_a_scorecard_against_a_since_changed_base_is_refused(world):
    good_card(bhash="an-old-master").save()
    res = promote.promote("cand", VERDICT, skip_tests=True)
    assert not res.applied
    assert "BASE changed" in res.report.render()


def test_the_report_lists_every_failure_not_just_the_first(world):
    res = promote.promote("cand", "", skip_tests=True)
    failed = [g.name for g in res.report.gates if not (g.passed or g.skipped)]
    assert set(failed) == {"scorecard", "verdict"}


def test_skipping_the_test_gate_is_recorded_as_skipped_not_passed(world):
    g = policy.check("cand", verdict=VERDICT, skip_tests=True)
    tests = [x for x in g.gates if x.name == "tests"][0]
    assert tests.skipped and not tests.passed


# -- conditional gates ------------------------------------------------------

def test_a_liquidity_variant_additionally_requires_a_replay(world):
    toml_io.dump({"name": "liq", "question": "does the OI floor bind?",
                  "base": "frozen", "status": "testing",
                  "touches": ["liquidity"],
                  "knobs": {"liq_min_volume": {"value": 50.0,
                                               "why": "twice the live floor"}}},
                 V.VARIANTS_DIR / "liq.toml")
    good_card("liq").save()
    res = promote.promote("liq", VERDICT, skip_tests=True)
    assert not res.applied
    assert any(g.name == "replay" for g in res.report.gates)
    assert "no replay on disk" in res.report.render()


def test_a_non_liquidity_variant_needs_no_replay(world):
    good_card().save()
    names = [g.name for g in policy.check("cand", VERDICT, True).gates]
    assert "replay" not in names


# -- thresholds (inert until the owner fills them in) -----------------------

def test_thresholds_are_inert_by_default(world):
    g = policy._gate_thresholds(V.load("cand"), {"thresholds": {}}, {})
    assert g.passed and "empty by design" in g.detail


def test_a_threshold_once_set_is_enforced(world, tmp_path):
    """The machinery the owner's pinned bar will drop into. Proven now so that
    filling in `[thresholds]` later is a config change, not a code change."""
    pol_path = tmp_path / "policy.toml"
    toml_io.dump({"evidence": {"required": ["scorecard", "verdict"],
                               "accept_tiers": ["live"],
                               "allow_dirty_tree": False,
                               "require_fresh": True},
                  "thresholds": {"sharpe": 99.0}}, pol_path)
    good_card().save()
    rep = policy.check("cand", VERDICT, True, policy_path=pol_path)
    th = [g for g in rep.gates if g.name == "thresholds"][0]
    assert not th.passed and "sharpe" in th.detail
    assert not rep.passed


# -- the write ---------------------------------------------------------------

def test_dry_run_writes_nothing(world):
    good_card().save()
    before = (V.VARIANTS_DIR / "frozen.toml").read_text()
    res = promote.promote("cand", VERDICT, dry_run=True, skip_tests=True)
    assert not res.applied
    assert (V.VARIANTS_DIR / "frozen.toml").read_text() == before
    assert "WOULD CHANGE" in res.render()


def test_a_clean_promotion_writes_the_knob_and_its_reasoning(world):
    good_card().save()
    res = promote.promote("cand", VERDICT, skip_tests=True,
                          today="2026-08-20")
    assert res.applied, res.render()
    from bench import config as bench_config
    assert bench_config.load_frozen()["target_dte"] == 7

    knob = V.load("frozen").knobs["target_dte"]
    assert "half the theta window" in knob.why      # the variant's own reason
    assert "promoted 2026-08-20" in knob.why        # the provenance footer
    assert VERDICT in knob.why
    assert knob.since == "2026-08-20"
    assert knob.source == "variant:cand"


def test_a_promotion_updates_the_lock_and_the_changelog(world):
    good_card().save()
    promote.promote("cand", VERDICT, skip_tests=True, today="2026-08-20")
    doc = lock.load()
    assert doc["current"]["target_dte"] == 7
    last = doc["history"][-1]
    assert last["variant"] == "cand"
    assert last["changed"]["target_dte"] == [11, 7]
    assert last["verdict"] == VERDICT
    md = lock.changelog_markdown()
    assert "2026-08-20" in md and "`cand`" in md and "`target_dte`" in md


def test_a_promotion_marks_the_variant_promoted(world):
    good_card().save()
    promote.promote("cand", VERDICT, skip_tests=True, today="2026-08-20")
    v = V.load("cand")
    assert v.status == "promoted"
    assert v.updated == "2026-08-20"
    assert v.evidence and v.evidence[-1]["kind"] == "promotion"


def test_the_lock_history_keeps_the_migration_entry_forever(world):
    good_card().save()
    promote.promote("cand", VERDICT, skip_tests=True, today="2026-08-20")
    hist = lock.load()["history"]
    assert hist[0]["by"] == "migration"
    assert hist[0]["knobs"]["target_dte"] == 11


def test_force_promotes_despite_failing_gates_and_records_that_it_did(world):
    res = promote.promote("cand", VERDICT, skip_tests=True, force=True,
                          today="2026-08-20")
    assert res.applied
    assert "--force" in V.load("frozen").knobs["target_dte"].why


def test_the_registry_is_regenerated_by_a_promotion(world):
    good_card().save()
    promote.promote("cand", VERDICT, skip_tests=True, today="2026-08-20")
    md = (V.VARIANTS_DIR / "README.md").read_text()
    assert "`cand`" in md and "promoted" in md


def test_re_promoting_replaces_the_reasoning_and_the_lock_keeps_the_history(world):
    """The division of labour between the two files.

    `frozen.toml` carries the reasoning that is TRUE NOW -- the variant's
    current `why` plus exactly one provenance footer. It does not accrete every
    past footer, because a knob's justification that grows without bound stops
    being read, and the stale half is the half that misleads.

    `frozen.lock.json` carries the HISTORY -- one entry per promotion, forever,
    including the original migration. That is where "what did this knob used to
    be, and why did we change it" is answered."""
    good_card().save()
    promote.promote("cand", VERDICT, skip_tests=True, today="2026-08-20")

    v = V.load("cand")
    v.knobs["target_dte"] = V.Knob("target_dte", 5, "narrower still")
    v.save()
    good_card().save()
    promote.promote("cand", "Five beats seven on the same window.",
                    skip_tests=True, today="2026-08-25")

    why = V.load("frozen").knobs["target_dte"].why
    assert why.count("--- promoted ") == 1, "the master must not accrete footers"
    assert why.startswith("narrower still")          # the CURRENT reasoning
    assert VERDICT not in why                        # the superseded one is gone

    hist = lock.load()["history"]
    assert len(hist) == 3                            # migration + two promotions
    assert [h.get("by") for h in hist] == ["migration", "promote", "promote"]
    assert hist[1]["changed"]["target_dte"] == [11, 7]
    assert hist[2]["changed"]["target_dte"] == [7, 5]
    assert hist[1]["verdict"] == VERDICT              # nothing is lost, it moved
