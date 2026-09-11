"""What the bench refuses to do, and how clearly it says so.

Most of these assert on MESSAGES. That is deliberate: a bench that fails
obscurely is a bench nobody uses, and a bench nobody uses is how thirty
untracked harnesses ended up in `scratchpad/`. Every refusal here has to name
the fix.
"""
from __future__ import annotations

import pytest

from bench import config as bench_config
from bench import registry, replay, run, tiers, toml_io
from bench import variant as V


# -- tiers -------------------------------------------------------------------

def test_the_default_tier_is_the_faithful_window_not_the_longest():
    """`full` starts 2022-06, where the OI store covers 86 of 534 tickers, so a
    liquidity-gated config refuses 443 names on data availability. Defaulting
    there produces a confident wrong answer."""
    assert tiers.DEFAULT_TIER == "live"
    assert tiers.get("live").start == tiers.OI_COMPLETE_FROM


def test_every_evidence_tier_starts_inside_full_oi_coverage_or_says_why():
    live = tiers.get("live")
    assert live.is_evidence and live.start >= tiers.OI_COMPLETE_FROM
    full = tiers.get("full")
    assert full.is_evidence and "OI-blind" in full.blurb


def test_smoke_and_mid_are_never_evidence():
    assert not tiers.get("smoke").is_evidence
    assert not tiers.get("mid").is_evidence


def test_smoke_sits_inside_oi_coverage():
    """The first smoke window was 2024 H1, entirely before the OI store starts.
    Every entry was refused and both arms scored a flat 0.00% -- which reads as
    "the change does nothing" rather than "the tier is broken"."""
    assert tiers.get("smoke").start >= tiers.OI_COMPLETE_FROM


def test_an_unknown_tier_explains_why_windows_are_not_free_form():
    with pytest.raises(ValueError, match="not comparable"):
        tiers.get("last-6-months")


def test_max_dte_covers_the_selectable_band():
    from src.engine_v2.options.select import derived_band
    for target in (7, 11, 21):
        assert tiers.max_dte_for(target) >= derived_band(target)[1]


# -- config boundary ---------------------------------------------------------

def test_a_missing_master_config_raises_rather_than_defaulting(monkeypatch,
                                                               tmp_path):
    """A bot that starts on rules nobody chose is worse than one that does not
    start. This is the single most important refusal in the subsystem."""
    monkeypatch.setattr(V, "VARIANTS_DIR", tmp_path)
    with pytest.raises(bench_config.ConfigError, match="does not start"):
        bench_config.load_frozen()


def test_a_knob_whose_code_is_on_a_branch_names_the_branch(monkeypatch,
                                                           tmp_path):
    d = tmp_path / "variants"
    d.mkdir()
    toml_io.dump({"name": "plain", "question": "", "status": "promoted"},
                 d / "plain.toml")
    toml_io.dump({"name": "frozen", "question": "", "base": "plain",
                  "status": "promoted"}, d / "frozen.toml")
    toml_io.dump({"name": "x", "question": "q", "base": "frozen",
                  "status": "draft",
                  "code": {"branch": "feat/assignment-exit",
                           "new_flags": ["not_a_field_yet"]},
                  "knobs": {"not_a_field_yet": {"value": True, "why": "w"}}},
                 d / "x.toml")
    monkeypatch.setattr(V, "VARIANTS_DIR", d)
    with pytest.raises(bench_config.ConfigError,
                       match="Check out `feat/assignment-exit`"):
        bench_config.knobs_for("x")


def test_ops_are_separate_from_knobs():
    """Window and slot count are the conditions of the experiment, not the
    strategy. Mixing them is how two variants stop being comparable."""
    ops = bench_config.ops_for("frozen")
    knobs = bench_config.load_frozen()
    assert "tier" in ops and "n_slots" in ops
    assert not (set(ops) & set(knobs))


# -- run guards --------------------------------------------------------------

class FakeArm:
    def __init__(self, **kw):
        from src.engine_v2.options.wheel import WheelConfig
        self.name, self.variant = "variant", "x"
        self.cfg = WheelConfig(**kw)
        self.knobs = kw


def test_a_solo_mechanic_is_refused_by_name():
    """`run_portfolio_wheel` runs the plain+basis wheel only. The live bot IS
    the portfolio path, so a knob it cannot run can never reach the master."""
    with pytest.raises(run.RunRefused, match="roll_tested_puts"):
        run._refuse_solo_mechanics(FakeArm(roll_tested_puts=True))
    with pytest.raises(run.RunRefused, match="put_stop_mult"):
        run._refuse_solo_mechanics(FakeArm(put_stop_mult=2.0))
    with pytest.raises(run.RunRefused, match="regime gate"):
        run._refuse_solo_mechanics(FakeArm(regime_entry_gate=True))


def test_the_solo_refusal_points_at_the_solo_engine():
    with pytest.raises(run.RunRefused, match="run_wheel"):
        run._refuse_solo_mechanics(FakeArm(liquidate_assignment=True))


def test_a_portfolio_safe_variant_passes_the_guard():
    run._refuse_solo_mechanics(FakeArm(target_dte=7, call_min_strike="basis"))


def test_missing_open_interest_is_declared_not_absorbed():
    import pandas as pd
    from src.engine_v2.options.wheel import WheelConfig
    arms = [FakeArm(liq_min_open_interest=250.0)]
    chains = {"SPY": pd.DataFrame({"open_interest": [1.0, None, None, None]}),
              "QQQ": pd.DataFrame({"open_interest": [None, None, None, None]})}
    out = run._divergences(arms, chains, None, ["SPY", "QQQ"])
    assert out and "open_interest is present on 12.5% of chain rows" in out[0]
    assert "data availability, not on liquidity" in out[0]


def test_full_open_interest_coverage_declares_nothing():
    import pandas as pd
    arms = [FakeArm(liq_min_open_interest=250.0)]
    chains = {"SPY": pd.DataFrame({"open_interest": [1.0, 2.0, 3.0]})}
    assert run._divergences(arms, chains, None, ["SPY"]) == []


def test_a_config_that_does_not_gate_on_oi_declares_nothing_about_it():
    import pandas as pd
    arms = [FakeArm(target_dte=11)]
    chains = {"SPY": pd.DataFrame({"open_interest": [None, None]})}
    assert not any("open_interest" in d
                   for d in run._divergences(arms, chains, None, ["SPY"]))


def test_the_iv_ranker_always_declares_its_solver_provenance():
    import pandas as pd
    arms = [FakeArm(rank_by="iv_rank")]
    out = run._divergences(arms, {"SPY": pd.DataFrame({"x": [1]})}, None, ["SPY"])
    assert any("SOLVED IV history" in d for d in out)
    assert any("vendor_iv" in d for d in out)


def test_every_rank_by_that_reads_vrp_loads_the_iv_history():
    """Regression: `vrp_viable`'s smoke run raised "needs an IV history" at
    the engine guard because `_run_arm`'s loader gate was a separate inline
    tuple from the engine's validation list (portfolio.py) and nobody had
    added the new value to both. `iv_rank` and `vrp` must stay in this set
    too, or the same bug reopens for them."""
    assert set(run.RANK_BY_NEEDS_IV_HISTORY) >= {"iv_rank", "vrp", "vrp_viable"}


# -- replay guards -----------------------------------------------------------

def test_replay_without_snapshots_names_the_env_var(monkeypatch, tmp_path):
    monkeypatch.setattr(replay, "snapshot_dir", lambda: tmp_path / "nope")
    with pytest.raises(replay.ReplayRefused) as e:
        replay.available_days()
    assert "WHEELBOT_STATE_DIR" in str(e.value)
    assert "no chain snapshots" in str(e.value)


def test_replay_lists_only_days_in_the_window(monkeypatch, tmp_path):
    d = tmp_path / "chains"
    d.mkdir()
    for name in ("2026-08-01.json", "2026-08-05.json", "2026-08-12.json",
                 "notadate.json"):
        (d / name).write_text("{}")
    monkeypatch.setattr(replay, "snapshot_dir", lambda: d)
    got = [str(x.date()) for x in replay.available_days("2026-08-02",
                                                        "2026-08-11")]
    assert got == ["2026-08-05"]


def test_replay_refusals_only_count_entry_gates():
    warns = [(1, "entry_gated_illiquid", "SPY"),
             (1, "entry_gated_illiquid", "SPY"),
             (1, "held_mark_missing", "QQQ")]
    assert replay._refusals(warns) == {"entry_gated_illiquid": ["SPY"]}


def test_replay_dte_coverage_reports_the_band_it_would_select_from():
    import pandas as pd
    from src.engine_v2.options.wheel import WheelConfig
    chains = {"SPY": pd.DataFrame({"dte": [9, 10, 11]}),
              "QQQ": pd.DataFrame({"dte": [1, 2]})}
    cov = replay._dte_coverage(chains, WheelConfig(target_dte=11))
    assert cov["tickers_with_band"] == 1 and cov["tickers_in_snapshot"] == 2


# -- registry ----------------------------------------------------------------

def test_the_registry_lists_every_variant_and_marks_the_master():
    md = registry.registry_markdown()
    for name in V.all_variants():
        assert f"`{name}`" in md
    assert "master document" in md
    assert "control arm" in md


def test_the_registry_documents_the_workflow_in_order():
    md = registry.registry_markdown()
    for cmd in ("bench new", "bench run", "bench replay", "bench promote"):
        assert cmd in md
    assert md.index("bench new") < md.index("bench run") < md.index("bench promote")
