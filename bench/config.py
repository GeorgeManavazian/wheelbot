"""Turn a variant into a WheelConfig, and hand the live bot its master config.

This is the only module `live/run_daily.py` imports from the bench, and it is
the boundary where "a paragraph of the essay" becomes "the rules the engine
runs". Two properties matter more than anything else here:

1. **It never guesses.** A missing `variants/frozen.toml`, an unreadable one, a
   knob that is not a WheelConfig field -- every one of these RAISES. A bot that
   starts on default rules because its config file was absent is worse than a
   bot that does not start, because the first one trades and nobody notices. The
   whole `live/` tree is built on that stance (see `chain_store.py` on why a
   stale snapshot returns None rather than yesterday's chains); this follows it.

2. **It adds nothing.** `load_frozen()` returns exactly the knobs the master
   file sets -- no defaults folded in, no keys invented. The result is a dict
   you splat into `WheelConfig(**...)`, byte-identical to the literal that lived
   in `run_daily.py` up to 2026-08-14. `bench doctor` checks that the master
   file still equals `variants/frozen.lock.json`.
"""
from __future__ import annotations

from typing import Any

from bench import variant as _v


class ConfigError(RuntimeError):
    """Raised instead of degrading. See the module docstring."""


def _resolve_or_die(name: str) -> _v.Resolved:
    try:
        return _v.resolve(name)
    except FileNotFoundError as e:
        raise ConfigError(
            f"variant `{name}` could not be loaded: {e}\n"
            f"The bot does not start on a config nobody chose. The live runner "
            f"needs `variants/` deployed beside the code."
        ) from e
    except Exception as e:                                  # noqa: BLE001
        raise ConfigError(f"variant `{name}` is malformed: {e}") from e


def missing_flags(resolved: _v.Resolved) -> list[str]:
    """Knobs the variant sets that this checkout's WheelConfig does not have.

    Non-empty means the variant's code is on a branch you have not checked out.
    That is a normal state for a variant under development and a fatal one for
    the master config, so the two callers treat it differently."""
    known = set(_v.wheel_fields())
    return [k for k in resolved.knobs if k not in known]


def knobs_for(name: str) -> dict[str, Any]:
    """The flattened knob dict for a variant, ready to splat into WheelConfig.

    Raises if any knob is not a field of the CURRENT WheelConfig -- the message
    names the branch to check out, because `TypeError: unexpected keyword
    argument 'assignment_exit_mode'` two frames deep in a dataclass constructor
    is not a message anyone can act on."""
    r = _resolve_or_die(name)
    missing = missing_flags(r)
    if missing:
        branch = ""
        for vn in reversed(r.lineage):
            b = _v.load(vn).branch
            if b:
                branch = f" Check out `{b}` first."
                break
        raise ConfigError(
            f"variant `{name}` sets {missing}, which the WheelConfig in this "
            f"checkout does not have.{branch}")
    return dict(r.knobs)


def load_frozen() -> dict[str, Any]:
    """THE MASTER CONFIG. What the live paper bot runs.

    `live/run_daily.py` does `FROZEN = load_frozen()` at module scope, so this
    call is on the path of every live step, every chain snapshot, every intraday
    pass and every IV accrual run. It reads one small TOML file and does no I/O
    beyond that."""
    return knobs_for(_v.MASTER_VARIANT)


def to_wheel_config(name: str, **overrides):
    """Build a real WheelConfig for a variant.

    `overrides` are the per-run facts that are not strategy -- `ticker`,
    `starting_capital` -- exactly as `run_daily` passes them today. They win
    over the variant, because a variant describes rules and a run describes
    conditions."""
    from src.engine_v2.options.wheel import WheelConfig
    knobs = knobs_for(name)
    knobs.update(overrides)
    return WheelConfig(**knobs)


def ops_for(name: str) -> dict[str, Any]:
    """The experiment conditions (window, universe tier, slots, capital) a
    variant inherits or overrides. Separate from knobs on purpose: two variants
    compared over different windows are not compared at all."""
    return dict(_resolve_or_die(name).ops)
