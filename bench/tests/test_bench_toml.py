"""The TOML emitter. Small, but it writes the live bot's config file, so every
value kind it can emit must round-trip exactly."""
from __future__ import annotations

import pytest

from bench import toml_io


def rt(d: dict) -> dict:
    return toml_io.loads(toml_io.dumps(d))


def test_scalars_round_trip():
    d = {"s": "hello", "b": True, "f": 0.05, "i": 250, "neg": -1.5}
    assert rt(d) == d


def test_bool_is_never_written_as_an_int():
    """`bool` is a subclass of `int` in Python. Emitting True as 1 would make
    `earnings_blackout` load as an int, and `if cfg.earnings_blackout` would
    still be truthy -- a bug that hides until something compares identity."""
    assert "true" in toml_io.dumps({"x": True})
    assert rt({"x": True})["x"] is True


def test_multiline_prose_round_trips_exactly():
    """The `why` fields are the point of the format. A knob's 25-line
    measurement note must survive byte for byte."""
    why = ("A2b (2026-08-11): it refuses on CHEAPNESS, not illiquidity.\n"
           "\n"
           "Option prices move on a 1-cent grid, so at a $0.10 mid the\n"
           'tightest market that can exist is already 10% of mid.\n')
    assert rt({"why": why})["why"] == why


def test_quotes_inside_prose_survive():
    why = 'The owner said "a no-brainer to have it on".\nSecond line.'
    assert rt({"why": why})["why"] == why


def test_prose_ending_in_a_quote_does_not_fuse_with_the_delimiter():
    why = 'he called it "provisional"\nand then said "ship it"'
    assert rt({"why": why})["why"] == why


def test_backslashes_are_not_reinterpreted_as_escapes():
    why = "regex-ish: \\n is not a newline here\nbut this is"
    assert rt({"why": why})["why"] == why


def test_nested_tables_and_arrays_of_tables():
    d = {"name": "x",
         "code": {"branch": "feat/y", "new_flags": ["a", "b"]},
         "evidence": [{"kind": "tests", "ok": True},
                      {"kind": "scorecard", "ok": False}]}
    assert rt(d) == d


def test_none_raises_rather_than_vanishing():
    """TOML has no null. Silently dropping the key would turn "explicitly off,
    and here is the 25-line reason" into "never mentioned" -- two states the
    schema exists to keep apart."""
    with pytest.raises(ValueError, match="unset"):
        toml_io.dumps({"x": None})


def test_a_table_of_only_subtables_gets_no_redundant_header():
    """`[knobs.target_dte]` implicitly creates `knobs`. A bare `[knobs]` above
    it is noise in a file meant to be read by a human."""
    text = toml_io.dumps({"knobs": {"target_dte": {"value": 7, "why": "x"}}})
    assert "[knobs]" not in text
    assert "[knobs.target_dte]" in text
    assert toml_io.loads(text)["knobs"]["target_dte"]["value"] == 7


def test_header_comment_is_emitted():
    text = toml_io.dumps({"a": 1}, header="generated\ndo not edit")
    assert text.startswith("# generated\n# do not edit\n")
    assert toml_io.loads(text) == {"a": 1}


def test_unsupported_type_raises_instead_of_guessing():
    with pytest.raises(TypeError):
        toml_io.dumps({"x": {1, 2}})


def test_dump_is_atomic_and_leaves_no_tmp_file(tmp_path):
    p = tmp_path / "sub" / "v.toml"
    toml_io.dump({"a": 1}, p)
    assert toml_io.load(p) == {"a": 1}
    assert list(tmp_path.rglob("*.tmp")) == []
