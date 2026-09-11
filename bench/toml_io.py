"""TOML read/write with ZERO third-party dependencies.

WHY TOML AND NOT YAML. `variants/frozen.toml` is the live bot's config, and
the live runner loads it at import time. PyYAML is not in `requirements.txt`,
so choosing YAML would mean a bot that refuses to start until someone remembers
to pip-install on the deployment host -- the "works here, dead there" failure
class. `tomllib` is stdlib from Python 3.11, so the read path costs nothing. Only the WRITE path is missing from stdlib, and that is this file.

WHY NOT JSON. The whole point of a variant file is the `why` beside each knob --
the 25-line explanation of the A2b liquidity finding is the most valuable text
in `run_daily.py`. JSON turns that into one `\\n`-escaped line, which nobody
edits by hand, which means nobody writes it. TOML's triple-quoted strings keep
prose readable as prose.

The emitter is deliberately small: it handles exactly the value kinds the
variant schema uses (str, bool, int, float, list-of-scalar, dict, list-of-dict)
and raises on anything else rather than guessing. A general TOML writer would be
more code and more ways to be subtly wrong.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

# `None` has no TOML representation. The variant schema never emits it -- a knob
# meant to be None is written as `unset = true` (see bench/variant.py), which is
# both representable and self-documenting. Any None reaching the emitter is a
# bug in the caller, so it raises rather than dropping the key silently.
_NO_NULL = ("TOML has no null. A knob whose value is None must be written as "
            "`unset = true` -- see bench/variant.py:Knob.")


def load(path: str | Path) -> dict:
    """Parse a TOML file into a plain dict. Raises FileNotFoundError if absent
    and tomllib.TOMLDecodeError if malformed -- never returns a partial or
    empty dict on error, because a half-read config is how a bot trades rules
    nobody chose."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def loads(text: str) -> dict:
    return tomllib.loads(text)


def _esc_basic(s: str) -> str:
    """Escape for a single-line basic string."""
    out = s.replace("\\", "\\\\").replace('"', '\\"')
    return out.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def _fmt_scalar(v) -> str:
    if isinstance(v, bool):          # before int -- bool IS an int in Python
        return "true" if v else "false"
    if v is None:
        raise ValueError(_NO_NULL)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        # repr round-trips exactly; inf/nan have no TOML form.
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError(f"TOML cannot represent {v!r}")
        return repr(v)
    if isinstance(v, str):
        if "\n" in v:
            return _fmt_multiline(v)
        return f'"{_esc_basic(v)}"'
    raise TypeError(f"no TOML form for {type(v).__name__}: {v!r}")


def _fmt_multiline(s: str) -> str:
    """A triple-quoted basic string. Prose only -- backslashes are escaped so a
    stray one in a comment cannot turn into an escape sequence on the way back,
    and a trailing quote is escaped so it cannot fuse with the closing
    delimiter."""
    body = s.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    if body.endswith('"'):
        body = body[:-1] + '\\"'
    # Leading newline after the opening delimiter is stripped by every TOML
    # parser, so add one for readability without changing the value.
    return '"""\n' + body + '"""'


def _fmt_array(v: list) -> str:
    return "[" + ", ".join(_fmt_scalar(x) for x in v) + "]"


def _is_scalar(v) -> bool:
    return isinstance(v, (str, bool, int, float))


def _classify(v) -> str:
    """One name per value kind the emitter handles. Anything else RAISES.

    Written as an explicit classification rather than a chain of elif branches
    because the first version of this function let unhandled values fall
    through both passes and vanish from the output -- in the file that holds the
    live bot's config. A key that silently disappears is strictly worse than a
    crash: `bench frozen` would print 15 knobs where the bot runs 16, and every
    reader would believe it."""
    if _is_scalar(v):
        return "scalar"
    if isinstance(v, dict):
        return "table"
    if isinstance(v, list):
        if not v:
            return "empty_array"
        if all(_is_scalar(x) for x in v):
            return "array"
        if all(isinstance(x, dict) for x in v):
            return "table_array"
        raise TypeError("mixed-type array is not supported")
    if v is None:
        raise ValueError(_NO_NULL)
    raise TypeError(f"no TOML form for {type(v).__name__}: {v!r}")


def _emit_table(d: dict, prefix: str, out: list) -> None:
    """Emit one table's scalars/arrays, then recurse into its sub-tables.

    TOML requires every bare key/value pair to precede the first `[sub.table]`
    header inside the same table, so the two passes are not a style choice."""
    kinds = {}
    for k, v in d.items():
        try:
            kinds[k] = _classify(v)
        except (TypeError, ValueError) as e:
            name = f"{prefix}.{k}" if prefix else k
            raise type(e)(f"{name}: {e}") from None

    for k, v in d.items():
        if kinds[k] == "scalar":
            out.append(f"{k} = {_fmt_scalar(v)}")
        elif kinds[k] == "array":
            out.append(f"{k} = {_fmt_array(v)}")
        elif kinds[k] == "empty_array":
            out.append(f"{k} = []")

    for k, v in d.items():
        name = f"{prefix}.{k}" if prefix else k
        if kinds[k] == "table":
            # A table whose keys are ALL sub-tables needs no header of its own:
            # `[knobs.target_dte]` implicitly creates `knobs`, and emitting a
            # bare `[knobs]` above it is noise in a file meant to be read.
            only_subtables = v and all(isinstance(x, dict) for x in v.values())
            if not only_subtables:
                out.append("")
                out.append(f"[{name}]")
            _emit_table(v, name, out)
        elif kinds[k] == "table_array":
            for item in v:
                out.append("")
                out.append(f"[[{name}]]")
                _emit_table(item, name, out)


def dumps(d: dict, header: str = "") -> str:
    """Serialize a dict to TOML text. `header` is emitted as leading `#` comment
    lines -- the one place a comment survives, used to stamp generated files
    with "do not hand-edit, run this instead"."""
    out: list[str] = []
    if header:
        for line in header.rstrip("\n").split("\n"):
            out.append(f"# {line}" if line else "#")
        out.append("")
    _emit_table(d, "", out)
    text = "\n".join(out).strip("\n") + "\n"
    return text


def dump(d: dict, path: str | Path, header: str = "") -> None:
    """Write atomically (tmp + rename). A variant file half-written by a
    crashed promote is a config nobody chose -- same reasoning as
    live/chain_store.save_chain_snapshot."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(dumps(d, header), encoding="utf-8")
    tmp.replace(path)
