"""Stamps that make a scorecard mean something a month later.

A number in `results/row01.json` answers "what did this config score?" only if
you also know WHICH config, WHICH engine, and WHICH data produced it. None of
the ten grid arms recorded any of the three, so re-running one and getting a
different number is indistinguishable from a bug -- and every one of those files
sits under the gitignored `results/`, so there is nothing to re-read anyway.

Three stamps, each answering one of those questions:

  config_hash  -- the flattened knob dict. Changes when the variant changes.
  engine_sha   -- git HEAD, plus whether the tree was dirty. A scorecard from a
                  dirty tree is provisional by construction and says so.
  data_fp      -- the parquet files in scope, by path and size. Catches a
                  re-pull that changed history under a stable filename, which is
                  the failure this project has already hit (the NVDA and SPY
                  holes, and the 2026-07-25 subscription lapse that froze the
                  corpus mid-window).

Everything here is cheap: no parquet is opened, only stat()'d.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_hash(knobs: dict[str, Any]) -> str:
    """Stable hash of a resolved knob dict. Sorted keys and JSON with no
    whitespace, so it depends on values only -- reordering a variant file must
    not invalidate its evidence, but changing a threshold must."""
    return _sha(json.dumps(knobs, sort_keys=True, separators=(",", ":"),
                           default=str))[:16]


def git_provenance(root: Path | None = None) -> dict:
    """HEAD sha, branch, and whether the tree is dirty.

    `dirty` is not cosmetic. A scorecard produced from uncommitted edits cannot
    be re-run by anyone, including you tomorrow, so `bench promote` treats a
    dirty scorecard as unusable evidence unless the policy says otherwise."""
    root = root or REPO_ROOT

    def _git(*args: str) -> str:
        try:
            return subprocess.run(("git", "-C", str(root)) + args,
                                  capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:                                   # noqa: BLE001
            return ""

    status = _git("status", "--porcelain")
    # Untracked scratch files are not a reproducibility risk; modified tracked
    # ones are. Counting only the latter keeps `dirty` meaningful in a repo
    # whose scratchpad/ is permanently untracked.
    tracked_dirty = [ln for ln in status.split("\n")
                     if ln.strip() and not ln.startswith("??")]
    return {"sha": _git("rev-parse", "HEAD")[:12],
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(tracked_dirty),
            "dirty_files": [ln[3:] for ln in tracked_dirty][:20]}


def _month_keys(start, end) -> set[str]:
    import pandas as pd
    s, e = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    return {d.strftime("%Y-%m") for d in
            pd.date_range(s, e, freq="MS").union([s, e])}


def data_fingerprint(tickers: Iterable[str], start, end,
                     stores: Iterable[Path] | None = None) -> dict:
    """Fingerprint every parquet the run could read, without reading any.

    Returns the hash plus the counts behind it, because a bare hash tells you
    two runs differ and nothing about how. `n_files` dropping by 300 is a
    different problem from `bytes` shifting by 4KB."""
    from bench.loaders import chains as _chains
    stores = list(stores) if stores else [_chains.STORE, _chains.OI_STORE]
    want = _month_keys(start, end)
    entries: list[str] = []
    n_files = 0
    total = 0
    tickers = list(tickers)
    for store in stores:
        # The store paths are repo-relative by default and absolute when
        # overridden, so resolve before comparing -- `relative_to` on a
        # relative path against an absolute root raises, and a fingerprint that
        # can throw is a fingerprint that gets caught and skipped.
        root = Path(store)
        if not root.is_absolute():
            root = REPO_ROOT / root
        if not root.is_dir():
            continue
        for tk in sorted(tickers):
            d = root / tk
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.parquet")):
                if p.stem not in want:
                    continue
                size = p.stat().st_size
                entries.append(f"{rel(p)}:{size}")
                n_files += 1
                total += size
    return {"hash": _sha("\n".join(entries))[:16],
            "n_files": n_files,
            "bytes": total,
            "n_tickers": len(tickers),
            "stores": [rel(s) for s in stores]}


def rel(p) -> str:
    """Path relative to the repo when it is inside it, absolute otherwise.

    NEVER RAISES. `Path.relative_to` throws when the path is outside the root,
    which is every path under a pytest tmp_path and every store pointed at by
    an env override. A display helper that can abort a promotion mid-write is a
    display helper that will."""
    p = Path(p)
    ap = p if p.is_absolute() else (REPO_ROOT / p)
    try:
        return str(ap.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(ap)


def environment() -> dict:
    """Python and pandas versions. A metric that moves because pandas changed
    its resample default is not a strategy finding."""
    import sys
    out = {"python": ".".join(str(x) for x in sys.version_info[:3])}
    try:
        import pandas as pd
        out["pandas"] = pd.__version__
    except Exception:                                       # noqa: BLE001
        pass
    out["host"] = os.uname().nodename if hasattr(os, "uname") else ""
    return out
