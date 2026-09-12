"""Playground/exam split with sha256 checksum sealing + exam access log."""
from __future__ import annotations
import hashlib
import pathlib
from datetime import datetime, timezone
import pandas as pd

PLAYGROUND = (pd.Timestamp("2007-01-01"), pd.Timestamp("2020-12-31"))

EXAM_START = pd.Timestamp("2021-01-01")

def exam_bounds(today: pd.Timestamp | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Compute exam window bounds at call time, not import time.
    Passing `today` explicitly makes tests deterministic."""
    end = pd.Timestamp(today).normalize() if today else pd.Timestamp.today().normalize()
    return (EXAM_START, end)

# Retain module-level EXAM for backward compat; document that it is a lazy snapshot.
EXAM = exam_bounds()

class ExamAccessDenied(RuntimeError):
    pass

def slice_playground(bars: pd.DataFrame) -> pd.DataFrame:
    return bars.loc[PLAYGROUND[0]:PLAYGROUND[1]]

def slice_exam(bars: pd.DataFrame, allow: bool, run_id: str,
               log_path: str = "logs/exam_access.log",
               today: pd.Timestamp | None = None) -> pd.DataFrame:
    if not allow:
        raise ExamAccessDenied("exam split touched without allow=True")
    p = pathlib.Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    with p.open("a") as f:
        f.write(f"{now}\trun_id={run_id}\n")
    bounds = exam_bounds(today)
    return bars.loc[bounds[0]:bounds[1]]

def checksum_split(df: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(payload).hexdigest()

def seal_splits(bars: pd.DataFrame, sealed_dir: str = "splits/") -> dict:
    d = pathlib.Path(sealed_dir)
    d.mkdir(parents=True, exist_ok=True)
    sealed_file = d / "SEALED.txt"
    if sealed_file.exists():
        raise FileExistsError(f"{sealed_file} already sealed; remove manually to reseal")
    info = {
        "playground_sha256": checksum_split(slice_playground(bars)),
        "sealed_at": datetime.now(timezone.utc).isoformat(),
    }
    _exam_bounds = exam_bounds()
    exam = bars.loc[_exam_bounds[0]:_exam_bounds[1]]
    if len(exam):
        info["exam_sha256"] = checksum_split(exam)
    sealed_file.write_text(
        "\n".join(f"{k}: {v}" for k, v in info.items()) + "\n"
    )
    import stat as _stat
    sealed_file.chmod(_stat.S_IRUSR | _stat.S_IRGRP | _stat.S_IROTH)
    return info
