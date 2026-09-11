"""Background backtest runs, so a UI can start one and stay usable.

WHY THIS EXISTS. A `live`-tier run loads 531 tickers over 23 months and takes
minutes. Streamlit is synchronous -- a run started inside a page freezes the
whole app until it finishes, and any interaction in the meantime is lost. So the
run is spawned as a DETACHED child process and every scrap of its state lives on
disk under `bench/jobs/<id>/`.

ON DISK, NOT IN MEMORY, and that is the whole design. Streamlit re-executes the
entire script on every click, so anything held in a Python variable is gone by
the next interaction; `st.session_state` survives clicks but not a page reload or
a second browser tab. A job record in a file survives all three, plus the app
being restarted mid-run -- you can close the dashboard, reopen it, and the run is
still there.

DETACHED, so the child outlives its parent: `start_new_session=True` puts it in
its own process group, which means a Streamlit restart (or Ctrl-C) does not take
the backtest down with it.

HOW COMPLETION IS DETECTED. The child is `python -m bench.jobs <job_dir>`, which
runs the job in-process and writes its own terminal state. If it dies so hard it
cannot write anything -- OOM on a 8GB chain load is the realistic case -- the
reader notices the pid is gone with no terminal state recorded and reports
`failed`, rather than showing `running` forever. A job that lies about still
running is worse than one that admits it died.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

JOBS_DIR = Path(__file__).resolve().parent / "jobs"
REPO_ROOT = Path(__file__).resolve().parent.parent

QUEUED, RUNNING, DONE, FAILED, CANCELLED = (
    "queued", "running", "done", "failed", "cancelled")
TERMINAL = (DONE, FAILED, CANCELLED)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    variant: str
    tier: str
    state: str
    started: str = ""
    finished: str = ""
    pid: int = 0
    base: str = ""
    note: str = ""
    scorecard: str = ""
    error: str = ""
    returncode: int | None = None
    # Which jobs tree this record lives in. Serialized so a job read back from
    # disk knows where it came from -- an earlier version mutated the module
    # global instead, which silently redirected every LATER caller (and every
    # subsequent test) at whichever directory was passed last.
    root: str = ""

    @property
    def dir(self) -> Path:
        return Path(self.root or JOBS_DIR) / self.id

    @property
    def log_path(self) -> Path:
        return self.dir / "output.log"

    @property
    def running(self) -> bool:
        return self.state in (QUEUED, RUNNING)

    def tail(self, n: int = 40) -> str:
        try:
            lines = self.log_path.read_text(errors="replace").split("\n")
        except OSError:
            return ""
        return "\n".join(lines[-n:])

    def elapsed_secs(self) -> float:
        if not self.started:
            return 0.0
        end = self.finished or _now()
        try:
            a = datetime.fromisoformat(self.started)
            b = datetime.fromisoformat(end)
            return max(0.0, (b - a).total_seconds())
        except ValueError:
            return 0.0


# ---------------------------------------------------------------------------
# Record io
# ---------------------------------------------------------------------------

def _record_path(job_dir: Path) -> Path:
    return Path(job_dir) / "job.json"


def _write(job: Job) -> None:
    job.dir.mkdir(parents=True, exist_ok=True)
    p = _record_path(job.dir)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(job), indent=1), encoding="utf-8")
    tmp.replace(p)


def _read(job_dir: Path) -> Job | None:
    try:
        d = json.loads(_record_path(job_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return Job(**{k: v for k, v in d.items() if k in Job.__dataclass_fields__})


def _pid_alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True          # exists, owned by someone else
    return True


# ---------------------------------------------------------------------------
# Client API
# ---------------------------------------------------------------------------

def start(variant: str, tier: str = "", base: str = "", note: str = "",
          root: Path | None = None) -> Job:
    """Spawn a detached backtest and return immediately."""
    job = Job(id=f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{variant}-"
                 f"{uuid.uuid4().hex[:6]}",
              variant=variant, tier=tier, base=base, note=note,
              state=QUEUED, started=_now(),
              root=str(root) if root else "")
    _write(job)

    log = open(job.log_path, "wb")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "bench.jobs", str(job.dir)],
            cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,          # outlives a Streamlit restart
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT),
                 "PYTHONUNBUFFERED": "1"})
    finally:
        log.close()
    job.pid = proc.pid
    job.state = RUNNING
    _write(job)
    return job


def get(job_id: str, root: Path | None = None) -> Job | None:
    """Read a job, reconciling the record against reality.

    A job whose process is gone but whose record still says RUNNING died without
    being able to write -- the realistic cause is the kernel killing it during
    an 8GB chain load. Reporting that as `failed` is the point; leaving it as
    `running` would show a spinner forever."""
    d = Path(root or JOBS_DIR) / job_id
    job = _read(d)
    if job is None:
        return None
    if job.running and job.pid and not _pid_alive(job.pid):
        job.state = FAILED
        job.finished = job.finished or _now()
        job.error = job.error or (
            "the run process disappeared without recording a result -- most "
            "likely killed by the OS during a large chain load. See output.log.")
        _write(job)
    return job


def all_jobs(root: Path | None = None, limit: int = 50) -> list:
    d = Path(root or JOBS_DIR)
    if not d.is_dir():
        return []
    out = []
    for sub in sorted(d.iterdir(), reverse=True):
        if not sub.is_dir():
            continue
        j = get(sub.name, root=d)
        if j:
            out.append(j)
        if len(out) >= limit:
            break
    return out


def cancel(job_id: str, root: Path | None = None) -> Job | None:
    job = get(job_id, root)
    if job is None or not job.running:
        return job
    if job.pid:
        try:
            # The child is its own session leader (start_new_session), so
            # signalling the GROUP reaps any helper it spawned too.
            os.killpg(os.getpgid(job.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    job.state = CANCELLED
    job.finished = _now()
    _write(job)
    return job


def clear_finished(root: Path | None = None) -> int:
    """Delete terminal jobs' directories. Scorecards are NOT touched -- they
    live under bench/scorecards/ and are the durable artifact."""
    import shutil
    d = Path(root or JOBS_DIR)
    n = 0
    for job in all_jobs(root=d, limit=1000):
        if not job.running:
            shutil.rmtree(d / job.id, ignore_errors=True)
            n += 1
    return n


# ---------------------------------------------------------------------------
# The child process
# ---------------------------------------------------------------------------

def _run(job_dir: Path) -> int:
    """Entry point of the detached child: `python -m bench.jobs <job_dir>`."""
    job = _read(job_dir)
    if job is None:
        print(f"no job record in {job_dir}", file=sys.stderr)
        return 2
    job.state = RUNNING
    job.pid = os.getpid()
    job.started = job.started or _now()
    _write(job)

    try:
        from bench import fingerprint, run as run_mod
        card = run_mod.run(job.variant, tier_name=job.tier or None,
                           base_override=job.base or None, save=True,
                           verbose=True, notes=job.note)
        job.scorecard = fingerprint.rel(card.path) if card.path else ""
        job.state = DONE
        job.returncode = 0
    except BaseException as e:                              # noqa: BLE001
        # BaseException on purpose: a MemoryError or a SIGTERM-driven
        # KeyboardInterrupt must still be RECORDED, not vanish into a dead
        # process that the reader has to guess about.
        import traceback
        traceback.print_exc()
        job.state = FAILED
        job.error = f"{type(e).__name__}: {e}"
        job.returncode = 1
    finally:
        job.finished = _now()
        _write(job)
    return job.returncode or 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m bench.jobs <job_dir>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(_run(Path(sys.argv[1])))
