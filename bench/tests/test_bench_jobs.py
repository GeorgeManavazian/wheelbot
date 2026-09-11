"""Background run jobs.

No test here spawns a real backtest -- that would take minutes and need the
7GB chain store. They exercise the record layer, which is where the bugs that
matter live: a job that reports `running` forever, a jobs tree that bleeds
between callers, a crash that leaves no trace.
"""
from __future__ import annotations

import json
import os

import pytest

from bench import jobs


def make(tmp_path, **kw) -> jobs.Job:
    j = jobs.Job(id=kw.pop("id", "j1"), variant=kw.pop("variant", "cand"),
                 tier=kw.pop("tier", "smoke"), state=kw.pop("state", jobs.RUNNING),
                 root=str(tmp_path), **kw)
    jobs._write(j)
    return j


def test_a_job_round_trips_through_disk(tmp_path):
    make(tmp_path, note="hello", pid=os.getpid())
    got = jobs.get("j1", root=tmp_path)
    assert got.variant == "cand" and got.tier == "smoke"
    assert got.note == "hello"


def test_the_jobs_root_is_not_shared_between_callers(tmp_path):
    """An earlier version mutated the module global in start(), which silently
    redirected every LATER caller at whichever directory was passed last."""
    a, b = tmp_path / "a", tmp_path / "b"
    make(a, id="ja")
    make(b, id="jb")
    assert jobs.get("ja", root=a) is not None
    assert jobs.get("ja", root=b) is None
    assert jobs.JOBS_DIR == jobs.Path(jobs.__file__).resolve().parent / "jobs"


def test_a_dead_process_is_reported_failed_not_running_forever(tmp_path):
    """The realistic failure: the OS kills the run during an 8GB chain load, so
    the child never gets to write a terminal state. A spinner that never stops
    is the worst possible answer."""
    make(tmp_path, pid=999_999_999)          # a pid that cannot exist
    got = jobs.get("j1", root=tmp_path)
    assert got.state == jobs.FAILED
    assert "disappeared" in got.error
    assert got.finished


def test_a_finished_job_is_left_alone(tmp_path):
    make(tmp_path, state=jobs.DONE, pid=999_999_999, finished="2026-08-14T00:00:00+00:00")
    got = jobs.get("j1", root=tmp_path)
    assert got.state == jobs.DONE
    assert got.error == ""


def test_a_live_process_still_reads_as_running(tmp_path):
    make(tmp_path, pid=os.getpid())
    assert jobs.get("j1", root=tmp_path).state == jobs.RUNNING


def test_a_job_with_no_pid_yet_is_not_declared_dead(tmp_path):
    """Between writing the queued record and Popen returning, pid is 0. That
    window must not read as a failure."""
    make(tmp_path, state=jobs.QUEUED, pid=0)
    assert jobs.get("j1", root=tmp_path).state == jobs.QUEUED


def test_missing_job_returns_none_rather_than_raising(tmp_path):
    assert jobs.get("nope", root=tmp_path) is None


def test_a_corrupt_record_returns_none_rather_than_crashing_the_page(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    (d / "job.json").write_text("{not json")
    assert jobs.get("bad", root=tmp_path) is None
    assert jobs.all_jobs(root=tmp_path) == []


def test_all_jobs_lists_newest_first(tmp_path):
    make(tmp_path, id="20260814-090000-a", state=jobs.DONE)
    make(tmp_path, id="20260814-100000-b", state=jobs.DONE)
    assert [j.id for j in jobs.all_jobs(root=tmp_path)] == \
           ["20260814-100000-b", "20260814-090000-a"]


def test_cancel_marks_the_job_and_is_idempotent(tmp_path):
    make(tmp_path, pid=0)
    j = jobs.cancel("j1", root=tmp_path)
    assert j.state == jobs.CANCELLED and j.finished
    again = jobs.cancel("j1", root=tmp_path)
    assert again.state == jobs.CANCELLED


def test_cancelling_a_finished_job_does_not_rewrite_it(tmp_path):
    make(tmp_path, state=jobs.DONE, finished="2026-08-14T00:00:00+00:00")
    j = jobs.cancel("j1", root=tmp_path)
    assert j.state == jobs.DONE
    assert j.finished == "2026-08-14T00:00:00+00:00"


def test_clear_finished_keeps_running_jobs(tmp_path):
    make(tmp_path, id="done1", state=jobs.DONE)
    make(tmp_path, id="live1", state=jobs.RUNNING, pid=os.getpid())
    assert jobs.clear_finished(root=tmp_path) == 1
    assert [j.id for j in jobs.all_jobs(root=tmp_path)] == ["live1"]


def test_elapsed_is_zero_before_a_start_and_positive_after(tmp_path):
    j = make(tmp_path, started="", state=jobs.QUEUED)
    assert j.elapsed_secs() == 0.0
    j2 = make(tmp_path, id="j2", started="2026-08-14T00:00:00+00:00",
              finished="2026-08-14T00:02:30+00:00", state=jobs.DONE)
    assert j2.elapsed_secs() == pytest.approx(150.0)


def test_tail_of_a_missing_log_is_empty_not_an_exception(tmp_path):
    assert make(tmp_path).tail() == ""


def test_the_child_records_a_failure_instead_of_dying_silently(tmp_path):
    """The child runs the backtest in-process. If the run raises -- a missing
    prerequisite is the common case -- it must land in the record, because a
    dead process with an empty record is indistinguishable from an OOM."""
    make(tmp_path, variant="does-not-exist", tier="smoke")
    rc = jobs._run(tmp_path / "j1")
    assert rc == 1
    j = jobs.get("j1", root=tmp_path)
    assert j.state == jobs.FAILED
    assert "does-not-exist" in j.error or "ConfigError" in j.error
    assert j.finished


def test_the_child_refuses_a_directory_with_no_record(tmp_path):
    assert jobs._run(tmp_path / "empty") == 2


def test_written_records_are_atomic_and_leave_no_tmp_file(tmp_path):
    make(tmp_path)
    assert list(tmp_path.rglob("*.tmp")) == []
    assert json.loads((tmp_path / "j1" / "job.json").read_text())["id"] == "j1"
