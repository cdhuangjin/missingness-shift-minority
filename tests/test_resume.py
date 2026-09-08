"""Resume: completed run keys are not re-executed."""

import tempfile
from pathlib import Path

from src.resume_manager import ResumeManager


def test_completed_keys_are_skipped():
    with tempfile.TemporaryDirectory() as d:
        rm = ResumeManager(Path(d))
        rm.save("a", {"raw": {"x": 1}, "missing": {"y": 2}})
        rm.save("b", {"raw": {"x": 2}, "missing": {"y": 3}})
        assert rm.is_complete("a")
        assert rm.is_complete("b")
        assert rm.completed_count() == 2
        assert not rm.is_complete("c")


def test_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        rm = ResumeManager(Path(d))
        payload = {"raw": [1, 2, 3], "missing": {"a": 1}}
        rm.save("key-x", payload)
        loaded = rm.load("key-x")
        assert loaded["run_key"] == "key-x"
        assert loaded["data"] == payload


def test_failure_recording_and_progress():
    with tempfile.TemporaryDirectory() as d:
        rm = ResumeManager(Path(d))
        rm.record_failure("bad", "ValueError: boom")
        rm.save_progress(3, 10, 1, "last-key")
        progress = rm.load_progress()
        assert progress["completed_runs"] == 3
        assert progress["total_runs"] == 10
        assert progress["failed_runs"] == 1
        assert progress["last_completed_key"] == "last-key"
