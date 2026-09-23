"""The ML service is stopped between runs, so waking it must be reliable
and, when it fails, must never take the request down with it.
"""

from __future__ import annotations

import time

import pytest

from app.ml.client import MLServiceClient, MLServiceUnavailableError
from app.ml.wake import mark_activity, wake_and_wait


@pytest.fixture
def sentinel(tmp_path, monkeypatch):
    path = tmp_path / "run" / "ml.wanted"
    monkeypatch.setenv("ML_WAKE_FILE", str(path))
    return path


def _client(responses):
    """A client whose health check returns each response in turn."""
    calls = {"n": 0}

    def get_json(url, timeout_s):
        i = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        outcome = responses[i]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    client = MLServiceClient(base_url="http://ml:8001", get_json=get_json)
    return client, calls


def test_mark_activity_creates_the_sentinel_and_its_directory(sentinel):
    assert not sentinel.parent.exists()
    mark_activity()
    assert sentinel.exists()


def test_mark_activity_advances_the_mtime(sentinel):
    mark_activity()
    first = sentinel.stat().st_mtime_ns
    time.sleep(0.01)
    mark_activity()
    assert sentinel.stat().st_mtime_ns >= first


def test_mark_activity_is_a_noop_when_unconfigured(monkeypatch, tmp_path):
    monkeypatch.delenv("ML_WAKE_FILE", raising=False)
    mark_activity()  # must not raise


def test_mark_activity_survives_an_unwritable_path(monkeypatch, tmp_path):
    # A deployment can hand us a volume that is not mounted; nobody is
    # supervising in that case, which is not an error.
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    monkeypatch.setenv("ML_WAKE_FILE", str(blocker / "sub" / "ml.wanted"))
    mark_activity()


def test_wake_returns_immediately_when_already_up(sentinel):
    client, calls = _client([{"status": "ok"}])
    assert wake_and_wait(client, timeout_s=5.0) is True
    assert calls["n"] == 1
    assert sentinel.exists(), "waking must also count as activity"


def test_wake_polls_until_the_service_answers(sentinel, monkeypatch):
    monkeypatch.setattr("app.ml.wake._POLL_INTERVAL_S", 0.01)
    down = MLServiceUnavailableError("starting")
    client, calls = _client([down, down, {"status": "ok"}])
    assert wake_and_wait(client, timeout_s=5.0) is True
    assert calls["n"] == 3


def test_wake_gives_up_without_raising(sentinel, monkeypatch):
    # Ingestion still produces native evidence when OCR is unavailable, so
    # a timeout must be reported, not raised.
    monkeypatch.setattr("app.ml.wake._POLL_INTERVAL_S", 0.01)
    client, _ = _client([MLServiceUnavailableError("down")])
    assert wake_and_wait(client, timeout_s=0.05) is False
