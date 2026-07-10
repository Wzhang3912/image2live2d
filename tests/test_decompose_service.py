"""Phase 5 Tier-2 — remote decompose client (decompose.from_service). HTTP is mocked.

The client uses an async job protocol: POST /decompose -> {job_id}; poll /jobs/{id}; GET
/jobs/{id}/result -> PSD. Each call is short so a long inference can't drop the connection."""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

from image2live2d.core import decompose
from image2live2d.core.types import LayerStack


class _FakeResp:
    def __init__(self, data: bytes):
        self._d = data

    def read(self) -> bytes:
        return self._d

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_from_service_async_job_flow(tmp_path, monkeypatch):
    img = tmp_path / "hero.png"
    img.write_bytes(b"IMGDATA")
    calls: list[tuple[str, str]] = []

    def fake_urlopen(req, timeout=None):
        url, method = req.full_url, req.method
        calls.append((method, url))
        if method == "POST" and url.endswith("/decompose"):
            assert req.data == b"IMGDATA"
            assert req.headers.get("X-auth-token") == "secret"  # urllib capitalizes header keys
            return _FakeResp(json.dumps({"job_id": "JOB1"}).encode())
        if url.endswith("/jobs/JOB1"):
            return _FakeResp(json.dumps({"status": "done", "error": None}).encode())
        if url.endswith("/jobs/JOB1/result"):
            return _FakeResp(b"PSDBYTES")
        raise AssertionError(f"unexpected request {method} {url}")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    sentinel = LayerStack(layers=[], canvas_width=1, canvas_height=1)
    captured: dict = {}

    def fake_from_psd(psd_path, work, role_map=None):
        captured["psd"] = Path(psd_path)
        return sentinel

    monkeypatch.setattr(decompose, "from_psd", fake_from_psd)

    stack = decompose.from_service(img, "http://svc:8000/", tmp_path / "w",
                                   token="secret", poll_interval=0.0)

    assert stack is sentinel
    assert ("POST", "http://svc:8000/decompose") in calls
    assert any(m == "GET" and u.endswith("/jobs/JOB1") for m, u in calls)        # polled status
    assert any(u.endswith("/jobs/JOB1/result") for _, u in calls)               # fetched result
    assert captured["psd"].read_bytes() == b"PSDBYTES"


def test_from_service_raises_on_job_error(tmp_path, monkeypatch):
    img = tmp_path / "hero.png"
    img.write_bytes(b"X")

    def fake_urlopen(req, timeout=None):
        if req.full_url.endswith("/decompose"):
            return _FakeResp(json.dumps({"job_id": "J"}).encode())
        return _FakeResp(json.dumps({"status": "error", "error": "see-through boom"}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="boom"):
        decompose.from_service(img, "http://svc:8000", tmp_path / "w", poll_interval=0.0)


def test_from_service_retries_when_service_drops_the_job(tmp_path, monkeypatch):
    """If the service restarts and loses the job (404 'unknown job id'), the client re-submits and
    succeeds. Regression for the auto-managed GPU cold-start where a fresh service settles once."""
    import io
    img = tmp_path / "hero.png"; img.write_bytes(b"IMG")
    seen = {"submits": 0}

    def fake_urlopen(req, timeout=None):
        url, method = req.full_url, req.method
        if method == "POST" and url.endswith("/decompose"):
            seen["submits"] += 1
            return _FakeResp(json.dumps({"job_id": f"J{seen['submits']}"}).encode())
        if url.endswith("/jobs/J1"):  # first job vanished (service restarted) -> 404
            raise urllib.error.HTTPError(url, 404, "Not Found", {},
                                         io.BytesIO(b'{"detail":"unknown job id"}'))
        if url.endswith("/jobs/J2"):
            return _FakeResp(json.dumps({"status": "done", "error": None}).encode())
        if url.endswith("/jobs/J2/result"):
            return _FakeResp(b"PSD2")
        raise AssertionError(f"unexpected {method} {url}")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(decompose, "from_psd",
                        lambda psd_path, work, role_map=None: LayerStack(layers=[], canvas_width=1, canvas_height=1))
    decompose.from_service(img, "http://svc:8000", tmp_path / "w", poll_interval=0.0)  # must not raise
    assert seen["submits"] == 2  # re-submitted once after the 404


def test_from_service_missing_image(tmp_path):
    with pytest.raises(FileNotFoundError):
        decompose.from_service(tmp_path / "nope.png", "http://svc:8000", tmp_path / "w")


def test_from_service_wraps_network_error(tmp_path, monkeypatch):
    img = tmp_path / "hero.png"
    img.write_bytes(b"X")

    def boom(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(RuntimeError, match="unreachable"):
        decompose.from_service(img, "http://svc:8000", tmp_path / "w")
