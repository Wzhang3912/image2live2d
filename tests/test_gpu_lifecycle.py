"""Phase 5 Tier-2 — GCP auto start/stop lifecycle (gcloud + health HTTP are mocked)."""

from __future__ import annotations

import pytest

from image2live2d import gpu
from image2live2d.core import decompose
from image2live2d.core.types import LayerStack


class _Resp:
    status = 200

    def read(self):
        return b'{"ok": true}'

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_gcloud(record):
    def run(cmd, capture_output=True, text=True):
        record.append(cmd)

        class P:
            returncode = 0
            stderr = ""
            stdout = "1.2.3.4" if "describe" in cmd else ""
        return P()
    return run


def test_start_resolves_ip_and_waits_healthy(monkeypatch):
    calls = []
    monkeypatch.setattr("subprocess.run", _fake_gcloud(calls))
    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=10: _Resp())

    svc = gpu.GpuService("svc", "us-central1-a", poll_interval=0)
    url = svc.start()

    assert url == "http://1.2.3.4:8000"
    assert any("start" in c for c in calls) and any("describe" in c for c in calls)


def test_context_manager_starts_and_stops(monkeypatch):
    calls = []
    monkeypatch.setattr("subprocess.run", _fake_gcloud(calls))
    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=10: _Resp())

    with gpu.GpuService("svc", "z", poll_interval=0) as svc:
        assert svc.base_url == "http://1.2.3.4:8000"
    assert any("stop" in c for c in calls)  # stopped on exit


def test_decompose_managed_stops_even_on_error(monkeypatch):
    calls = []
    monkeypatch.setattr("subprocess.run", _fake_gcloud(calls))
    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=10: _Resp())

    def boom(*a, **k):
        raise RuntimeError("decompose blew up")

    monkeypatch.setattr(decompose, "from_service", boom)

    with pytest.raises(RuntimeError, match="blew up"):
        gpu.decompose_managed("img.png", "work/", instance="svc", zone="z")
    assert any("stop" in c for c in calls)  # VM stopped despite the failure


def test_decompose_managed_happy_path(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("subprocess.run", _fake_gcloud(calls))
    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=10: _Resp())
    sentinel = LayerStack(layers=[], canvas_width=1, canvas_height=1)
    seen = {}

    def fake_from_service(image, url, work, **kw):
        seen["url"] = url
        return sentinel

    monkeypatch.setattr(decompose, "from_service", fake_from_service)

    out = gpu.decompose_managed("img.png", tmp_path, instance="svc", zone="z")
    assert out is sentinel
    assert seen["url"] == "http://1.2.3.4:8000"
    assert [c for c in calls if "start" in c] and [c for c in calls if "stop" in c]


def test_keep_running_skips_stop(monkeypatch):
    calls = []
    monkeypatch.setattr("subprocess.run", _fake_gcloud(calls))
    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=10: _Resp())
    monkeypatch.setattr(decompose, "from_service",
                        lambda *a, **k: LayerStack(layers=[], canvas_width=1, canvas_height=1))

    gpu.decompose_managed("img.png", "w/", instance="svc", zone="z", stop_on_finish=False)
    assert not any("stop" in c for c in calls)
