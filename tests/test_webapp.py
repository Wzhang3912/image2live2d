"""Phase 5 — local web app upload handler (pure, no socket)."""

from __future__ import annotations

import io
import zipfile

import pytest

from image2live2d.app import handle_upload


def _layers_zip(tmp_path) -> bytes:
    pytest.importorskip("PIL")
    from image2live2d.samples import make_sample_layers

    layer_dir = make_sample_layers(tmp_path / "src")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for png in sorted(layer_dir.glob("*.png")):
            zf.write(png, arcname=png.name)
    return buf.getvalue()


def test_handle_upload_zip_of_layers(tmp_path):
    data = _layers_zip(tmp_path)
    res = handle_upload(data, "hero.zip", tmp_path / "work")
    assert res.name == "hero"
    assert res.passed
    assert res.parts > 0 and res.inp_bytes[:7] == b"TRNSRTS"  # nijilive .inp container magic


def test_handle_upload_nested_zip_is_flattened(tmp_path):
    pytest.importorskip("PIL")
    from image2live2d.samples import make_sample_layers

    layer_dir = make_sample_layers(tmp_path / "src")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for png in sorted(layer_dir.glob("*.png")):
            zf.write(png, arcname=f"some/nested/{png.name}")  # entries inside a folder
    res = handle_upload(buf.getvalue(), "char.zip", tmp_path / "work")
    assert res.passed and res.parts > 0


def test_handle_upload_rejects_flat_image(tmp_path):
    with pytest.raises(ValueError, match="already-separated layers"):
        handle_upload(b"\x89PNG", "photo.png", tmp_path / "work")


def test_handle_upload_empty_zip(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.txt", "no layers here")
    with pytest.raises(ValueError, match="no PNG layers"):
        handle_upload(buf.getvalue(), "x.zip", tmp_path / "work")


# --------------------------------------------------------------------------------------------------
# Async job pipeline (CI/CD-style UI backend)
# --------------------------------------------------------------------------------------------------
def _wait(job, timeout=30.0):
    import time
    t0 = time.monotonic()
    while job.status == "running" and time.monotonic() - t0 < timeout:
        time.sleep(0.05)
    return job


def test_async_job_runs_all_steps_and_previews(tmp_path):
    from image2live2d import app

    data = _layers_zip(tmp_path)
    job = _wait(app.get_job(app.start_job(data, "hero.zip")))
    assert job.status == "done", job.error
    names = [s.name for s in job.steps]
    assert names == ["Decompose", "Build meshes", "Landmarks", "Author rig",
                     "Physics", "Idle animation", "Emit .inp", "QA + audit"]
    assert all(s.status == "ok" for s in job.steps)
    assert job.result["passed"] and job.result["parts"] > 0
    assert job.inp_bytes[:7] == b"TRNSRTS"
    # preview: slider specs + a real PNG render of a driven pose
    specs = app.preview_param_specs(job)
    assert any(p["id"] == "ParamAngleX" for p in specs)
    png = app.render_job(job, {"ParamAngleX": 20.0}, res=128)
    assert png[:4] == b"\x89PNG"


def test_rig_json_export_for_live_runtime(tmp_path):
    """The /rig export feeds the in-browser live runtime: parts (mesh geometry, bottom->top) + every
    param's keyforms with per-part offsets keyed by part index, and one texture per part."""
    from image2live2d import app

    job = _wait(app.get_job(app.start_job(_layers_zip(tmp_path), "hero.zip")))
    assert job.status == "done", job.error
    rig = app.rig_json(job)
    assert rig["nparts"] == len(rig["parts"]) > 0
    p0 = rig["parts"][0]
    assert len(p0["verts"]) == len(p0["uvs"]) and p0["tris"]          # mesh geometry present
    ax = next(p for p in rig["params"] if p["id"] == "ParamAngleX")
    assert ax["keyforms"] and all(isinstance(k, int)                  # offsets keyed by part index
                                  for kf in ax["keyforms"] for k in kf["offsets"])
    # one texture path per exported part, all present on disk
    from image2live2d.app import server as _srv
    parts = _srv._ordered_parts(job)
    assert len(parts) == rig["nparts"]
    assert all((layer.texture_path).exists() for layer, _ in parts)


def test_rig_json_exports_playable_motion_clips(tmp_path):
    """The /rig export carries the shipped motion clips so the live view can *play* them (not just its
    own idle+cursor loop) — each lane is (param id, [[frame, value], ...]) the browser interpolates."""
    from image2live2d import app

    job = _wait(app.get_job(app.start_job(_layers_zip(tmp_path), "hero.zip")))
    assert job.status == "done", job.error
    rig = app.rig_json(job)
    names = {a["name"] for a in rig["anims"]}
    assert "idle" in names and "sweep" in names               # the loop + the every-param inspector
    idle = next(a for a in rig["anims"] if a["name"] == "idle")
    assert idle["loop"] and idle["fps"] > 0 and idle["length"] > 0
    lane = idle["lanes"][0]
    assert lane["p"] and all(len(kf) == 2 for kf in lane["k"])  # [frame, value] pairs
    # every lane targets a real parameter of this rig
    param_ids = {p["id"] for p in rig["params"]}
    assert all(ln["p"] in param_ids for a in rig["anims"] for ln in a["lanes"])


def test_async_job_flat_image_without_service_fails_at_decompose(tmp_path, monkeypatch):
    from image2live2d import app

    monkeypatch.delenv("IMAGE2LIVE2D_DECOMPOSE_URL", raising=False)
    monkeypatch.delenv("IMAGE2LIVE2D_GCP_INSTANCE", raising=False)
    job = _wait(app.get_job(app.start_job(b"\x89PNG\r\n", "photo.png")))
    assert job.status == "error"
    assert job.steps[0].name == "Decompose" and job.steps[0].status == "fail"
    assert "IMAGE2LIVE2D_DECOMPOSE_URL" in job.error


def test_managed_gpu_starts_and_always_releases(tmp_path, monkeypatch):
    """Managed mode (IMAGE2LIVE2D_GCP_INSTANCE): a flat-image job gets Start GPU + Stop GPU steps, and
    the GPU is released even when a later stage fails — so the VM always tears down."""
    from image2live2d import app

    monkeypatch.delenv("IMAGE2LIVE2D_DECOMPOSE_URL", raising=False)

    class FakeGpu:
        def __init__(self): self.acq = self.rel = 0
        def acquire(self, log=lambda m: None):
            self.acq += 1
            return "http://localhost:8000"
        def release(self): self.rel += 1

    fake = FakeGpu()
    monkeypatch.setattr("image2live2d.app.server._gpu_manager", lambda: fake)

    # decompose will fail (no real service at the tunnel URL) — release must still happen
    job = _wait(app.get_job(app.start_job(b"\x89PNG\r\nnot-an-image", "hero.png")))
    assert job.steps[0].name == "Start GPU" and job.steps[0].status == "ok"
    assert fake.acq == 1 and fake.rel == 1          # acquired once, released once (guaranteed)
    assert job.status == "error"                     # (decompose failed, as expected without a GPU)
