"""Local web app — upload art, watch the pipeline run, preview & download the riggable ``.inp``.

Two surfaces over a stdlib ``http.server`` (no extra deps):

* ``handle_upload`` — the original synchronous PSD/zip-of-layers -> ``.inp`` path (pure, unit-tested).
* an **async job API + single-page UI** — upload a flat GPT-image (decomposed via a See-through GPU
  service, ``IMAGE2LIVE2D_DECOMPOSE_URL``) or a layered ``.psd``/``.zip``; the page shows each pipeline
  stage running CI/CD-style with timings + pass/fail, then an interactive deformation preview (sliders
  drive the rig, server renders the pose) and a download button.

Run it::

    python -m image2live2d --serve                 # http://127.0.0.1:8000
    IMAGE2LIVE2D_DECOMPOSE_URL=http://VM:8000 python -m image2live2d --serve   # enable flat-image upload
"""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import tempfile
import threading
import time
import traceback
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..api import convert_layers, convert_psd  # api lives at the package root

# --------------------------------------------------------------------------------------------------
# Synchronous upload handler (unchanged contract — tests depend on it)
# --------------------------------------------------------------------------------------------------


@dataclass
class UploadResult:
    name: str
    inp_bytes: bytes
    parts: int
    params: int
    physics: int
    passed: bool
    reasons: list[str]


SUPPORTED = (".psd", ".zip")
_FLAT_IMAGE = (".png", ".jpg", ".jpeg", ".webp")


def handle_upload(data: bytes, filename: str, work_root: str | Path) -> UploadResult:
    """Convert an uploaded PSD or zip-of-layer-PNGs into a ``.inp`` (+ QA). Pure: no HTTP.

    ``.zip`` entries are flattened to their basenames, so a zip of ``{order}_{role}.png`` files works
    whether or not they're inside a folder."""
    work = Path(work_root)
    work.mkdir(parents=True, exist_ok=True)
    name = Path(filename).stem or "model"
    ext = Path(filename).suffix.lower()
    out = work / "out"

    if ext == ".psd":
        psd = work / f"{name}.psd"
        psd.write_bytes(data)
        result = convert_psd(psd, out, name=name)
    elif ext == ".zip":
        layers = work / "layers"
        layers.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            pngs = [n for n in zf.namelist() if n.lower().endswith(".png") and not n.endswith("/")]
            if not pngs:
                raise ValueError("zip contains no PNG layers")
            for n in pngs:
                (layers / Path(n).name).write_bytes(zf.read(n))
        result = convert_layers(layers, out, name=name)
    else:
        raise ValueError(
            f"unsupported upload {ext!r}: this build needs already-separated layers — upload a .psd "
            "or a .zip of '{order}_{role}.png' files (single-image decomposition is gated)"
        )

    return UploadResult(
        name=result.name,
        inp_bytes=result.inp_path.read_bytes(),
        parts=result.qa.parts,
        params=result.qa.params,
        physics=result.qa.physics,
        passed=result.qa.passed,
        reasons=result.qa.reasons,
    )


# --------------------------------------------------------------------------------------------------
# Async job pipeline (powers the CI/CD-style UI)
# --------------------------------------------------------------------------------------------------

# slider-worthy params for the live preview, in display order
_PREVIEW_PARAMS = [
    "ParamAngleX", "ParamAngleY", "ParamAngleZ", "ParamEyeLOpen", "ParamEyeROpen",
    "ParamEyeBallX", "ParamEyeBallY", "ParamMouthOpenY", "ParamMouthForm",
    "ParamBodyAngleX", "ParamBodyAngleZ",
]
_RUNAWAY = 0.30


@dataclass
class _Step:
    name: str
    status: str = "pending"   # pending | running | ok | fail | skip
    seconds: float = 0.0


@dataclass
class _Job:
    id: str
    name: str
    status: str = "running"   # running | done | error
    error: str = ""
    steps: list[_Step] = field(default_factory=list)
    result: dict = field(default_factory=dict)
    # cached artifacts (not serialized)
    work: Path | None = None
    stack: object = None
    meshes: object = None
    params: object = None
    physics: object = None
    inp_bytes: bytes = b""
    source: bytes = b""

    def public(self) -> dict:
        return {
            "id": self.id, "name": self.name, "status": self.status, "error": self.error,
            "steps": [{"name": s.name, "status": s.status, "seconds": round(s.seconds, 2)}
                      for s in self.steps],
            "result": self.result, "has_inp": bool(self.inp_bytes),
        }


_JOBS: dict[str, _Job] = {}
_LOCK = threading.Lock()


def _decompose_config() -> tuple[str | None, str | None]:
    return os.environ.get("IMAGE2LIVE2D_DECOMPOSE_URL"), os.environ.get("IMAGE2LIVE2D_DECOMPOSE_TOKEN")


class _GpuTunnel:
    """Auto start/stop the See-through GPU VM + a secure SSH tunnel around flat-image decompose.

    On the first flat-image job it starts the VM (retrying through ``STOCKOUT``), opens an SSH port
    forward (laptop ``localhost:port`` -> VM ``localhost:8000`` — no firewall), and waits for the
    service to report healthy. Jobs share one VM via a refcount; when the last one finishes the VM is
    torn down after ``grace`` seconds (so back-to-back uploads reuse a warm VM). Uses local ``gcloud``
    auth. Serialized by a lock — a single L4 handles one decompose at a time anyway."""

    def __init__(self, instance, zone, project=None, port=8000, grace=120.0):
        self.instance, self.zone, self.project = instance, zone, project
        self.port, self.grace = port, grace
        self._lock = threading.RLock()
        self._refs = 0
        self._proc = None
        self._up = False
        self._timer = None
        self.status = "idle"  # idle | starting | up | stopping (surfaced to the UI)

    def base_url(self) -> str:
        return f"http://localhost:{self.port}"

    def _gcloud(self, *args, timeout=120):
        cmd = ["gcloud", "compute", *args, f"--zone={self.zone}"]
        if self.project:
            cmd.append(f"--project={self.project}")
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def acquire(self, log=lambda m: None) -> str:
        """Ensure the VM + tunnel are up (starting them if needed) and return the base URL."""
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            self._refs += 1
            if not self._up:
                self._start(log)
            return self.base_url()

    def _start(self, log):
        self.status = "starting"
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            st = self._gcloud("instances", "describe", self.instance,
                              "--format=value(status)").stdout.strip()
            if st == "RUNNING":
                break
            log("starting GPU VM (waiting for L4 capacity)…")
            out = self._gcloud("instances", "start", self.instance, "--quiet")
            blob = (out.stderr + out.stdout).upper()
            time.sleep(30 if ("STOCKOUT" in blob or "EXHAUST" in blob or "DOES NOT HAVE" in blob) else 10)
        else:
            raise RuntimeError("GPU VM did not reach RUNNING (L4 capacity) within 20 min")
        # Warm SSH + confirm the service is healthy ON the VM first (this also propagates the SSH key /
        # populates known_hosts, so the -N port-forward below won't stall on a first-connect prompt),
        # restarting it if a cold boot hasn't brought it up yet.
        log("connecting to GPU, checking service…")
        # Restart the service AT MOST ONCE (only if it's not already answering), then poll without
        # restarting again — repeatedly restarting a slow-booting service leaves it fragile and it can
        # drop the just-submitted job (the 404 "unknown job id"). One nudge, then let it settle.
        self._gcloud("ssh", self.instance, "--ssh-flag=-o ConnectTimeout=15",
                     "--command=curl -fs http://localhost:8000/health >/dev/null || "
                     "sudo systemctl restart seethrough", timeout=60)
        healthy = False
        for _ in range(45):  # ~6 min for sshd + a cold service start (no further restarts)
            r = self._gcloud("ssh", self.instance, "--ssh-flag=-o ConnectTimeout=10",
                             "--command=curl -fs http://localhost:8000/health >/dev/null && echo I2L_OK",
                             timeout=30)
            if "I2L_OK" in r.stdout:
                healthy = True
                break
            time.sleep(8)
        if not healthy:
            raise RuntimeError("See-through service did not become healthy on the VM")
        log("opening secure SSH tunnel…")
        cmd = ["gcloud", "compute", "ssh", self.instance, f"--zone={self.zone}"]
        if self.project:
            cmd.append(f"--project={self.project}")
        cmd += ["--", "-N", "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30",
                "-o", "ExitOnForwardFailure=yes", "-L", f"{self.port}:localhost:8000"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log("waiting for the tunnel…")
        for _ in range(30):  # service already confirmed healthy; just wait for the forward to bind
            try:
                urllib.request.urlopen(self.base_url() + "/health", timeout=5)
                self._up = True
                self.status = "up"
                log("GPU ready")
                return
            except Exception:  # noqa: BLE001
                if self._proc.poll() is not None:  # tunnel process died — surface it
                    break
                time.sleep(4)
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None
        raise RuntimeError("SSH tunnel did not forward to the decompose service")

    def release(self):
        """Drop a reference; schedule teardown when the last job finishes (after the grace window)."""
        with self._lock:
            self._refs = max(0, self._refs - 1)
            if self._refs == 0 and self._up and not self._timer:
                self._timer = threading.Timer(self.grace, self._teardown)
                self._timer.daemon = True
                self._timer.start()

    def _teardown(self):
        with self._lock:
            self._timer = None
            if self._refs > 0:
                return
            self.status = "stopping"
            if self._proc:
                self._proc.terminate()
                self._proc = None
            try:
                self._gcloud("instances", "stop", self.instance, "--quiet")
            except Exception:  # noqa: BLE001
                pass
            self._up = False
            self.status = "idle"


_GPU: "_GpuTunnel | None" = None
_GPU_LOCK = threading.Lock()


def _gpu_manager() -> "_GpuTunnel | None":
    """The managed GPU tunnel if configured (``IMAGE2LIVE2D_GCP_INSTANCE`` set and no static URL),
    else None. A static ``IMAGE2LIVE2D_DECOMPOSE_URL`` takes precedence (assume it's already up)."""
    if os.environ.get("IMAGE2LIVE2D_DECOMPOSE_URL"):
        return None
    inst = os.environ.get("IMAGE2LIVE2D_GCP_INSTANCE")
    if not inst:
        return None
    global _GPU
    with _GPU_LOCK:
        if _GPU is None:
            _GPU = _GpuTunnel(
                inst,
                os.environ.get("IMAGE2LIVE2D_GCP_ZONE", "us-central1-a"),
                os.environ.get("IMAGE2LIVE2D_GCP_PROJECT"),
                int(os.environ.get("IMAGE2LIVE2D_GCP_PORT", "8000")),
                float(os.environ.get("IMAGE2LIVE2D_GCP_GRACE", "120")),
            )
        return _GPU


def _flat_decompose_available() -> bool:
    return bool(os.environ.get("IMAGE2LIVE2D_DECOMPOSE_URL") or os.environ.get("IMAGE2LIVE2D_GCP_INSTANCE"))


def start_job(data: bytes, filename: str) -> str:
    """Register a job and kick off the pipeline on a worker thread. Returns the job id."""
    jid = uuid.uuid4().hex[:12]
    job = _Job(id=jid, name=Path(filename).stem or "model", source=data)
    job.work = Path(tempfile.mkdtemp(prefix=f"i2l_job_{jid}_"))
    with _LOCK:
        _JOBS[jid] = job
    threading.Thread(target=_run_pipeline, args=(job, data, filename), daemon=True).start()
    return jid


def get_job(jid: str) -> "_Job | None":
    with _LOCK:
        return _JOBS.get(jid)


def _run_pipeline(job: _Job, data: bytes, filename: str) -> None:
    """Execute the pipeline stage-by-stage, timing each and recording pass/fail. Caches artifacts so
    the preview/download endpoints can serve from this job."""
    from ..backends.nijilive import NijiliveEmitter
    from ..core import decompose, mesh as meshmod, motion, physics
    from ..core.assemble import assemble_rig
    from ..core.qa import evaluate
    from ..core.rig import author_rig, select_template
    from ..pipeline import _lift_occluded_accessories, _safe_landmarks
    from .. import preview

    ext = Path(filename).suffix.lower()
    work = job.work
    ctx: dict = {}

    def step(name: str, fn):
        s = _Step(name=name, status="running")
        job.steps.append(s)
        t0 = time.monotonic()
        try:
            fn()
            s.status = "ok"
        except Exception as exc:  # noqa: BLE001 — surface any stage failure to the UI
            s.status = "fail"
            s.seconds = time.monotonic() - t0
            job.status = "error"
            job.error = f"{name}: {exc}"
            raise
        finally:
            if s.status != "fail":
                s.seconds = time.monotonic() - t0

    def s_decompose():
        url, token = _decompose_config()
        if ext == ".psd":
            psd = work / f"{job.name}.psd"
            psd.write_bytes(data)
            ctx["stack"] = decompose.from_psd(psd, work / "layers")
        elif ext == ".zip":
            layers = work / "layers"
            layers.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                pngs = [n for n in zf.namelist() if n.lower().endswith(".png") and not n.endswith("/")]
                if not pngs:
                    raise ValueError("zip contains no PNG layers")
                for n in pngs:
                    (layers / Path(n).name).write_bytes(zf.read(n))
            ctx["stack"] = decompose.from_layer_dir(layers)
        elif ext in _FLAT_IMAGE:
            target = ctx.get("decompose_url") or url  # managed tunnel URL, else static
            if not target:
                raise ValueError("flat-image upload needs a See-through decompose service — set "
                                 "IMAGE2LIVE2D_DECOMPOSE_URL, or IMAGE2LIVE2D_GCP_INSTANCE for "
                                 "auto start/stop (or upload a .psd / .zip of layers)")
            img = work / f"{job.name}{ext}"
            img.write_bytes(data)
            ctx["stack"] = decompose.from_service(img, target, work_dir=work / "layers", token=token)
        else:
            raise ValueError(f"unsupported upload {ext!r}")
        job.stack = ctx["stack"]

    def s_meshes():
        ctx["meshes"] = meshmod.build_meshes(ctx["stack"])
        _lift_occluded_accessories(ctx["stack"], ctx["meshes"])
        job.meshes = ctx["meshes"]

    def s_landmarks():
        ctx["landmarks"] = _safe_landmarks(ctx["stack"])

    def s_rig():
        auth = author_rig(ctx["stack"], ctx["meshes"], select_template(ctx["stack"]),
                          landmarks=ctx["landmarks"])
        ctx["auth"] = auth
        job.params = auth.parameters

    def s_physics():
        try:
            ctx["physics"] = physics.generate_physics(
                ctx["stack"], ctx["auth"].parameters, meshes=ctx["meshes"])
        except NotImplementedError:
            ctx["physics"] = []
        job.physics = ctx["physics"]

    def s_animation():
        ctx["anims"] = (motion.generate_idle(ctx["auth"].parameters)
                        + motion.generate_expressions(ctx["auth"].parameters))

    def s_emit():
        rig = assemble_rig(
            name=job.name, source=None, stack=ctx["stack"], meshes=ctx["meshes"],
            deformers=ctx["auth"].deformers, parameters=ctx["auth"].parameters,
            physics=ctx["physics"], animations=ctx["anims"],
        )
        ctx["rig"] = rig
        out = work / "out"
        inp = NijiliveEmitter().emit(rig, out)
        job.inp_bytes = Path(inp).read_bytes()

    def s_qa():
        rep = evaluate(ctx["rig"], job.name)
        dead, runaway = [], []
        for p in ctx["auth"].parameters:
            moved = preview.max_displacement(p, preview.extreme_value(p))
            if not moved:
                dead.append(p.id)
            elif max(moved.values()) > _RUNAWAY:
                runaway.append(p.id)
        job.result = {
            "parts": len(ctx["stack"].layers), "params": len(ctx["auth"].parameters),
            "physics": len(ctx["physics"]), "passed": rep.passed, "reasons": rep.reasons,
            "dead": dead, "runaway": runaway,
        }

    # Managed GPU: for a flat image with no static URL, auto start the VM + tunnel, then tear down.
    gpu = _gpu_manager() if ext in _FLAT_IMAGE and not _decompose_config()[0] else None

    def s_start_gpu():
        ctx["gpu"] = gpu
        ctx["decompose_url"] = gpu.acquire(log=lambda m: setattr(job.steps[-1], "name", f"Start GPU — {m}"))
        job.steps[-1].name = "Start GPU"  # restore clean label once ready

    def s_stop_gpu():
        if ctx.get("gpu") and not ctx.get("gpu_released"):
            ctx["gpu"].release()
            ctx["gpu_released"] = True

    core = [
        ("Decompose", s_decompose), ("Build meshes", s_meshes), ("Landmarks", s_landmarks),
        ("Author rig", s_rig), ("Physics", s_physics), ("Idle animation", s_animation),
        ("Emit .inp", s_emit), ("QA + audit", s_qa),
    ]
    pipeline = ([("Start GPU", s_start_gpu)] + core + [("Stop GPU", s_stop_gpu)]) if gpu else core
    try:
        for name, fn in pipeline:
            step(name, fn)
        job.status = "done"
    except Exception:
        for s in job.steps:
            if s.status == "pending":
                s.status = "skip"
        if not job.error:
            job.error = traceback.format_exc().splitlines()[-1]
    finally:
        # guarantee the GPU is released even if a stage failed (else the VM never tears down)
        if ctx.get("gpu") and not ctx.get("gpu_released"):
            ctx["gpu"].release()
            ctx["gpu_released"] = True


def render_job(job: _Job, settings: dict, *, res: int = 384) -> bytes:
    """Render the cached rig at ``settings`` to PNG bytes (live preview), on a light background."""
    from PIL import Image
    from .. import preview

    img = preview.render_pose(job.stack, job.meshes, job.params, settings, res=res)
    bg = Image.new("RGB", (res, res), (250, 250, 252))
    bg.paste(img, (0, 0), img)
    buf = io.BytesIO()
    bg.save(buf, "PNG")
    return buf.getvalue()


def preview_param_specs(job: _Job) -> list[dict]:
    """The slider-worthy params present in this rig, with ranges, in display order."""
    if not job.params:
        return []
    by_id = {p.id: p for p in job.params}
    out = []
    for pid in _PREVIEW_PARAMS:
        p = by_id.get(pid)
        if p:
            out.append({"id": p.id, "min": p.min, "max": p.max, "default": p.default})
    return out


def _ordered_parts(job: _Job):
    """(layer, mesh) pairs in draw order (bottom->top) for parts that have a mesh — the index in this
    list is the part id used by the rig-export JSON and the texture endpoint."""
    if not job.stack or not job.meshes:
        return []
    mbp = {m.part_id: m for m in job.meshes}
    out = []
    for layer in sorted(job.stack.layers, key=lambda L: L.draw_order):
        m = mbp.get(layer.id)
        if m is not None:
            out.append((layer, m))
    return out


def rig_json(job: _Job) -> dict:
    """Export the rig as browser-consumable JSON for the in-page live runtime: parts (mesh geometry,
    bottom->top) + every parameter's keyforms (per-part vertex-offset deltas, keyed by part index).
    Textures are fetched separately via /tex/{i}. Model space is y-up, normalized [0,1]."""
    parts_meta = _ordered_parts(job)
    idx_of = {layer.id: i for i, (layer, _) in enumerate(parts_meta)}
    parts = [{"verts": [[round(x, 5), round(y, 5)] for x, y in m.vertices],
              "uvs": [[round(u, 5), round(v, 5)] for u, v in m.uvs],
              "tris": [list(t) for t in m.triangles]}
             for _, m in parts_meta]
    params = []
    for p in (job.params or []):
        kfs = []
        for kf in p.keyforms:
            offs = {idx_of[pid]: [[round(dx, 5), round(dy, 5)] for dx, dy in o]
                    for pid, o in kf.mesh_offsets.items() if pid in idx_of}
            kfs.append({"value": kf.value, "offsets": offs})
        params.append({"id": p.id, "min": p.min, "max": p.max,
                       "default": p.default, "keyforms": kfs})
    phys = [{"output": r.output_param, "drivers": r.all_drivers(), "model": r.model.value}
            for r in (job.physics or [])]
    # group hierarchy (head/body) so the live runtime moves each group as one rigid unit, matching
    # the emitted .inp. Part refs are indices into `parts`; pivots are model-space (y-up); rot maps
    # param -> [axis, radians-at-extreme].
    from ..backends.nijilive.puppet import head_group_ids, _HEAD_ROT, _BODY_ROT
    dm = [(layer, m) for layer, m in parts_meta]
    hid = head_group_ids(dm)
    idx = {layer.id: i for i, (layer, _) in enumerate(parts_meta)}
    head_i = [idx[layer.id] for layer, _ in parts_meta if layer.id in hid]
    body_i = [idx[layer.id] for layer, _ in parts_meta if layer.id not in hid]

    def _pivot(members):
        pts = [v for layer, m in parts_meta if idx[layer.id] in members for v in m.vertices]
        if not pts:
            return [0.5, 0.0]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return [round((min(xs) + max(xs)) / 2, 5), round(min(ys), 5)]

    def _rotmap(m):
        return {pid: [ch.split(".")[-1], rad] for pid, (ch, rad) in m.items()}

    groups = {"head": {"parts": head_i, "pivot": _pivot(set(head_i)), "rot": _rotmap(_HEAD_ROT)},
              "body": {"parts": body_i, "pivot": _pivot(set(head_i) | set(body_i)),
                       "rot": _rotmap(_BODY_ROT)}}
    return {"nparts": len(parts), "parts": parts, "params": params, "physics": phys, "groups": groups}


def live2d_bundle_dir(job: "_Job") -> Path | None:
    """Build (once) a complete Live2D bundle (.moc3 + model3/physics3/cdi3/motion3 + real textures)
    from the job's rig and return its directory. Serves the Route-A render panel so the actual
    generated .moc3 can be loaded by a real Cubism runtime in the browser (A/B vs the .inp preview)."""
    if job.work is None or job.stack is None or job.params is None or job.meshes is None:
        return None
    out = Path(job.work) / "live2d"
    model3 = out / "model.model3.json"
    if model3.is_file():
        return out
    import json as _json
    from ..core.assemble import assemble_rig
    from ..core import motion
    from ..backends.live2d import model3 as _model3, physics3 as _physics3, cdi3 as _cdi3, motion3 as _motion3
    from ..backends.live2d.moc3_binary import write_moc3
    from ..backends.live2d.moc3_emit import build_atlas, rig_to_moc3
    rig = assemble_rig(
        name="model", source=None, stack=job.stack, meshes=job.meshes, deformers=[],
        parameters=job.params, physics=job.physics or [],
        animations=motion.generate_idle(job.params) + motion.generate_expressions(job.params),
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "textures").mkdir(exist_ok=True)
    layers = Path(job.stack.layers[0].texture_path).parent   # decomposed layer PNGs

    # single shared texture ATLAS + remapped UVs (real Live2D models use an atlas, not per-part
    # textures; standard web runtimes/pixi-live2d-display need this to render our model correctly)
    atlas_img, uv_remap = build_atlas(rig, layers)
    atlas_img.save(out / "textures" / "atlas.png")
    (out / "model.moc3").write_bytes(write_moc3(rig_to_moc3(rig, atlas_uv=uv_remap)))

    def wj(rel, doc):
        (out / rel).write_text(_json.dumps(doc, indent=2))
    physics_file = None
    if rig.physics:
        physics_file = "model.physics3.json"
        wj(physics_file, _physics3.physics3(rig))
    cdi_file = "model.cdi3.json"
    wj(cdi_file, _cdi3.cdi3(rig))
    motions = {}
    for anim in rig.animations:
        rel = f"model.{anim.name}.motion3.json"
        wj(rel, _motion3.motion3(anim))
        motions.setdefault(anim.name.capitalize(), []).append(rel)
    wj("model.model3.json", _model3.model3(
        rig, moc="model.moc3", textures=["textures/atlas.png"], physics=physics_file,
        display_info=cdi_file, motions=motions or None))
    return out


_L2D_CTYPES = {".json": "application/json", ".moc3": "application/octet-stream",
               ".png": "image/png", ".js": "text/javascript"}


# --------------------------------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------------------------------

_PAGE = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


def _make_handler():
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")  # always serve the latest UI (no stale cache)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):  # noqa: N802
            u = urlparse(self.path)
            path = u.path
            if path in ("/", "/index.html"):
                page = _PAGE.replace("__DECOMPOSE_READY__", "true" if _flat_decompose_available() else "false")
                self._send(200, page.encode(), "text/html; charset=utf-8")
                return
            if path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")   # no favicon -> silence the browser's auto-request
                return
            if path == "/cubismcore/live2dcubismcore.min.js":
                # Optional self-hosted Cubism Core (proprietary — never committed): if the operator points
                # IMAGE2LIVE2D_CUBISM_CORE_JS at a local copy, serve it so the Cubism render works offline
                # / under a strict CSP. Absent -> empty 200 stub, and the page falls back to the official CDN.
                local = os.environ.get("IMAGE2LIVE2D_CUBISM_CORE_JS")
                if local and Path(local).is_file():
                    self._send(200, Path(local).read_bytes(), "application/javascript")
                else:
                    # empty 200 stub (not 404) so the page's local-first <script> probe loads cleanly
                    # with no console error; Live2DCubismCore stays undefined -> the page uses the CDN.
                    self._send(200, b"/* no self-hosted cubism core */\n", "application/javascript")
                return
            parts = path.strip("/").split("/")
            if len(parts) >= 3 and parts[0] == "api" and parts[1] == "jobs":
                job = get_job(parts[2])
                if job is None:
                    self._json(404, {"error": "no such job"})
                    return
                sub = parts[3] if len(parts) > 3 else ""
                if sub == "":
                    self._json(200, job.public())
                    return
                if sub == "download":
                    if not job.inp_bytes:
                        self._json(409, {"error": "not ready"})
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Disposition", f'attachment; filename="{job.name}.inp"')
                    self.send_header("Content-Length", str(len(job.inp_bytes)))
                    self.end_headers()
                    self.wfile.write(job.inp_bytes)
                    return
                if sub == "source":
                    if not job.source:
                        self._json(404, {"error": "no source"})
                        return
                    self._send(200, job.source, "image/png")
                    return
                if sub == "params":
                    self._json(200, {"params": preview_param_specs(job)})
                    return
                if sub == "rig":
                    if job.params is None:
                        self._json(409, {"error": "rig not ready"})
                        return
                    self._json(200, rig_json(job))
                    return
                if sub == "live2d":  # /api/jobs/{id}/live2d/<file> — serve the Cubism bundle
                    if job.params is None:
                        self._json(409, {"error": "rig not ready"})
                        return
                    try:
                        bdir = live2d_bundle_dir(job)
                    except Exception as exc:  # noqa: BLE001
                        self._json(500, {"error": f"bundle build failed: {exc}"})
                        return
                    if bdir is None:
                        self._json(409, {"error": "rig not ready"})
                        return
                    rel = "/".join(parts[4:]) or "model.model3.json"
                    target = (bdir / rel).resolve()
                    if bdir.resolve() not in target.parents or not target.is_file():
                        self._json(404, {"error": "no such bundle file"})
                        return
                    self._send(200, target.read_bytes(),
                               _L2D_CTYPES.get(target.suffix, "application/octet-stream"))
                    return
                if sub == "tex":  # /api/jobs/{id}/tex/{i}
                    try:
                        i = int(parts[4])
                    except (ValueError, IndexError):
                        self._json(400, {"error": "bad texture index"})
                        return
                    op = _ordered_parts(job)
                    if not (0 <= i < len(op)):
                        self._json(404, {"error": "no such part"})
                        return
                    self._send(200, Path(op[i][0].texture_path).read_bytes(), "image/png")
                    return
                if sub == "render":
                    if job.params is None:
                        self._json(409, {"error": "rig not ready"})
                        return
                    q = parse_qs(u.query)
                    settings = {}
                    for p in (job.params or []):
                        if p.id in q:
                            try:
                                settings[p.id] = float(q[p.id][0])
                            except ValueError:
                                pass
                    try:
                        self._send(200, render_job(job, settings), "image/png")
                    except Exception as exc:  # noqa: BLE001
                        self._json(500, {"error": str(exc)})
                    return
            self._send(404, b"not found", "text/plain")

        def do_POST(self):  # noqa: N802
            u = urlparse(self.path)
            length = int(self.headers.get("Content-Length", "0"))
            data = self.rfile.read(length)
            qs = parse_qs(u.query)
            filename = (qs.get("name") or ["upload"])[0]
            if u.path == "/api/jobs":
                try:
                    jid = start_job(data, filename)
                    self._json(200, {"job_id": jid})
                except Exception as exc:  # noqa: BLE001
                    self._json(400, {"error": str(exc)})
                return
            if u.path == "/convert":  # legacy synchronous endpoint (kept)
                try:
                    with tempfile.TemporaryDirectory(prefix="i2l_web_") as tmp:
                        res = handle_upload(data, filename, tmp)
                    self._json(200, {
                        "ok": True, "name": res.name, "parts": res.parts, "params": res.params,
                        "physics": res.physics, "passed": res.passed, "reasons": res.reasons,
                        "inp_b64": base64.b64encode(res.inp_bytes).decode(),
                    })
                except (ValueError, FileNotFoundError, ImportError) as exc:
                    self._json(200, {"ok": False, "error": str(exc)})
                return
            self._send(404, b"not found", "text/plain")

        def log_message(self, *args):  # quiet
            pass

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the local web app (blocking). Ctrl-C to stop."""
    httpd = ThreadingHTTPServer((host, port), _make_handler())
    url, _ = _decompose_config()
    inst = os.environ.get("IMAGE2LIVE2D_GCP_INSTANCE")
    if url:
        mode = f"flat-image ON (static decompose: {url})"
    elif inst:
        mode = f"flat-image ON (auto-managed GPU: {inst} @ {os.environ.get('IMAGE2LIVE2D_GCP_ZONE','?')})"
    else:
        mode = "layered-only (.psd/.zip); set IMAGE2LIVE2D_GCP_INSTANCE for auto-managed flat images"
    print(f"image2Live2D web app -> http://{host}:{port}  [{mode}]  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()
