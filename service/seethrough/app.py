"""See-through decompose service — HTTP wrapper around See-through's PSD inference.

POST an image (raw bytes) to ``/decompose`` and get back a layered ``.psd`` (See-through's native
output), which the image2live2d client (``decompose.from_service``) turns into a LayerStack. Runs on a
GPU box (e.g. a GCP L4/T4 VM — see docs/DECOMPOSE_SERVICE.md).

Two inference paths (``INFERENCE_MODE`` env):
- ``warm`` (default): a long-lived in-process worker (``pipeline.WarmPipeline``) imports See-through
  once and reuses it — no per-request Python/import/CUDA-init cost. Falls back to subprocess if the
  in-process API doesn't match (so the service never hard-fails).
- ``subprocess``: spawn ``inference/scripts/inference_psd.py`` per request (always works; slowest).

Single GPU => requests are serialized. First request loads weights (~10–15 GB disk cache) and is slow.
Env: SEE_THROUGH_DIR, SEE_THROUGH_PYTHON, SEE_THROUGH_SCRIPT, SEE_THROUGH_OUTPUT, SEE_THROUGH_TOKEN,
     RESOLUTION, INFERENCE_MODE, SEE_THROUGH_TIMEOUT.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from pipeline import WarmPipeline, WarmUnavailable

log = logging.getLogger("seethrough")
logging.basicConfig(level=logging.INFO)

SEE_THROUGH_DIR = Path(os.environ.get("SEE_THROUGH_DIR", "/opt/see-through"))
PYTHON = os.environ.get("SEE_THROUGH_PYTHON", "python")
SCRIPT = os.environ.get("SEE_THROUGH_SCRIPT", "inference/scripts/inference_psd.py")
OUTPUT_DIR = Path(os.environ.get("SEE_THROUGH_OUTPUT", SEE_THROUGH_DIR / "workspace/layerdiff_output"))
TOKEN = os.environ.get("SEE_THROUGH_TOKEN")
RESOLUTION = os.environ.get("RESOLUTION")
TIMEOUT = int(os.environ.get("SEE_THROUGH_TIMEOUT", "900"))
# Default to the known-good subprocess path. Warm in-process is opt-in (INFERENCE_MODE=warm) until
# See-through's apply_* API is confirmed on the box — its real signature is positional and our warm
# wrapper is best-effort, so leaving it default would silently fall back per request.
INFERENCE_MODE = os.environ.get("INFERENCE_MODE", "subprocess").lower()

app = FastAPI(title="see-through decompose service", version="0.3.0")
_subproc_lock = threading.Lock()  # one GPU -> serialize the subprocess path
_warm: WarmPipeline | None = None

# Async job registry: inference takes minutes, so /decompose returns a job id immediately and the
# client polls — no single HTTP connection is held open across the whole inference (which a NAT/idle
# timeout would drop). job_id -> {"status": running|done|error, "psd": path|None, "error": str|None}.
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


@app.on_event("startup")
def _startup() -> None:
    global _warm
    if INFERENCE_MODE == "subprocess":
        log.info("INFERENCE_MODE=subprocess (warm pipeline disabled)")
        return
    try:
        wp = WarmPipeline(SEE_THROUGH_DIR, resolution=int(RESOLUTION) if RESOLUTION else None)
        wp.load()
        _warm = wp
        log.info("warm in-process pipeline loaded")
    except WarmUnavailable as exc:
        log.warning("warm pipeline unavailable (%s) — falling back to subprocess", exc)
        _warm = None


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "mode": "warm" if _warm else "subprocess",
        "see_through_dir": str(SEE_THROUGH_DIR),
        "exists": SEE_THROUGH_DIR.is_dir(),
    }


@app.post("/decompose", status_code=202)
async def decompose(request: Request) -> JSONResponse:
    """Start an async decompose job; returns {"job_id"} immediately. Poll GET /jobs/{id}."""
    if TOKEN and request.headers.get("X-Auth-Token") != TOKEN:
        raise HTTPException(status_code=401, detail="missing/invalid X-Auth-Token")
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="empty request body (POST the image bytes)")

    job_id = uuid.uuid4().hex
    work = Path(tempfile.mkdtemp(prefix="st_job_"))
    src = work / "input.png"
    src.write_bytes(data)
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "psd": None, "error": None, "started": time.time()}
    threading.Thread(target=_worker, args=(job_id, src, work), daemon=True).start()
    return JSONResponse({"job_id": job_id}, status_code=202)


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job id")
    secs = round(time.time() - job["started"], 1)
    return {"status": job["status"], "error": job["error"], "elapsed_seconds": secs,
            "mode": "warm" if _warm else "subprocess"}


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str) -> Response:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job id")
    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=job["error"] or "inference failed")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail="job not finished")
    psd = Path(job["psd"])
    return Response(
        content=psd.read_bytes(),
        media_type="image/vnd.adobe.photoshop",
        headers={"Content-Disposition": f'attachment; filename="{psd.name}"'},
    )


def _worker(job_id: str, src: Path, work: Path) -> None:
    """Run inference in a background thread, recording the result/error on the job."""
    try:
        psd = _infer(src, work / "out")
        with _jobs_lock:
            _jobs[job_id].update(status="done", psd=str(psd))
    except Exception as exc:  # noqa: BLE001 - report any failure to the client via the job
        log.warning("job %s failed: %s", job_id, exc)
        with _jobs_lock:
            _jobs[job_id].update(status="error", error=str(exc))


def _infer(src: Path, out_dir: Path) -> Path:
    """Warm in-process inference if available, else the subprocess CLI. Returns the layered .psd."""
    if _warm is not None:
        try:
            return _warm.infer(src, out_dir)
        except WarmUnavailable as exc:
            log.warning("warm infer unavailable (%s) — using subprocess", exc)
    return _subprocess_infer(src)


def _subprocess_infer(src: Path) -> Path:
    """Run See-through's CLI in a fresh process; return the newest layered .psd (raises on failure)."""
    cmd = [PYTHON, SCRIPT, "--srcp", str(src), "--save_to_psd"]
    if RESOLUTION:
        cmd += ["--resolution", RESOLUTION]
    with _subproc_lock:
        before = set(OUTPUT_DIR.rglob("*.psd")) if OUTPUT_DIR.exists() else set()
        proc = subprocess.run(
            cmd, cwd=str(SEE_THROUGH_DIR), capture_output=True, text=True, timeout=TIMEOUT
        )
        if proc.returncode != 0:
            tail = (proc.stderr or "")[-1500:] + "\n--stdout--\n" + (proc.stdout or "")[-1500:]
            raise RuntimeError(f"see-through failed (rc={proc.returncode}):\n{tail}")
        after = set(OUTPUT_DIR.rglob("*.psd")) if OUTPUT_DIR.exists() else set()
    pool = (after - before) or after
    # See-through writes both a layered "input.psd" AND a "input_depth.psd"; we want the layers one.
    pool = {p for p in pool if not p.name.endswith("_depth.psd")} or pool
    if not pool:
        raise RuntimeError("see-through produced no .psd")
    return max(pool, key=lambda p: p.stat().st_mtime)
