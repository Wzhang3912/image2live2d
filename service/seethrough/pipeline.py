"""Warm in-process See-through pipeline.

Replaces "spawn a fresh `python inference_psd.py` per request" with a long-lived worker that imports
See-through **once** and reuses it across requests. That removes, per request: Python interpreter
startup, the (heavy) See-through/torch/diffusers import, and CUDA context init. Weights stay
disk-cached and the process keeps the GPU context warm.

Honest scope note: See-through's `apply_layerdiff()` / `apply_marigold()` load their diffusion models
**internally on each call** and expose no "load once" entry point (verified from the repo). So keeping
the models **resident in VRAM** across requests needs a small patch to See-through's
`utils/inference_utils.py` to hoist pipeline construction — see `warm_models()` below. Without that
patch this worker still wins (no process/import/CUDA-init per call); with it, you also skip model
reload. The whole thing is guarded: if the in-process path raises (e.g. an API mismatch), `app.py`
falls back to the subprocess path, so the service never hard-fails.

The exact `apply_*` keyword names were read from `inference_psd.py`; treat them as best-effort and
confirm against the installed repo. `infer()` raises `WarmUnavailable` on any mismatch so the caller
can fall back.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

LAYERDIFF_REPO = os.environ.get("ST_LAYERDIFF_REPO", "layerdifforg/seethroughv0.0.2_layerdiff3d")
MARIGOLD_REPO = os.environ.get("ST_MARIGOLD_REPO", "24yearsold/seethroughv0.0.1_marigold")


class WarmUnavailable(RuntimeError):
    """Raised when the in-process path can't run (import/API mismatch). Caller should fall back."""


class WarmPipeline:
    """Imports See-through once; runs inference per image in-process (GPU-serialized)."""

    def __init__(self, see_through_dir: str | os.PathLike, *, resolution: int | None = None):
        self.dir = Path(see_through_dir)
        self.resolution = resolution
        self._iu = None  # see-through utils.inference_utils module
        self._lock = threading.Lock()  # one GPU -> serialize
        self._loaded = False

    def load(self) -> None:
        """Import See-through's inference utilities once (cheap part). Heavy model construction is
        deferred to the first `infer()` / to `warm_models()`."""
        if self._loaded:
            return
        if not self.dir.is_dir():
            raise WarmUnavailable(f"SEE_THROUGH_DIR not found: {self.dir}")
        sys.path.insert(0, str(self.dir))
        os.chdir(self.dir)  # See-through resolves assets relative to its repo root
        try:
            from utils import inference_utils as iu  # noqa: E402  (path injected above)
        except Exception as exc:  # pragma: no cover - needs the real repo
            raise WarmUnavailable(f"cannot import see-through utils.inference_utils: {exc}") from exc
        for fn in ("apply_layerdiff", "apply_marigold", "further_extr"):
            if not hasattr(iu, fn):
                raise WarmUnavailable(
                    f"see-through API mismatch: utils.inference_utils has no {fn!r} "
                    "(set INFERENCE_MODE=subprocess and report the real names)"
                )
        self._iu = iu
        self.warm_models()
        self._loaded = True

    def warm_models(self) -> None:
        """Hook to make the diffusion models **resident in VRAM** across requests.

        See-through doesn't expose load-once builders, so this is a no-op by default. To enable full
        residency, add (in your fork of see-through) factory functions that build the LayerDiff and
        Marigold pipelines and cache them as module globals, then construct them here, e.g.::

            self._layerdiff = self._iu.build_layerdiff(LAYERDIFF_REPO).to("cuda")
            self._marigold  = self._iu.build_marigold(MARIGOLD_REPO).to("cuda")

        and pass them into the apply_* calls in `infer()`. Documented in docs/DECOMPOSE_SERVICE.md.
        """
        return

    def infer(self, image_path: str | os.PathLike, out_dir: str | os.PathLike) -> Path:
        """Run See-through on one image (in-process) and return the produced ``.psd`` path."""
        if not self._loaded:
            self.load()
        iu = self._iu
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        before = set(out.rglob("*.psd"))

        kw = {"resolution": self.resolution} if self.resolution else {}
        with self._lock:
            try:
                saved = iu.apply_layerdiff(str(image_path), repo_id=LAYERDIFF_REPO,
                                           save_dir=str(out), **kw)
                saved = iu.apply_marigold(str(image_path), repo_id=MARIGOLD_REPO,
                                          save_dir=str(out), **kw) or saved
                iu.further_extr(saved, save_to_psd=True)
            except TypeError as exc:  # signature drift -> let caller fall back to subprocess
                raise WarmUnavailable(f"see-through apply_* signature mismatch: {exc}") from exc

        fresh = set(out.rglob("*.psd")) - before
        pool = fresh or set(out.rglob("*.psd"))
        pool = {p for p in pool if not p.name.endswith("_depth.psd")} or pool  # layers, not depth
        if not pool:
            raise WarmUnavailable("in-process inference produced no .psd")
        return max(pool, key=lambda p: p.stat().st_mtime)
