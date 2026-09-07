"""Explicit, resumable local model installation; no downloads during app startup."""
import json
import os
import shutil
import threading
from pathlib import Path

REPOS = {"layerdiff": "layerdifforg/seethroughv0.0.2_layerdiff3d",
         "depth": "24yearsold/seethroughv0.0.1_marigold"}


class ModelManager:
    def __init__(self, root=None):
        self.root = Path(root or os.environ.get("MODEL_DIR", "/models"))
        self.lock = threading.Lock()
        self.state = "idle"
        self.message = "Download the See-through models to enable flat images."

    def paths(self):
        try:
            manifest = json.loads((self.root / "installed.json").read_text())
            paths = {}
            for name in REPOS:
                entry = manifest[name]
                path = self.root / name
                if not entry["files"] or not all((path / f).is_file() for f in entry["files"]):
                    return {}
                paths[name] = str(path)
            return paths
        except (OSError, ValueError, KeyError, TypeError):
            return {}

    def status(self):
        import torch
        gpu = torch.cuda.is_available()
        ready = bool(self.paths())
        return {"gpu": gpu, "gpu_name": torch.cuda.get_device_name(0) if gpu else None,
                "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if gpu else None,
                "ready": gpu and ready, "models_installed": ready,
                "state": "ready" if ready and self.state == "idle" else self.state,
                "message": "Models installed. Ready for your first conversion." if ready and self.state == "idle" else self.message,
                "free_gb": round(shutil.disk_usage(self.root if self.root.exists() else self.root.parent).free / 2**30, 1)}

    def install(self):
        if self.paths():
            return
        if not self.lock.acquire(blocking=False):
            return
        self.state = "downloading"
        threading.Thread(target=self._download, daemon=True).start()

    def _download(self):
        try:
            from huggingface_hub import HfApi, snapshot_download
            self.root.mkdir(parents=True, exist_ok=True)
            manifest = {}
            for name, repo in REPOS.items():
                self.message = f"Downloading {name}; interrupted downloads resume when retried."
                info = HfApi().model_info(repo, files_metadata=True)
                revision = info.sha
                required = sum(max(0, (file.size or 0) - (
                    (self.root / name / file.rfilename).stat().st_size
                    if (self.root / name / file.rfilename).is_file() else 0))
                    for file in (getattr(info, "siblings", None) or []))
                if required + 2**30 > shutil.disk_usage(self.root).free:
                    raise RuntimeError(f"Free {required / 2**30 + 1:.1f} GiB for {name} and retry.")
                path = Path(snapshot_download(repo, revision=revision, local_dir=self.root / name))
                files = [str(p.relative_to(path)) for p in path.rglob('*')
                         if p.is_file() and '.cache' not in p.parts]
                if not files or not (path / 'model_index.json').is_file():
                    raise RuntimeError(f"Incomplete {name} model download.")
                manifest[name] = {"revision": revision, "files": files}
            target = self.root / "installed.tmp"
            target.write_text(json.dumps(manifest))
            target.replace(self.root / "installed.json")
            self.state, self.message = "ready", "Models installed. Upload an image to run your first GPU conversion."
        except Exception as exc:
            self.state = "error"
            self.message = str(exc) if isinstance(exc, RuntimeError) else f"Model download failed: {type(exc).__name__}. Check container logs and retry."
            import logging
            logging.getLogger(__name__).exception("Model installation failed")
        finally:
            self.lock.release()
