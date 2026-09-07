"""Readiness and fixed local GPU companion API."""
import json
import os
import urllib.request


def companion(action=False):
    url = os.environ.get("IMAGE2LIVE2D_LOCAL_GPU_URL", "http://seethrough:8000")
    req = urllib.request.Request(url + "/setup/models" if action else url + "/setup",
                                 data=b"{}" if action else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3) as response:
        return json.load(response)


def readiness():
    missing = []
    for module in ("PIL", "numpy", "psd_tools"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    try:
        local = companion()
    except (OSError, ValueError):
        local = {"ready": False, "state": "unavailable", "gpu": False,
                 "message": "Local GPU container is not running. On an NVIDIA GPU machine, install the NVIDIA driver and Container Toolkit, then run ./start.sh --gpu. The first container build can take several minutes. Apple/AMD GPUs are not supported by this setup."}
    external = not os.environ.get("IMAGE2LIVE2D_LOCAL_GPU_URL") and bool(
        os.environ.get("IMAGE2LIVE2D_DECOMPOSE_URL") or os.environ.get("IMAGE2LIVE2D_GCP_INSTANCE"))
    return {"external_configured": external, "layers_ready": not missing, "missing_dependencies": missing, "local": local}
