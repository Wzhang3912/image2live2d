"""GCP GPU lifecycle — auto start the decompose VM for a job, auto stop when done.

So you only pay for the GPU while an image is actually being processed: the pipeline starts the
instance, waits for the service to report healthy, runs the remote decompose, then stops the instance
(even on error). Uses the ``gcloud`` CLI (so it relies on your local gcloud auth — no service-account
keys handled here).

    from image2live2d.gpu import GpuService
    with GpuService("seethrough-svc", "us-central1-a") as svc:   # starts + waits for /health
        stack = svc.decompose("hero.png", "work/")
    # instance stopped here

or the one-shot helper ``decompose_managed(...)``.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .core import decompose
from .core.decompose import RoleMapper, role_from_layer_name
from .core.types import LayerStack


class GpuService:
    """Start/stop a GCP GPU instance around remote decompose calls."""

    def __init__(
        self,
        instance: str,
        zone: str,
        *,
        port: int = 8000,
        token: str | None = None,
        project: str | None = None,
        boot_timeout: float = 480.0,
        stop_on_exit: bool = True,
        poll_interval: float = 5.0,
    ) -> None:
        self.instance = instance
        self.zone = zone
        self.port = port
        self.token = token
        self.project = project
        self.boot_timeout = boot_timeout
        self.stop_on_exit = stop_on_exit
        self.poll_interval = poll_interval
        self.base_url: str | None = None

    # -- gcloud plumbing ----------------------------------------------------------------------
    def _gcloud(self, *args: str, capture: bool = False) -> str:
        cmd = ["gcloud", "compute", *args, f"--zone={self.zone}"]
        if self.project:
            cmd.append(f"--project={self.project}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"gcloud failed: {' '.join(cmd)}\n{proc.stderr.strip()}")
        return proc.stdout.strip() if capture else ""

    def external_ip(self) -> str:
        ip = self._gcloud(
            "instances", "describe", self.instance,
            "--format=get(networkInterfaces[0].accessConfigs[0].natIP)",
            capture=True,
        )
        if not ip:
            raise RuntimeError(f"{self.instance}: no external IP (is it running / does it have one?)")
        return ip

    # -- lifecycle ----------------------------------------------------------------------------
    def start(self) -> str:
        """Start the instance, resolve its IP, and block until the service is healthy. Returns the
        base URL."""
        self._gcloud("instances", "start", self.instance)
        ip = self.external_ip()
        self.base_url = f"http://{ip}:{self.port}"
        self._wait_healthy()
        return self.base_url

    def stop(self) -> None:
        self._gcloud("instances", "stop", self.instance)

    def _wait_healthy(self) -> None:
        url = f"{self.base_url}/health"
        waited = 0.0
        while waited <= self.boot_timeout:
            try:
                with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
                    if resp.status == 200 and json.loads(resp.read()).get("ok"):
                        return
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(self.poll_interval)
            waited += self.poll_interval
        raise RuntimeError(f"{self.instance}: service not healthy at {url} after {self.boot_timeout}s")

    # -- work ---------------------------------------------------------------------------------
    def decompose(
        self,
        image_path: str | Path,
        work_dir: str | Path,
        *,
        role_map: RoleMapper = role_from_layer_name,
        timeout: float = 900.0,
    ) -> LayerStack:
        if self.base_url is None:
            raise RuntimeError("call start() (or use the context manager) before decompose()")
        return decompose.from_service(
            image_path, self.base_url, work_dir, role_map=role_map, token=self.token, timeout=timeout
        )

    def __enter__(self) -> "GpuService":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        if self.stop_on_exit:
            self.stop()


def decompose_managed(
    image_path: str | Path,
    work_dir: str | Path,
    *,
    instance: str,
    zone: str,
    port: int = 8000,
    token: str | None = None,
    project: str | None = None,
    stop_on_finish: bool = True,
    role_map: RoleMapper = role_from_layer_name,
) -> LayerStack:
    """One-shot: start the GPU VM, decompose ``image_path``, stop the VM (unless ``stop_on_finish``
    is False). Stops the instance even if decomposition raises."""
    with GpuService(instance, zone, port=port, token=token, project=project,
                    stop_on_exit=stop_on_finish) as svc:
        return svc.decompose(image_path, work_dir, role_map=role_map)
