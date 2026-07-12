"""Local web app — upload art, watch the pipeline run, preview & download the riggable ``.inp``.

Implementation lives in :mod:`.server`; this package re-exports its public API.
"""

from __future__ import annotations

from .server import (
    get_job,
    handle_upload,
    live2d_bundle_dir,
    preview_param_specs,
    render_job,
    rig_json,
    serve,
    start_job,
    SUPPORTED,
    UploadResult,
)

__all__ = [
    "get_job",
    "handle_upload",
    "live2d_bundle_dir",
    "preview_param_specs",
    "render_job",
    "rig_json",
    "serve",
    "start_job",
    "SUPPORTED",
    "UploadResult",
]
