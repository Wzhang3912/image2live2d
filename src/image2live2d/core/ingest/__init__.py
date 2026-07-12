"""Stage 0 — Ingest. Acquire a source image (user upload, URL, or generation).

Implementation lives in :mod:`.load`; this package re-exports its public API.
"""

from __future__ import annotations

from .load import (
    generate_image,
    load_image,
)

__all__ = [
    "generate_image",
    "load_image",
]
