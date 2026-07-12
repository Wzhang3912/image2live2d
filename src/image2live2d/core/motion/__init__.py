"""Stage 5b — Motion. Procedurally author a looping **idle** animation.

Implementation lives in :mod:`.idle`; this package re-exports its public API.
"""

from __future__ import annotations

from .idle import (
    FPS,
    generate_idle,
    IDLE_FRAMES,
)

__all__ = [
    "FPS",
    "generate_idle",
    "IDLE_FRAMES",
]
