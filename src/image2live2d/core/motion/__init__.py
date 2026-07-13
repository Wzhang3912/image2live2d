"""Stage 5b — Motion. Procedurally author a looping **idle** animation + a basic **expression sheet**.

Implementation lives in :mod:`.idle` (idle loop) and :mod:`.expressions` (smile/surprise/sad/angry);
this package re-exports their public API.
"""

from __future__ import annotations

from .expressions import EXPRESSION_NAMES, generate_expressions
from .idle import (
    FPS,
    generate_idle,
    IDLE_FRAMES,
)

__all__ = [
    "FPS",
    "generate_idle",
    "IDLE_FRAMES",
    "generate_expressions",
    "EXPRESSION_NAMES",
]
