"""Stage 5 — Physics. Procedurally generate pendulum physics for hair/cloth.

Implementation lives in :mod:`.generate`; this package re-exports its public API.
"""

from __future__ import annotations

from .generate import (
    generate_physics,
)

__all__ = [
    "generate_physics",
]
