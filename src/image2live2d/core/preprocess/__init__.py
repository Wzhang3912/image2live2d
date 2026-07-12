"""Stage 1 — Preprocess. Background removal, normalization, character crop.

Implementation lives in :mod:`.prepare`; this package re-exports its public API.
"""

from __future__ import annotations

from .prepare import (
    prepare,
)

__all__ = [
    "prepare",
]
