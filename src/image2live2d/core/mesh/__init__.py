"""Stage 3 — Mesh. Build a deformation mesh per layer.

Implementation lives in :mod:`.build`; this package re-exports its public API.
"""

from __future__ import annotations

from .build import (
    alpha_bbox,
    AlphaSampler,
    build_mesh,
    build_meshes,
    DEFAULT_ALPHA_THRESHOLD,
    DEFAULT_CELL_SAMPLES,
    DEFAULT_GRID,
    FULL_UV,
    grid_mesh,
    UvRect,
)

__all__ = [
    "alpha_bbox",
    "AlphaSampler",
    "build_mesh",
    "build_meshes",
    "DEFAULT_ALPHA_THRESHOLD",
    "DEFAULT_CELL_SAMPLES",
    "DEFAULT_GRID",
    "FULL_UV",
    "grid_mesh",
    "UvRect",
]
