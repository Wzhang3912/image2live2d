"""Stage 4 — Rig authoring. The hard part: turn layers+meshes into deformers + parameters.

Implementation lives in :mod:`.author`; this package re-exports its public API.
"""

from __future__ import annotations

from .author import (
    author_rig,
    detect_landmarks,
    RigAuthoring,
    select_template,
    Template,
)

__all__ = [
    "author_rig",
    "detect_landmarks",
    "RigAuthoring",
    "select_template",
    "Template",
]
