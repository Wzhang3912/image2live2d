"""Stage 4b — Structure. The dynamics-score detector: decide *which parts need physics* and how.

Public API is re-exported here; the implementation lives in :mod:`.dynamics`. See that module (and
docs/AUTORIG_PHYSICS_UNIVERSAL_PLAN.md) for the design.
"""

from __future__ import annotations

from .dynamics import (
    DEFAULT_ALPHA_THRESHOLD,
    DEFAULT_SAMPLES,
    AlphaSampler,
    DynamicsVerdict,
    PartDynamics,
    PartProbe,
    PhysicalClass,
    analyze_stack,
    score_dynamics,
)

__all__ = [
    "AlphaSampler",
    "DEFAULT_SAMPLES",
    "DEFAULT_ALPHA_THRESHOLD",
    "DynamicsVerdict",
    "PhysicalClass",
    "PartProbe",
    "PartDynamics",
    "score_dynamics",
    "analyze_stack",
]
