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
from .graph import (
    BODY,
    BODY_ROLES,
    HEAD,
    HEAD_ROLES,
    RigGraph,
    RigNode,
    analyze_structure,
    build_rig_graph,
)
from .strands import (
    HAIR_BASE_TUNING,
    HAIR_DRIVER,
    StrandSpec,
    hair_specs_from_params,
    hair_strands,
    strand_param_id,
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
    "HEAD",
    "BODY",
    "HEAD_ROLES",
    "BODY_ROLES",
    "RigNode",
    "RigGraph",
    "build_rig_graph",
    "analyze_structure",
    "HAIR_BASE_TUNING",
    "HAIR_DRIVER",
    "StrandSpec",
    "hair_strands",
    "hair_specs_from_params",
    "strand_param_id",
]
