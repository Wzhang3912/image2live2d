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
    analyze_meshes,
    analyze_stack,
    mesh_probes,
    score_dynamics,
)
from .appendages import (
    AppendageSpec,
    accessory_appendages,
    garment_appendages,
)
from .calibrate import (
    Metrics,
    SweepPoint,
    best_thresholds,
    evaluate,
    physics3_output_params,
    pool_metrics,
    predicted_physics,
    roc_auc,
    sweep,
)
from .graph import (
    ARM_L,
    ARM_R,
    BODY,
    BODY_ROLES,
    HEAD,
    HEAD_ROLES,
    RigGraph,
    RigNode,
    analyze_structure,
    build_rig_graph,
)
from .skirt import (
    SKIRT_PARAM_IDS,
    ZoneSpec,
    material_from_geometry,
    skirt_cloth,
    skirt_specs_from_params,
    skirt_zones,
)
from .strands import (
    HAIR_BASE_TUNING,
    HAIR_DRIVER,
    StrandSpec,
    hair_specs_from_params,
    hair_strands,
    mesh_components,
    strand_param_id,
)
from .limbs import (
    split_bundled_pairs,
    split_fused_legs,
)
from .zorder import (
    BROWS,
    FACE_FEATURES,
    normalize_face_zorder,
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
    "analyze_meshes",
    "mesh_probes",
    "HEAD",
    "BODY",
    "ARM_L",
    "ARM_R",
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
    "mesh_components",
    "strand_param_id",
    "SKIRT_PARAM_IDS",
    "ZoneSpec",
    "material_from_geometry",
    "skirt_cloth",
    "skirt_zones",
    "skirt_specs_from_params",
    "AppendageSpec",
    "accessory_appendages",
    "garment_appendages",
    "Metrics",
    "SweepPoint",
    "physics3_output_params",
    "predicted_physics",
    "evaluate",
    "sweep",
    "best_thresholds",
    "pool_metrics",
    "roc_auc",
    "FACE_FEATURES",
    "BROWS",
    "normalize_face_zorder",
    "split_bundled_pairs",
    "split_fused_legs",
]
