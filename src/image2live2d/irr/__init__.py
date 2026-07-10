"""Intermediate Rig Representation (IRR) — the format-neutral keystone of the pipeline."""

from .params import (
    PHYSICS_PARAM_IDS,
    STANDARD_PARAM_IDS,
    make_parameter,
    standard_parameters,
)
from .schema import (
    IRR_VERSION,
    Deformer,
    DeformerType,
    Keyform,
    Mesh,
    Meta,
    Parameter,
    Part,
    PhysicsRig,
    Rig,
    SemanticRole,
    Texture,
    Vec2,
)
from .validate import Issue, Severity, lint

__all__ = [
    "IRR_VERSION",
    "Vec2",
    "Texture",
    "Mesh",
    "Part",
    "Deformer",
    "DeformerType",
    "Keyform",
    "Parameter",
    "PhysicsRig",
    "Meta",
    "Rig",
    "SemanticRole",
    "STANDARD_PARAM_IDS",
    "PHYSICS_PARAM_IDS",
    "make_parameter",
    "standard_parameters",
    "Issue",
    "Severity",
    "lint",
]
