"""Standard parameter catalog.

We adopt **Live2D's standard parameter IDs** verbatim in the IRR. This is a deliberate design
choice: motion clips (``.motion3.json``), ARKit face-tracking mappings, and TTS lip-sync all key
off these ids, so the *same* animation data drives both the nijilive (Route B) and Live2D
(Route A) backends without translation.

Ranges follow Live2D's conventions so exported models feel native in the ecosystem.
"""

from __future__ import annotations

import re

from .schema import Parameter

# (id, min, max, default)
_PARAM_SPECS: list[tuple[str, float, float, float]] = [
    # Head
    ("ParamAngleX", -30.0, 30.0, 0.0),
    ("ParamAngleY", -30.0, 30.0, 0.0),
    ("ParamAngleZ", -30.0, 30.0, 0.0),
    # Eyes
    ("ParamEyeLOpen", 0.0, 1.0, 1.0),
    ("ParamEyeROpen", 0.0, 1.0, 1.0),
    ("ParamEyeBallX", -1.0, 1.0, 0.0),
    ("ParamEyeBallY", -1.0, 1.0, 0.0),
    # Brows
    ("ParamBrowLY", -1.0, 1.0, 0.0),
    ("ParamBrowRY", -1.0, 1.0, 0.0),
    # Mouth
    ("ParamMouthForm", -1.0, 1.0, 0.0),
    ("ParamMouthOpenY", 0.0, 1.0, 0.0),
    # Body
    ("ParamBodyAngleX", -10.0, 10.0, 0.0),
    ("ParamBodyAngleY", -10.0, 10.0, 0.0),
    ("ParamBodyAngleZ", -10.0, 10.0, 0.0),
    ("ParamBreath", 0.0, 1.0, 0.0),
]

# Physics outputs (driven by the physics rig, not directly by an animator).
_PHYSICS_PARAM_SPECS: list[tuple[str, float, float, float]] = [
    ("ParamHairFront", -1.0, 1.0, 0.0),
    ("ParamHairSide", -1.0, 1.0, 0.0),
    ("ParamHairBack", -1.0, 1.0, 0.0),
    # Cloth/skirt hem sway, split into left/center/right zones so the hem ripples like cloth and each
    # zone reacts to the nearest lower-body motion (the near leg + body sway). Non-standard ids.
    ("ParamSkirtL", -1.0, 1.0, 0.0),
    ("ParamSkirtC", -1.0, 1.0, 0.0),
    ("ParamSkirtR", -1.0, 1.0, 0.0),
]

# Limb articulation (Phase 3). NOTE: Live2D has **no canonical arm/leg parameter ids** — these are
# our own conventions for procedural limb rotation about a shoulder/hip joint. They will NOT be
# driven by stock motion clips or ARKit mappings (which only key the head/eye/mouth/body params
# above); animators/motions must target them explicitly. Documented in docs/PHASE3_PLAN.md.
_LIMB_PARAM_SPECS: list[tuple[str, float, float, float]] = [
    ("ParamArmLA", -10.0, 10.0, 0.0),   # whole-arm swing about the shoulder
    ("ParamArmRA", -10.0, 10.0, 0.0),
    ("ParamLegLA", -10.0, 10.0, 0.0),   # whole-leg swing about the hip
    ("ParamLegRA", -10.0, 10.0, 0.0),
    ("ParamArmLB", -10.0, 10.0, 0.0),   # forearm bend about the elbow (lower segment only)
    ("ParamArmRB", -10.0, 10.0, 0.0),
    ("ParamLegLB", -10.0, 10.0, 0.0),   # lower-leg bend about the knee
    ("ParamLegRB", -10.0, 10.0, 0.0),
]

_ALL = {spec[0]: spec for spec in (*_PARAM_SPECS, *_PHYSICS_PARAM_SPECS, *_LIMB_PARAM_SPECS)}

# Public id constants (importable, autocomplete-friendly).
STANDARD_PARAM_IDS: tuple[str, ...] = tuple(s[0] for s in _PARAM_SPECS)
PHYSICS_PARAM_IDS: tuple[str, ...] = tuple(s[0] for s in _PHYSICS_PARAM_SPECS)
LIMB_PARAM_IDS: tuple[str, ...] = tuple(s[0] for s in _LIMB_PARAM_SPECS)


# Extra hair strands (P2) mint suffixed ids off a base physics param — e.g. a second side-tail is
# ``ParamHairSide2``. They share the base's [-1, 1] range; the base ids stay in the catalog above.
_HAIR_STRAND_RE = re.compile(r"^ParamHair(?:Front|Side|Back)\d+$")


def make_parameter(param_id: str) -> Parameter:
    """Create an empty (keyform-less) ``Parameter`` for a known standard id, or a suffixed hair-strand
    id (``ParamHairSide2`` …) which mints a physics-output param with the standard [-1, 1] range."""
    if param_id in _ALL:
        _id, lo, hi, default = _ALL[param_id]
        return Parameter(id=_id, min=lo, max=hi, default=default)
    if _HAIR_STRAND_RE.match(param_id):
        return Parameter(id=param_id, min=-1.0, max=1.0, default=0.0)
    raise KeyError(f"unknown standard parameter id {param_id!r}")


def standard_parameters(include_physics: bool = True) -> list[Parameter]:
    """Return the full standard parameter set as empty ``Parameter`` objects, ready for the rig
    authoring stage to populate with keyforms."""
    ids = list(STANDARD_PARAM_IDS) + (list(PHYSICS_PARAM_IDS) if include_physics else [])
    return [make_parameter(i) for i in ids]
