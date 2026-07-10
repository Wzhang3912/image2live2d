"""Shared Live2D (Route A) mapping helpers — parameter metadata, groups, hit areas.

Coordinate note: Cubism's canvas is **y-up with a centered origin**, the same handedness as the IRR
(which kept y-up precisely for this), so Route A needs **no y-flip** (contrast the nijilive emitter).
Geometry itself lives in the closed ``.moc3``; everything in this package is *geometry-free*
parameter/part metadata, which is why it's fully writable headless.

Parameters are grouped for the Cubism display-info (``.cdi3``) folders and for the ``.model3``
``Groups`` (EyeBlink / LipSync), all keyed off our standard Live2D parameter ids.
"""

from __future__ import annotations

from ...irr.schema import Rig, SemanticRole

# (param id, display name, cdi group) — drives both display names and cdi3 grouping.
_PARAM_META: list[tuple[str, str, str]] = [
    ("ParamAngleX", "Angle X", "Head"),
    ("ParamAngleY", "Angle Y", "Head"),
    ("ParamAngleZ", "Angle Z", "Head"),
    ("ParamEyeLOpen", "Eye Open L", "Eyes"),
    ("ParamEyeROpen", "Eye Open R", "Eyes"),
    ("ParamEyeBallX", "Eyeball X", "Eyes"),
    ("ParamEyeBallY", "Eyeball Y", "Eyes"),
    ("ParamBrowLY", "Brow L Y", "Eyes"),
    ("ParamBrowRY", "Brow R Y", "Eyes"),
    ("ParamMouthForm", "Mouth Form", "Mouth"),
    ("ParamMouthOpenY", "Mouth Open", "Mouth"),
    ("ParamBodyAngleX", "Body Angle X", "Body"),
    ("ParamBodyAngleY", "Body Angle Y", "Body"),
    ("ParamBodyAngleZ", "Body Angle Z", "Body"),
    ("ParamBreath", "Breath", "Body"),
    ("ParamHairFront", "Hair Front", "Physics"),
    ("ParamHairSide", "Hair Side", "Physics"),
    ("ParamHairBack", "Hair Back", "Physics"),
    ("ParamSkirt", "Skirt", "Physics"),
    ("ParamArmLA", "Arm L", "Limbs"),
    ("ParamArmRA", "Arm R", "Limbs"),
    ("ParamLegLA", "Leg L", "Limbs"),
    ("ParamLegRA", "Leg R", "Limbs"),
]

_DISPLAY = {pid: name for pid, name, _ in _PARAM_META}
_GROUP = {pid: group for pid, _, group in _PARAM_META}
# Group display order (only groups actually used by a rig are emitted).
_GROUP_ORDER = ["Head", "Eyes", "Mouth", "Body", "Physics", "Limbs"]

EYE_BLINK_PARAMS = ("ParamEyeLOpen", "ParamEyeROpen")
LIP_SYNC_PARAMS = ("ParamMouthOpenY",)

# Semantic roles that get a clickable hit area, and the Cubism hit-area name.
_HIT_ROLE_NAMES: dict[SemanticRole, str] = {
    SemanticRole.face_base: "Head",
    SemanticRole.torso: "Body",
}


def display_name(param_id: str) -> str:
    """Human-friendly display name for a parameter (falls back to the id)."""
    return _DISPLAY.get(param_id, param_id)


def group_id(param_id: str) -> str:
    """cdi3 group id for a parameter ("" if uncategorized)."""
    return _GROUP.get(param_id, "")


def used_groups(rig: Rig) -> list[str]:
    """The cdi3 parameter groups actually used by this rig, in display order."""
    present = {group_id(p.id) for p in rig.parameters}
    return [g for g in _GROUP_ORDER if g in present]


def part_display_name(role: SemanticRole) -> str:
    """Title-cased display name for a part from its semantic role (e.g. ``hair_front`` -> Hair Front)."""
    return role.value.replace("_", " ").title()


def model_groups(rig: Rig) -> list[dict]:
    """``.model3`` ``Groups`` — EyeBlink / LipSync parameter groups for the params present."""
    ids = rig.parameter_ids()
    groups: list[dict] = []
    eye = [p for p in EYE_BLINK_PARAMS if p in ids]
    if eye:
        groups.append({"Target": "Parameter", "Name": "EyeBlink", "Ids": eye})
    lip = [p for p in LIP_SYNC_PARAMS if p in ids]
    if lip:
        groups.append({"Target": "Parameter", "Name": "LipSync", "Ids": lip})
    return groups


def hit_areas(rig: Rig) -> list[dict]:
    """``.model3`` ``HitAreas`` — one per hit-eligible part (Head from the face, Body from the torso)."""
    out: list[dict] = []
    for part in rig.parts:
        name = _HIT_ROLE_NAMES.get(part.semantic_role)
        if name:
            out.append({"Id": part.id, "Name": name})
    return out
