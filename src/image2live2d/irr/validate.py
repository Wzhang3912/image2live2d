"""Soft validation / linting for a ``Rig``.

The schema's own ``model_validator`` enforces *structural integrity* (hard errors, raised on
construction). This module adds *quality lint* — non-fatal warnings that flag rigs which are valid
but likely to animate poorly. The QA harness (Phase 2+) runs these and tracks them per phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .params import STANDARD_PARAM_IDS
from .schema import Rig, SemanticRole

# Roles a believable face rig is expected to contain.
_EXPECTED_FACE_ROLES = {
    SemanticRole.face_base,
    SemanticRole.mouth,
}

# Each eye side is satisfied by EITHER its lash part OR its eye-white: the white alone is riggable
# (blink collapses it; pupil/look use the pupil part), so a missing eyelash shouldn't fail the gate.
# See-through commonly emits one combined eyelash -> only eye_l, while splitting the whites L/R.
_EYE_SIDES = {
    "eye (left)": (SemanticRole.eye_l, SemanticRole.eye_white_l),
    "eye (right)": (SemanticRole.eye_r, SemanticRole.eye_white_r),
}

# Parameters that should have keyforms for the face to move at all.
_MOVEMENT_PARAMS = {
    "ParamAngleX",
    "ParamAngleY",
    "ParamEyeLOpen",
    "ParamEyeROpen",
    "ParamMouthOpenY",
}


class Severity(str, Enum):
    warning = "warning"
    info = "info"


@dataclass(frozen=True)
class Issue:
    severity: Severity
    code: str
    message: str


def lint(rig: Rig) -> list[Issue]:
    """Return a list of non-fatal quality issues. Empty list == clean."""
    issues: list[Issue] = []
    roles = {p.semantic_role for p in rig.parts}
    param_ids = rig.parameter_ids()
    params_with_keyforms = {p.id for p in rig.parameters if p.keyforms}

    for role in _EXPECTED_FACE_ROLES:
        if role not in roles:
            issues.append(
                Issue(Severity.warning, "missing_role", f"no part with role {role.value!r}")
            )

    for label, side_roles in _EYE_SIDES.items():
        if not (set(side_roles) & roles):
            issues.append(
                Issue(Severity.warning, "missing_role", f"no part for {label} (lash or eye-white)")
            )

    for pid in _MOVEMENT_PARAMS:
        if pid not in param_ids:
            issues.append(Issue(Severity.warning, "missing_param", f"missing parameter {pid!r}"))
        elif pid not in params_with_keyforms:
            issues.append(
                Issue(Severity.warning, "no_keyforms", f"parameter {pid!r} has no keyforms (won't move)")
            )

    for part in rig.parts:
        if rig.mesh_for(part.id) is None:
            issues.append(
                Issue(Severity.warning, "no_mesh", f"part {part.id!r} has no mesh (won't render/deform)")
            )

    # Draw order should be unambiguous.
    orders = [p.draw_order for p in rig.parts]
    if len(set(orders)) != len(orders):
        issues.append(Issue(Severity.info, "draw_order_ties", "parts share draw_order values"))

    for p in rig.parameters:
        if p.id not in STANDARD_PARAM_IDS and p.id not in rig.parameter_ids():
            continue  # placeholder for future custom-param policy
    return issues


def format_issues(issues: list[Issue]) -> str:
    if not issues:
        return "OK — no lint issues"
    return "\n".join(f"[{i.severity.value}] {i.code}: {i.message}" for i in issues)
