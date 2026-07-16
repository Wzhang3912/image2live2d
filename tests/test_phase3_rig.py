"""Phase 3 #14/#16 — landmark-corrected face solver + limb articulation."""

from __future__ import annotations

from pathlib import Path

import pytest

from image2live2d.core.landmark import EyeLandmarks, Landmarks, MouthLandmarks
from image2live2d.core.mesh import grid_mesh
from image2live2d.core.rig import author_rig, select_template
from image2live2d.core.rig.author import _BLINK
from image2live2d.core.types import Layer, LayerStack
from image2live2d.irr.schema import SemanticRole as R


def _stack_and_meshes(parts):
    """parts: list of (id, role, rect). Full-alpha grid mesh per part."""
    layers, meshes = [], []
    for i, (pid, role, rect) in enumerate(parts):
        layers.append(Layer(id=pid, semantic_role=role, texture_path=Path(f"{pid}.png"),
                            draw_order=i * 10, width=64, height=64))
        meshes.append(grid_mesh(pid, rect, lambda u, v: 255, grid=2))
    stack = LayerStack(layers=layers, canvas_width=64, canvas_height=64)
    return stack, meshes


def _param(params, pid):
    return next((p for p in params if p.id == pid), None)


def _kf(param, value):
    return next(k for k in param.keyforms if k.value == value)


# --------------------------------------------------------------------------------------------------
# Runaway caps (edge-case hardening): head-turn bounded + pivot-anchored; mouth-open bounded
# --------------------------------------------------------------------------------------------------
def _max_disp(param):
    return max((abs(complex(dx, dy)) for kf in param.keyforms
                for offs in kf.mesh_offsets.values() for dx, dy in offs), default=0.0)


def test_head_turn_bounded_and_pivot_anchored_on_huge_silhouette():
    """A head group inflated by floor-length hair must not fling on turn: the warp is uniform-scaled
    to <= _TURN_CAP and the pivot is anchored (the face barely translates). Regression for the edge
    cases where the head slid off the body on look L/R."""
    from image2live2d.core.rig.author import _TURN_CAP
    parts = [("face_base", R.face_base, (0.42, 0.78, 0.58, 0.95)),   # small face up top
             ("hair_back", R.hair_back, (0.30, 0.02, 0.70, 0.97))]    # huge hair -> inflated bbox
    stack, meshes = _stack_and_meshes(parts)
    auth = author_rig(stack, meshes, select_template(stack))
    for pid in ("ParamAngleX", "ParamAngleY"):
        assert _max_disp(_param(auth.parameters, pid)) <= _TURN_CAP + 1e-9


def test_mouth_open_bounded_on_tall_mesh():
    """Mouth-open drop scales with mesh height; an oversized See-through mouth layer must still be
    capped to _MOUTH_CAP rather than dropping a vertex across the canvas."""
    from image2live2d.core.rig.author import _MOUTH_CAP
    parts = [("face_base", R.face_base, (0.2, 0.5, 0.8, 0.95)),
             ("mouth", R.mouth, (0.35, 0.20, 0.65, 0.55))]            # absurdly tall mouth layer
    stack, meshes = _stack_and_meshes(parts)
    auth = author_rig(stack, meshes, select_template(stack))          # no landmarks -> bbox drop path
    assert _max_disp(_param(auth.parameters, "ParamMouthOpenY")) <= _MOUTH_CAP + 1e-9


# #14 — blink collapses toward the landmark lid axis, not each part's own midline
# --------------------------------------------------------------------------------------------------
def _nonzero(param):
    return any(any(dx or dy for dx, dy in kf.mesh_offsets.get(pid, []))
               for kf in param.keyforms for pid in kf.mesh_offsets)


def test_degenerate_landmarks_fall_back_so_pupils_and_mouth_still_move():
    """A collapsed (~0 extent) eye/mouth landmark must not produce a dead rig: pupil-look and
    mouth-open fall back to the part's own bbox. Regression — the all-params audit found pupils
    (eye width 0 -> travel 0) and mouth (height 0 -> drop scaled to 0) dead on real characters."""
    parts = [("face_base", R.face_base, (0.20, 0.50, 0.80, 0.95)),
             ("pupil_l", R.pupil_l, (0.36, 0.62, 0.44, 0.68)),
             ("pupil_r", R.pupil_r, (0.56, 0.62, 0.64, 0.68)),
             ("mouth", R.mouth, (0.44, 0.55, 0.56, 0.60))]
    stack, meshes = _stack_and_meshes(parts)
    p = (0.5, 0.65)
    mo = (0.5, 0.57)
    degen = Landmarks(
        eye_l=EyeLandmarks(center=p, lid_top=p, lid_bottom=p, inner=p, outer=p, pupil=p),
        eye_r=EyeLandmarks(center=p, lid_top=p, lid_bottom=p, inner=p, outer=p, pupil=p),
        mouth=MouthLandmarks(center=mo, left_corner=mo, right_corner=mo, top=mo, bottom=mo),
    )
    assert degen.eye_l.width == 0.0 and degen.mouth.height == 0.0  # truly degenerate
    auth = author_rig(stack, meshes, select_template(stack), landmarks=degen)
    assert _nonzero(_param(auth.parameters, "ParamEyeBallX"))    # pupils still look
    assert _nonzero(_param(auth.parameters, "ParamMouthOpenY"))  # mouth still opens


def test_blink_uses_landmark_lid_axis():
    # pupil bbox midline (0.64) differs from the eye lid axis (0.65)
    parts = [("eye_l", R.eye_l, (0.30, 0.60, 0.50, 0.70)),
             ("pupil_l", R.pupil_l, (0.38, 0.62, 0.42, 0.66))]
    stack, meshes = _stack_and_meshes(parts)

    plain = author_rig(stack, meshes, select_template(stack)).parameters
    lm = Landmarks(eye_l=EyeLandmarks(center=(0.40, 0.65), lid_top=(0.40, 0.70),
                                      lid_bottom=(0.40, 0.60), inner=(0.30, 0.65),
                                      outer=(0.50, 0.65), pupil=(0.40, 0.64)))
    corrected = author_rig(stack, meshes, select_template(stack), landmarks=lm).parameters

    # closed keyform (value 0) pupil offset at a vertex y=0.62
    pmesh = next(m for m in meshes if m.part_id == "pupil_l")
    vi = next(i for i, (_, y) in enumerate(pmesh.vertices) if abs(y - 0.62) < 1e-9)

    plain_dy = _kf(_param(plain, "ParamEyeLOpen"), 0.0).mesh_offsets["pupil_l"][vi][1]
    corr_dy = _kf(_param(corrected, "ParamEyeLOpen"), 0.0).mesh_offsets["pupil_l"][vi][1]
    # The collapse travels _BLINK of the way to the axis, not all of it (see _BLINK: a full collapse
    # makes the eye vanish rather than close). What this test pins is *which axis* it collapses toward.
    assert plain_dy == pytest.approx(_BLINK * (0.64 - 0.62))   # own bbox midline
    assert corr_dy == pytest.approx(_BLINK * (0.65 - 0.62))    # shared lid axis
    assert corr_dy != pytest.approx(plain_dy)


def test_blink_leaves_the_lid_line_visible_instead_of_erasing_the_eye():
    """A shut eye is drawn as a lid line — it must not collapse to zero area and disappear.

    A full collapse lands every vertex on the lid axis, so the triangles go zero-area and the eye
    vanishes into blank skin. Measured through the native Cubism core, Hiyori never degenerates an eye
    mesh: its most-collapsed one still keeps 14.6% of its open height at ParamEyeLOpen=0.
    """
    parts = [("eye_l", R.eye_l, (0.30, 0.60, 0.50, 0.70)),
             ("eye_white_l", R.eye_white_l, (0.33, 0.62, 0.47, 0.68)),
             ("pupil_l", R.pupil_l, (0.38, 0.62, 0.42, 0.66))]
    stack, meshes = _stack_and_meshes(parts)
    params = author_rig(stack, meshes, select_template(stack)).parameters
    closed = _kf(_param(params, "ParamEyeLOpen"), 0.0)

    for m in meshes:
        ys = [y + closed.mesh_offsets[m.part_id][i][1] for i, (_, y) in enumerate(m.vertices)]
        open_h = max(y for _, y in m.vertices) - min(y for _, y in m.vertices)
        assert max(ys) - min(ys) > 0.05 * open_h, f"{m.part_id} collapsed to nothing"


# --------------------------------------------------------------------------------------------------
# #14 — mouth open pivots on the landmark lip line
# --------------------------------------------------------------------------------------------------
def test_mouth_open_uses_landmark_pivot():
    parts = [("mouth", R.mouth, (0.40, 0.30, 0.60, 0.40))]  # mesh bbox center y = 0.35
    stack, meshes = _stack_and_meshes(parts)
    lm = Landmarks(mouth=MouthLandmarks(center=(0.50, 0.34), left_corner=(0.40, 0.34),
                                        right_corner=(0.60, 0.34), top=(0.50, 0.39),
                                        bottom=(0.50, 0.31)))
    params = author_rig(stack, meshes, select_template(stack), landmarks=lm).parameters
    mp = _param(params, "ParamMouthOpenY")
    open_kf = _kf(mp, 1.0)
    mmesh = meshes[0]
    # Lens open about the landmark pivot (0.34): below it the lip drops, above it rises a little, and
    # both taper to the mouth corners (x=0.40/0.60, half-width 0.10). Assert per vertex with that taper.
    cx, half = 0.50, 0.10
    saw_drop = saw_rise = False
    for (x, y), (dx, dy) in zip(mmesh.vertices, open_kf.mesh_offsets["mouth"]):
        taper = max(0.0, 1.0 - abs(x - cx) / half)
        if taper <= 1e-9 or y == pytest.approx(0.34):
            assert dy == pytest.approx(0.0)            # a corner (or the pivot line) is anchored
        elif y < 0.34:
            assert dy < 0.0                            # lower lip drops
            saw_drop = True
        else:
            assert dy > 0.0                            # upper lip rises
            saw_rise = True
    assert saw_drop and saw_rise


# --------------------------------------------------------------------------------------------------
# #14 — pupil travel bounded by real eye size (pupils stay in the eye)
# --------------------------------------------------------------------------------------------------
def test_eyeball_travel_bounded_by_eye_width():
    parts = [("pupil_l", R.pupil_l, (0.48, 0.58, 0.52, 0.62))]  # pupil bbox width 0.04
    stack, meshes = _stack_and_meshes(parts)
    lm = Landmarks(eye_l=EyeLandmarks(center=(0.50, 0.60), lid_top=(0.50, 0.64),
                                      lid_bottom=(0.50, 0.56), inner=(0.40, 0.60),
                                      outer=(0.60, 0.60), pupil=(0.50, 0.60)))  # eye width 0.20
    params = author_rig(stack, meshes, select_template(stack), landmarks=lm).parameters
    ex = _param(params, "ParamEyeBallX")
    dx = _kf(ex, 1.0).mesh_offsets["pupil_l"][0][0]
    assert dx == pytest.approx(0.25 * 0.20)  # _EYEBALL_FRAC * eye width, not pupil bbox
    # the shifted pupil center stays within [inner.x, outer.x]
    assert 0.40 <= 0.50 + dx <= 0.60


# --------------------------------------------------------------------------------------------------
# #16 — limb articulation requires landmark joints
# --------------------------------------------------------------------------------------------------
_LIMB_PARTS = [("face_base", R.face_base, (0.2, 0.5, 0.8, 0.95)),
               ("torso", R.torso, (0.35, 0.20, 0.65, 0.55)),
               ("arm_l", R.arm_l, (0.20, 0.20, 0.35, 0.55)),
               ("leg_l", R.leg_l, (0.40, 0.00, 0.50, 0.25))]


def test_limbs_emitted_from_mesh_geometry():
    """Limbs are authored from their own mesh, so they no longer need (or trust) landmark joints.

    The joint used to come from a silhouette landmark, but the de-cardboard split leaves the two arms
    sharing one texture, so that silhouette is of *both* arms and its centroid is the body midline. The
    mesh carries only this side's triangles, so it is the honest source — and it is present whether or
    not landmarks were extracted."""
    stack, meshes = _stack_and_meshes(_LIMB_PARTS)
    for lm in (Landmarks(), None):
        params = author_rig(stack, meshes, select_template(stack), landmarks=lm).parameters
        assert _param(params, "ParamArmLA") is not None
        assert _param(params, "ParamLegLA") is not None
        arm = _param(params, "ParamArmLA")
        moved = any(dx or dy for dx, dy in _kf(arm, arm.max).mesh_offsets["arm_l"])
        assert moved


def test_limb_pivots_on_its_own_centre_not_the_midline():
    """The bug the Cubism render caught: a limb pivoting on the body midline instead of its own
    shoulder swings the far end (the hand/foot) in a wide arc and tears it off the body.

    ``arm_l`` sits at x 0.20-0.35 (centre 0.275); the body midline is 0.50. A rotation about the true
    shoulder barely moves the top of the limb and moves the bottom the most — a hinge. A rotation about
    the midline (0.225 to the arm's right) instead throws the *whole* limb sideways, so even the topmost
    vertices shift a lot. We assert the top stays put relative to the bottom, which only holds if the
    pivot is on the limb."""
    stack, meshes = _stack_and_meshes(_LIMB_PARTS)
    params = author_rig(stack, meshes, select_template(stack), landmarks=None).parameters
    arm_mesh = next(m for m in meshes if m.part_id == "arm_l")
    arm = _param(params, "ParamArmLA")
    offs = _kf(arm, arm.max).mesh_offsets["arm_l"]

    ys = [y for _, y in arm_mesh.vertices]
    top_y, bot_y = max(ys), min(ys)
    top_shift = max(abs(offs[i][0]) for i, (_, y) in enumerate(arm_mesh.vertices) if y == top_y)
    bot_shift = max(abs(offs[i][0]) for i, (_, y) in enumerate(arm_mesh.vertices) if y == bot_y)
    # a true shoulder hinge: the shoulder end barely moves, the hand end moves most
    assert top_shift < 0.25 * bot_shift, (top_shift, bot_shift)


# --------------------------------------------------------------------------------------------------
# Regression: solver without landmarks is unchanged (bbox fallback still works)
# --------------------------------------------------------------------------------------------------
def test_solver_without_landmarks_still_authors_face():
    parts = [("face_base", R.face_base, (0.2, 0.1, 0.8, 0.9)),
             ("eye_l", R.eye_l, (0.30, 0.55, 0.45, 0.63)),
             ("mouth", R.mouth, (0.42, 0.30, 0.58, 0.38))]
    stack, meshes = _stack_and_meshes(parts)
    params = author_rig(stack, meshes, select_template(stack)).parameters
    ids = {p.id for p in params}
    assert {"ParamEyeLOpen", "ParamMouthOpenY", "ParamAngleX", "ParamAngleZ"} <= ids


# --------------------------------------------------------------------------------------------------
# Integration: full pipeline emits limb params for a full-body sample
# --------------------------------------------------------------------------------------------------
def test_pipeline_fullbody_emits_limb_params(tmp_path):
    pytest.importorskip("PIL")
    from image2live2d.core import decompose
    from image2live2d.pipeline import rig_from_stack
    from image2live2d.samples import make_sample_fullbody

    layer_dir = make_sample_fullbody(tmp_path / "fb")
    rig = rig_from_stack(decompose.from_layer_dir(layer_dir), name="fb")
    ids = {p.id for p in rig.parameters}
    assert ids & {"ParamArmLA", "ParamArmRA", "ParamLegLA", "ParamLegRA"}
