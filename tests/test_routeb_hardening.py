"""Route B hardening — cloth/skirt physics, idle auto-animation, richer role heuristics."""

from __future__ import annotations

from pathlib import Path

import pytest

from image2live2d.core import motion
from image2live2d.core.landmark import Landmarks
from image2live2d.core.assemble import assemble_rig
from image2live2d.core.decompose import role_from_layer_name
from image2live2d.core.mesh import grid_mesh
from image2live2d.core.physics import generate_physics
from image2live2d.core.rig import author_rig, select_template
from image2live2d.core.types import Layer, LayerStack
from image2live2d.irr.params import make_parameter
from image2live2d.irr.schema import SemanticRole as R


def _stack(parts):
    layers, meshes = [], []
    for i, (pid, role, rect) in enumerate(parts):
        layers.append(Layer(id=pid, semantic_role=role, texture_path=Path(f"{pid}.png"),
                            draw_order=i * 10, width=64, height=64))
        meshes.append(grid_mesh(pid, rect, lambda u, v: 255, grid=2))
    return LayerStack(layers=layers, canvas_width=64, canvas_height=64), meshes


# --------------------------------------------------------------------------------------------------
# #19 cloth/skirt physics
# --------------------------------------------------------------------------------------------------
def test_cloth_zones_and_spring_pendulum():
    parts = [("face_base", R.face_base, (0.2, 0.5, 0.8, 0.95)),
             ("torso", R.torso, (0.35, 0.25, 0.65, 0.55)),
             ("skirt", R.clothing, (0.30, 0.05, 0.70, 0.30))]
    stack, meshes = _stack(parts)
    auth = author_rig(stack, meshes, select_template(stack))
    ids = {p.id for p in auth.parameters}
    assert {"ParamSkirtL", "ParamSkirtC", "ParamSkirtR"} <= ids  # hem split into zones

    rigs = {r.output_param: r for r in generate_physics(stack, auth.parameters)}
    for z in ("ParamSkirtL", "ParamSkirtC", "ParamSkirtR"):
        r = rigs[z]
        assert r.model.value == "spring_pendulum"        # springy cloth, not rigid
        assert r.driver_param == "ParamBodyAngleX"        # body sway is the primary driver
        assert "ParamBodyAngleZ" in r.all_drivers()       # body lean also excites it (lower body)


def test_skirt_zone_driven_by_near_leg():
    # with leg joints (landmarks), the side zones couple to the near leg -> all lower body affects skirt
    parts = [("torso", R.torso, (0.35, 0.25, 0.65, 0.55)),
             ("skirt", R.clothing, (0.30, 0.05, 0.70, 0.30)),
             ("leg_l", R.leg_l, (0.40, 0.00, 0.48, 0.25)),
             ("leg_r", R.leg_r, (0.52, 0.00, 0.60, 0.25))]
    stack, meshes = _stack(parts)
    lm = Landmarks(joints={"leg_l": (0.44, 0.25), "leg_r": (0.56, 0.25)})
    auth = author_rig(stack, meshes, select_template(stack), landmarks=lm)
    rigs = {r.output_param: r for r in generate_physics(stack, auth.parameters)}
    assert "ParamLegLA" in rigs["ParamSkirtL"].all_drivers()
    assert "ParamLegRA" in rigs["ParamSkirtR"].all_drivers()


def test_skirt_zone_hem_swings_more_than_waist():
    # a normal skirt (waist -> above the knee); a part reaching the feet is skirt+legs-bundled and is
    # excluded from skirt physics (see _skirtable), so keep the fixture in the real skirt band
    stack, meshes = _stack([("skirt", R.clothing, (0.30, 0.24, 0.70, 0.48))])
    auth = author_rig(stack, meshes, select_template(stack))
    p = next(p for p in auth.parameters if p.id == "ParamSkirtC")
    kf_max = next(k for k in p.keyforms if k.value == p.max)
    offs = kf_max.mesh_offsets["skirt"]
    mesh = meshes[0]
    # compare within the zone's center column (x~0.5), where the window weight is non-zero
    col = [i for i, (x, _) in enumerate(mesh.vertices) if abs(x - 0.5) < 1e-6]
    lowest = min(col, key=lambda i: mesh.vertices[i][1])   # hem
    highest = max(col, key=lambda i: mesh.vertices[i][1])  # waist
    assert abs(offs[lowest][0]) > abs(offs[highest][0])


# --------------------------------------------------------------------------------------------------
# #20 idle auto-animation
# --------------------------------------------------------------------------------------------------
def test_generate_idle_lanes():
    params = [make_parameter(i) for i in
              ("ParamEyeLOpen", "ParamEyeROpen", "ParamBreath", "ParamBodyAngleX")]
    anims = motion.generate_idle(params)
    assert len(anims) == 1
    idle = anims[0]
    assert idle.name == "idle" and idle.loop and idle.length == motion.IDLE_FRAMES
    lane_ids = {ln.param_id for ln in idle.lanes}
    assert lane_ids == {"ParamEyeLOpen", "ParamEyeROpen", "ParamBreath", "ParamBodyAngleX"}
    # blink actually closes (a 0.0 keyframe) and reopens
    blink = next(ln for ln in idle.lanes if ln.param_id == "ParamEyeLOpen")
    vals = [k.value for k in blink.keyframes]
    assert 0.0 in vals and vals[0] == 1.0 and vals[-1] == 1.0


def test_idle_values_within_param_range():
    params = [make_parameter(i) for i in ("ParamBreath", "ParamBodyAngleX")]
    for lane in motion.generate_idle(params)[0].lanes:
        p = next(p for p in params if p.id == lane.param_id)
        for kf in lane.keyframes:
            assert p.min <= kf.value <= p.max


def test_generate_idle_empty_without_idle_params():
    # only a hair output param -> nothing to idle-animate
    assert motion.generate_idle([make_parameter("ParamHairFront")]) == []


def test_idle_uses_head_sway_without_body():
    params = [make_parameter("ParamAngleX")]  # no body param
    idle = motion.generate_idle(params)[0]
    assert {ln.param_id for ln in idle.lanes} == {"ParamAngleX"}


# --------------------------------------------------------------------------------------------------
# #20 emitter — animations land in the puppet, lanes resolve to param uuids
# --------------------------------------------------------------------------------------------------
def test_emitter_writes_animation_lanes():
    from image2live2d.backends.nijilive.puppet import build_puppet

    stack, meshes = _stack([("eye_l", R.eye_l, (0.3, 0.55, 0.45, 0.63)),
                            ("face_base", R.face_base, (0.2, 0.1, 0.8, 0.9))])
    auth = author_rig(stack, meshes, select_template(stack))
    anims = motion.generate_idle(auth.parameters)
    rig = assemble_rig(name="t", source=None, stack=stack, meshes=meshes,
                       deformers=auth.deformers, parameters=auth.parameters, physics=[],
                       animations=anims)
    build = build_puppet(rig)
    assert "idle" in build.puppet["animations"]
    idle = build.puppet["animations"]["idle"]
    assert idle["length"] == motion.IDLE_FRAMES and idle["leadIn"] == -1
    uuids = {p["uuid"] for p in build.puppet["param"]}
    for lane in idle["lanes"]:
        assert lane["uuid"] in uuids
        assert lane["merge_mode"] == "Forced"
        assert lane["interpolation"] in {"Linear", "Cubic", "Stepped", "Nearest", "Bezier"}


# --------------------------------------------------------------------------------------------------
# #21 richer role heuristics
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,role", [
    ("ponytail", R.hair_back),
    ("twin_tail", R.hair_back),
    ("braid", R.hair_back),
    ("sideburn", R.hair_side),
    ("back_hair", R.hair_back),
    ("jacket", R.clothing),
    ("school_uniform", R.clothing),
    ("left_boot", R.leg_l),
    ("right_thigh", R.leg_r),
    ("earring", R.accessory),
    ("waist", R.torso),
    ("forearm_l", R.arm_l),
])
def test_role_heuristics(name, role):
    assert role_from_layer_name(name) is role


def test_lift_accessory_above_occluding_hair():
    from image2live2d.pipeline import _lift_occluded_accessories
    parts = [("flower", R.accessory, (0.45, 0.80, 0.55, 0.90)),     # small head ornament, drawn behind
             ("hair_front", R.hair_front, (0.30, 0.75, 0.70, 0.98))]  # overlaps + drawn on top
    stack, meshes = _stack(parts)
    flower = next(L for L in stack.layers if L.id == "flower")
    hair = next(L for L in stack.layers if L.id == "hair_front")
    assert flower.draw_order < hair.draw_order
    _lift_occluded_accessories(stack, meshes)
    assert flower.draw_order > hair.draw_order  # lifted on top of the hair that hid it


def test_head_accessory_follows_head_turn_body_accessory_follows_body():
    """A head ornament must move with ParamAngleX (head turn); a waist accessory must not — it moves
    with ParamBodyAngleX instead. Regression for the 'detached bow': head accessories had no head-turn
    keyform, so they floated in place while the head turned away (caught by the deformation preview)."""
    parts = [("face_base", R.face_base, (0.20, 0.50, 0.80, 0.95)),
             ("bow", R.accessory, (0.40, 0.92, 0.60, 0.99)),     # on top of the head
             ("torso", R.torso, (0.30, 0.20, 0.70, 0.55)),
             ("belt", R.accessory, (0.40, 0.20, 0.60, 0.30))]    # at the waist
    stack, meshes = _stack(parts)
    auth = author_rig(stack, meshes, select_template(stack))

    def moved_by(param_id, part_id):
        p = next(p for p in auth.parameters if p.id == param_id)
        return any(any(dx or dy for dx, dy in kf.mesh_offsets.get(part_id, [])) for kf in p.keyforms)

    assert moved_by("ParamAngleX", "bow")            # head ornament turns with the head
    assert not moved_by("ParamAngleX", "belt")       # waist accessory does not
    assert moved_by("ParamBodyAngleX", "belt")       # ...it follows the body instead


def test_non_overlapping_accessory_not_lifted():
    from image2live2d.pipeline import _lift_occluded_accessories
    parts = [("cuff", R.accessory, (0.10, 0.15, 0.20, 0.30)),       # wrist cuff, nowhere near hair
             ("hair_front", R.hair_front, (0.30, 0.75, 0.70, 0.98))]
    stack, meshes = _stack(parts)
    cuff = next(L for L in stack.layers if L.id == "cuff")
    before = cuff.draw_order
    _lift_occluded_accessories(stack, meshes)
    assert cuff.draw_order == before  # untouched (no hair overlap)


@pytest.mark.parametrize("name,role", [
    # See-through's real layer vocabulary (calibrated against a live decompose run, 2026-06-30).
    # The trap: the substring "wear" contains "ear", so every *wear part used to map to ear_l.
    ("front hair", R.hair_front),
    ("back hair", R.hair_back),
    ("face", R.face_base),
    ("eyebrow", R.eyebrow_l),
    ("eyelash", R.eye_l),
    ("eyewhite", R.eye_white_l),
    ("irides", R.pupil_l),       # not "iris" — "irid"
    ("ears", R.ear_l),
    ("mouth", R.mouth),
    ("neck", R.neck),
    ("headwear", R.accessory),   # hat
    ("eyewear", R.accessory),    # glasses
    ("earwear", R.accessory),    # earrings
    ("handwear", R.accessory),   # gloves
    ("topwear", R.clothing),
    ("bottomwear", R.clothing),
    ("legwear", R.clothing),
    ("footwear", R.clothing),    # NOT ear_l (contains "ear")
    ("neckwear", R.clothing),
])
def test_seethrough_vocabulary(name, role):
    assert role_from_layer_name(name) is role
