"""P5 — the expression sheet (smile / surprise / sad / angry) as reusable animation clips.

Pure-core tests on the authored clips + a cross-backend check that both emitters inherit them (they key
only standard face params, so a stock face gets the whole sheet and both runtimes render it identically).
"""

from __future__ import annotations

from pathlib import Path

from image2live2d.backends.live2d.motion3 import motion3
from image2live2d.backends.nijilive.puppet import build_puppet
from image2live2d.core.assemble import assemble_rig
from image2live2d.core.mesh import grid_mesh
from image2live2d.core.motion import EXPRESSION_NAMES, generate_expressions, generate_idle
from image2live2d.core.rig import author_rig, select_template
from image2live2d.core.types import Layer, LayerStack
from image2live2d.irr.schema import Parameter
from image2live2d.irr.schema import SemanticRole as R


def _face_params():
    """A face rig with the params the expressions drive: eyes (blink), mouth (open/form), brows."""
    parts = [("face_base", R.face_base, (0.20, 0.45, 0.80, 0.95)),
             ("eye_l", R.eye_l, (0.30, 0.68, 0.45, 0.78)),
             ("eye_r", R.eye_r, (0.55, 0.68, 0.70, 0.78)),
             ("eyebrow_l", R.eyebrow_l, (0.30, 0.80, 0.45, 0.85)),
             ("eyebrow_r", R.eyebrow_r, (0.55, 0.80, 0.70, 0.85)),
             ("mouth", R.mouth, (0.42, 0.52, 0.58, 0.60))]
    layers, meshes = [], []
    for i, (pid, role, rect) in enumerate(parts):
        layers.append(Layer(id=pid, semantic_role=role, texture_path=Path(f"{pid}.png"),
                            draw_order=i * 10, width=64, height=64))
        meshes.append(grid_mesh(pid, rect, lambda u, v: 255, grid=2))
    stack = LayerStack(layers=layers, canvas_width=64, canvas_height=64)
    return stack, meshes, author_rig(stack, meshes, select_template(stack))


def test_full_sheet_authored_for_a_full_face():
    _, _, auth = _face_params()
    params = auth.parameters
    anims = {a.name: a for a in generate_expressions(params)}
    assert set(anims) == set(EXPRESSION_NAMES)               # every expression applies to a full face
    for a in anims.values():
        assert a.loop is False                               # an expression is triggered + held, not looped
        assert a.length == 24
        assert a.lanes                                       # each has at least one driven param


def test_pose_eases_from_default_and_holds():
    _, _, auth = _face_params()
    params = auth.parameters
    smile = next(a for a in generate_expressions(params) if a.name == "smile")
    form = next(ln for ln in smile.lanes if ln.param_id == "ParamMouthForm")
    frames = [(k.frame, k.value) for k in form.keyframes]
    assert frames[0] == (0, 0.0)                             # starts at the neutral default
    assert frames[1] == (8, 1.0)                             # eased to the pose by the ramp frame
    assert frames[2] == (24, 1.0)                            # and held to the clip end


def test_present_gated_and_clamped():
    # a rig that only has a mouth-form param: smile authors just that lane; no brow/eye lanes invented.
    params = [Parameter(id="ParamMouthForm", min=-1.0, max=1.0, default=0.0)]
    anims = {a.name: a for a in generate_expressions(params)}
    assert set(anims["smile"].lanes and {ln.param_id for ln in anims["smile"].lanes}) == {"ParamMouthForm"}
    # surprise drives none of this rig's params (no mouth-open/eye/brow) -> skipped entirely
    assert "surprise" not in anims

    # a narrow-range brow clamps the pose into range (angry wants -1.0, param only reaches -0.5)
    narrow = [Parameter(id="ParamBrowLY", min=-0.5, max=0.5, default=0.0)]
    angry = next(a for a in generate_expressions(narrow) if a.name == "angry")
    assert min(k.value for k in angry.lanes[0].keyframes) == -0.5


def test_no_face_params_no_expressions():
    assert generate_expressions([Parameter(id="ParamBreath", min=0.0, max=1.0, default=0.0)]) == []


def test_expressions_emit_on_both_backends():
    stack, meshes, auth = _face_params()
    params = auth.parameters
    anims = generate_idle(params) + generate_expressions(params)
    rig = assemble_rig(name="x", source=None, stack=stack, meshes=meshes, deformers=auth.deformers,
                       parameters=params, physics=[], animations=anims,
                       part_deformers=auth.part_deformers)
    puppet = build_puppet(rig).puppet
    name_by_uuid = {p["uuid"]: p["name"] for p in puppet["param"]}
    for name in EXPRESSION_NAMES:
        assert name in puppet["animations"]                  # nijilive inherits the clip
        niji_targets = {name_by_uuid[ln["uuid"]] for ln in puppet["animations"][name]["lanes"]}
        anim = next(a for a in rig.animations if a.name == name)
        irr_targets = {ln.param_id for ln in anim.lanes}
        live2d_targets = {c["Id"] for c in motion3(anim)["Curves"]}   # Live2D .motion3 inherits it too
        assert niji_targets == irr_targets == live2d_targets
