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


def test_shy_expression_fades_the_cheek_blush():
    # The "shy" expression eases ParamCheek from the neutral 0 to a full blush and holds it.
    params = [Parameter(id="ParamCheek", min=0.0, max=1.0, default=0.0),
              Parameter(id="ParamMouthForm", min=-1.0, max=1.0, default=0.0)]
    shy = next(a for a in generate_expressions(params) if a.name == "shy")
    cheek = next(ln for ln in shy.lanes if ln.param_id == "ParamCheek")
    assert cheek.keyframes[0].value == 0.0 and cheek.keyframes[-1].value == 1.0


def test_smile_layers_a_happy_squint_but_sad_angry_keep_eyes_open():
    # A full face gains ParamEyeL/RSmile, so the smile clip drives a genuine eye-form squint (eased from
    # the neutral 0 to the pose value and held); sad/angry deliberately do NOT touch any eye axis.
    _, _, auth = _face_params()
    anims = {a.name: a for a in generate_expressions(auth.parameters)}
    smile_ids = {ln.param_id for ln in anims["smile"].lanes}
    assert {"ParamEyeLSmile", "ParamEyeRSmile"} <= smile_ids
    squint = next(ln for ln in anims["smile"].lanes if ln.param_id == "ParamEyeLSmile")
    assert [(k.frame, k.value) for k in squint.keyframes][0] == (0, 0.0)   # from neutral
    assert squint.keyframes[-1].value > 0.0                                # to a held squint
    for name in ("sad", "angry"):
        ids = {ln.param_id for ln in anims[name].lanes}
        assert not any(i.startswith("ParamEye") for i in ids)              # no eye axis in sad/angry


def test_angry_and_sad_layer_a_brow_tilt():
    # A full face gains ParamBrowL/RForm; angry drives them to +max (furrow) and sad to -min (worried),
    # each eased from the neutral 0. Smile does NOT use the tilt axis.
    _, _, auth = _face_params()
    anims = {a.name: a for a in generate_expressions(auth.parameters)}
    angry_form = next(ln for ln in anims["angry"].lanes if ln.param_id == "ParamBrowLForm")
    sad_form = next(ln for ln in anims["sad"].lanes if ln.param_id == "ParamBrowLForm")
    assert angry_form.keyframes[0].value == 0.0 and angry_form.keyframes[-1].value > 0.0   # furrow
    assert sad_form.keyframes[0].value == 0.0 and sad_form.keyframes[-1].value < 0.0        # worried
    assert not any(ln.param_id.endswith("Form") and ln.param_id.startswith("ParamBrow")
                   for ln in anims["smile"].lanes)                                          # smile: no tilt


def test_brow_form_is_a_tilt_inner_end_drops_at_plus_one():
    # +1 (angry) must ROTATE the brow: the inner end (toward the face midline) drops and the outer rises,
    # and both brows furrow inward from one shared value — not a flat raise.
    eyes = [("eye_l", (0.30, 0.60, 0.45, 0.72)), ("eye_r", (0.55, 0.60, 0.70, 0.72))]
    brows = [("eyebrow_l", (0.28, 0.76, 0.46, 0.82)), ("eyebrow_r", (0.54, 0.76, 0.72, 0.82))]
    layers = [Layer(id="face_base", semantic_role=R.face_base, texture_path=Path("f.png"),
                    draw_order=0, width=64, height=64)]
    meshes = [grid_mesh("face_base", (0.2, 0.4, 0.8, 0.95), lambda u, v: 255, grid=2)]
    for i, (pid, rect) in enumerate(eyes + brows):
        role = {"eye_l": R.eye_l, "eye_r": R.eye_r, "eyebrow_l": R.eyebrow_l, "eyebrow_r": R.eyebrow_r}[pid]
        layers.append(Layer(id=pid, semantic_role=role, texture_path=Path(f"{pid}.png"),
                            draw_order=10 + i, width=64, height=64))
        meshes.append(grid_mesh(pid, rect, lambda u, v: 255, grid=6))
    stack = LayerStack(layers=layers, canvas_width=64, canvas_height=64)
    auth = author_rig(stack, meshes, select_template(stack))
    mbp = {m.part_id: m for m in meshes}
    face_cx = 0.5   # eye centroid
    for pid, form_id in (("eyebrow_l", "ParamBrowLForm"), ("eyebrow_r", "ParamBrowRForm")):
        p = next(pp for pp in auth.parameters if pp.id == form_id)
        off = p.keyforms[-1].mesh_offsets[pid]           # +1 keyform
        verts = mbp[pid].vertices
        pairs = list(zip(verts, off))
        inner = min(pairs, key=lambda t: abs(t[0][0] - face_cx))   # end nearest the midline
        outer = max(pairs, key=lambda t: abs(t[0][0] - face_cx))
        assert inner[1][1] < 0.0 < outer[1][1]           # inner drops, outer rises -> a tilt, mirrored L/R


def test_eye_smile_authors_an_upward_arch_not_a_flat_close():
    # The smile deform must be an ARCH (peak at the eye centre) — a vertex near the centre rises more than
    # one at the corner — so it reads as "^", distinct from the flat collapse a blink authors.
    eye_rect = (0.30, 0.60, 0.70, 0.80)
    layers = [Layer(id="face_base", semantic_role=R.face_base, texture_path=Path("f.png"),
                    draw_order=0, width=64, height=64),
              Layer(id="eye_l", semantic_role=R.eye_l, texture_path=Path("e.png"),
                    draw_order=10, width=64, height=64)]
    meshes = [grid_mesh("face_base", (0.2, 0.4, 0.8, 0.95), lambda u, v: 255, grid=2),
              grid_mesh("eye_l", eye_rect, lambda u, v: 255, grid=6)]
    stack = LayerStack(layers=layers, canvas_width=64, canvas_height=64)
    auth = author_rig(stack, meshes, select_template(stack))
    smile = next(p for p in auth.parameters if p.id == "ParamEyeLSmile")
    active = smile.keyforms[-1].mesh_offsets["eye_l"]
    verts = next(m for m in meshes if m.part_id == "eye_l").vertices
    cx = (eye_rect[0] + eye_rect[2]) / 2.0
    # the vertex nearest the horizontal centre vs one nearest a corner (both on the lower half so both rise)
    lower = [(i, x, y) for i, (x, y) in enumerate(verts) if y < (eye_rect[1] + eye_rect[3]) / 2.0]
    ctr = min(lower, key=lambda t: abs(t[1] - cx))
    corner = max(lower, key=lambda t: abs(t[1] - cx))
    assert active[ctr[0]][1] > active[corner[0]][1] > 0.0    # centre rises MORE than the corner -> an arch


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
