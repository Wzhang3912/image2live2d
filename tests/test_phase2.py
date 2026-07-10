"""Phase 2: procedural hair physics, body params, archetype classifier (all headless)."""

from __future__ import annotations

from pathlib import Path

from image2live2d.core.assemble import assemble_rig
from image2live2d.core.mesh import grid_mesh
from image2live2d.core.physics import generate_physics
from image2live2d.core.qa import deform_at
from image2live2d.core.rig import author_rig, select_template
from image2live2d.core.types import Layer, LayerStack
from image2live2d.irr.schema import Mesh, SemanticRole as R


def _mk(parts: list[tuple[str, R, tuple]]) -> tuple[LayerStack, list[Mesh]]:
    layers, meshes = [], []
    for i, (pid, role, rect) in enumerate(parts):
        layers.append(Layer(id=pid, semantic_role=role, texture_path=Path(f"{pid}.png"),
                            draw_order=i * 10, width=64, height=64))
        meshes.append(grid_mesh(pid, rect, lambda u, v: 255, grid=2))
    return LayerStack(layers=layers, canvas_width=64, canvas_height=64), meshes


def _rig(parts):
    stack, meshes = _mk(parts)
    template = select_template(stack)
    auth = author_rig(stack, meshes, template)
    phys = generate_physics(stack, auth.parameters)
    rig = assemble_rig(name="t", source=None, stack=stack, meshes=meshes,
                       deformers=auth.deformers, parameters=auth.parameters, physics=phys,
                       archetype=template.name)
    return rig, auth, phys


# --- archetype classifier -------------------------------------------------------------------------
def test_archetype_portrait():
    stack, _ = _mk([("face_base", R.face_base, (0.2, 0.5, 0.8, 0.95)),
                    ("mouth", R.mouth, (0.4, 0.55, 0.6, 0.62))])
    assert select_template(stack).name == "portrait_front"


def test_archetype_halfbody():
    stack, _ = _mk([("face_base", R.face_base, (0.3, 0.6, 0.7, 0.95)),
                    ("torso", R.torso, (0.3, 0.2, 0.7, 0.55)),
                    ("arm_l", R.arm_l, (0.1, 0.2, 0.3, 0.55))])
    assert select_template(stack).name == "halfbody"


def test_archetype_fullbody():
    stack, _ = _mk([("torso", R.torso, (0.3, 0.4, 0.7, 0.7)),
                    ("leg_l", R.leg_l, (0.35, 0.0, 0.5, 0.4))])
    assert select_template(stack).name == "fullbody"


# --- hair physics ---------------------------------------------------------------------------------
def test_hair_physics_authored_and_wired():
    rig, auth, phys = _rig([("face_base", R.face_base, (0.2, 0.1, 0.8, 0.9)),
                            ("hair_front", R.hair_front, (0.2, 0.6, 0.8, 0.98))])
    ids = rig.parameter_ids()
    assert {"ParamHairFront", "ParamAngleX"} <= ids
    # a pendulum drives the hair output from head turn
    assert any(p.driver_param == "ParamAngleX" and p.output_param == "ParamHairFront" for p in phys)
    assert rig.physics  # integrity validator accepted the physics refs


def test_hair_sway_moves_tips_not_roots():
    rig, _, _ = _rig([("face_base", R.face_base, (0.2, 0.1, 0.8, 0.9)),
                      ("hair_front", R.hair_front, (0.2, 0.6, 0.8, 0.98))])
    swung = deform_at(rig, "ParamHairFront", 1.0)["hair_front"]
    rest = rig.mesh_for("hair_front").vertices
    top = max(y for _, y in rest)
    for (nx, _), (rx, ry) in zip(swung, rest):
        if abs(ry - top) < 1e-9:
            assert abs(nx - rx) < 1e-9   # roots (top) stay put
        else:
            assert nx - rx > 0.0         # tips swing +x at +1


def test_no_physics_without_hair():
    stack, meshes = _mk([("face_base", R.face_base, (0.2, 0.1, 0.8, 0.9)),
                         ("mouth", R.mouth, (0.4, 0.4, 0.6, 0.5))])
    auth = author_rig(stack, meshes, select_template(stack))
    assert generate_physics(stack, auth.parameters) == []


# --- body params ----------------------------------------------------------------------------------
def test_body_params_authored_with_correct_range():
    rig, auth, _ = _rig([("face_base", R.face_base, (0.3, 0.6, 0.7, 0.95)),
                         ("torso", R.torso, (0.3, 0.2, 0.7, 0.6)),
                         ("leg_l", R.leg_l, (0.35, 0.0, 0.5, 0.2))])
    ids = rig.parameter_ids()
    assert {"ParamBodyAngleX", "ParamBodyAngleY", "ParamBodyAngleZ"} <= ids
    bx = next(p for p in rig.parameters if p.id == "ParamBodyAngleX")
    # keyforms span the body param's own +-10 range (not the head's +-30)
    assert [k.value for k in bx.keyforms] == [-10.0, 0.0, 10.0]


def test_body_sway_is_coherent_warp():
    rig, _, _ = _rig([("torso", R.torso, (0.2, 0.3, 0.8, 0.7)),
                      ("leg_l", R.leg_l, (0.3, 0.0, 0.5, 0.3))])
    right = deform_at(rig, "ParamBodyAngleX", 10.0)["torso"]
    rest = rig.mesh_for("torso").vertices
    dxs = [nx - rx for (nx, _), (rx, _) in zip(right, rest)]
    assert max(dxs) - min(dxs) > 1e-4  # per-vertex warp, not a rigid slide
