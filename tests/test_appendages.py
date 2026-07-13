"""P4 — accessory appendage sway. A dangling ornament gets a gentle pendulum driven by the structural
group the RigGraph bound it to (head ornament -> head turn; waist charm -> body sway), on top of still
following that group's turn.
"""

from __future__ import annotations

from pathlib import Path

from image2live2d.core.physics import generate_physics
from image2live2d.core.rig import author_rig, select_template
from image2live2d.core.structure import accessory_appendages, build_rig_graph
from image2live2d.core.types import Layer, LayerStack
from image2live2d.irr.schema import Mesh
from image2live2d.irr.schema import SemanticRole as R


def _mesh(pid, x0, y0, x1, y1):
    verts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return Mesh(part_id=pid, vertices=verts, uvs=[(0.0, 0.0)] * 4, triangles=[(0, 1, 2), (0, 2, 3)])


_PARTS = [
    ("face", R.face_base, (0.30, 0.55, 0.70, 0.95)),
    ("torso", R.torso, (0.35, 0.42, 0.65, 0.75)),
    ("bow", R.accessory, (0.42, 0.66, 0.58, 0.78)),    # on the head
    ("belt", R.accessory, (0.40, 0.30, 0.60, 0.38)),   # at the waist
]


def _scene(parts=_PARTS):
    layers, meshes = [], []
    for i, (pid, role, box) in enumerate(parts):
        layers.append(Layer(id=pid, semantic_role=role, texture_path=Path(f"{pid}.png"),
                            draw_order=i * 10, width=64, height=64))
        meshes.append(_mesh(pid, *box))
    return LayerStack(layers=layers, canvas_width=64, canvas_height=64), meshes


def test_appendage_driver_follows_the_graph_parent():
    stack, meshes = _scene()
    graph = build_rig_graph(stack, meshes)
    by_part = {a.part_id: a for a in accessory_appendages(stack, meshes, graph)}
    assert by_part["bow"].driver == "ParamAngleX"        # head ornament rides the head turn
    assert by_part["belt"].driver == "ParamBodyAngleX"   # waist charm rides the body sway
    assert {by_part["bow"].param_id, by_part["belt"].param_id} == {"ParamAcc0", "ParamAcc1"}


def test_author_gives_each_accessory_a_sway_param():
    stack, meshes = _scene()
    params = {p.id: p for p in author_rig(stack, meshes, select_template(stack)).parameters}
    assert "ParamAcc0" in params and "ParamAcc1" in params

    def parts_moved(pid):
        return {k for kf in params[pid].keyforms for k, offs in kf.mesh_offsets.items()
                if any(dx or dy for dx, dy in offs)}

    assert parts_moved("ParamAcc0") == {"bow"}           # each sway param drives only its ornament
    assert parts_moved("ParamAcc1") == {"belt"}


def test_accessory_still_follows_the_turn():
    # regression: adding sway must not remove the rigid turn-follow the accessory already had.
    stack, meshes = _scene()
    params = {p.id: p for p in author_rig(stack, meshes, select_template(stack)).parameters}

    def moved_by(param_id, part_id):
        return any(any(dx or dy for dx, dy in kf.mesh_offsets.get(part_id, []))
                   for kf in params[param_id].keyforms)

    assert moved_by("ParamAngleX", "bow")                # head ornament still turns with the head
    assert moved_by("ParamBodyAngleX", "belt")           # waist charm still sways with the body


def test_pendulums_driven_by_parent_and_gated_on_meshes():
    stack, meshes = _scene()
    params = author_rig(stack, meshes, select_template(stack)).parameters
    rigs = {r.output_param: r for r in generate_physics(stack, params, meshes=meshes)}
    assert rigs["ParamAcc0"].driver_param == "ParamAngleX"
    assert rigs["ParamAcc1"].driver_param == "ParamBodyAngleX"
    # mesh-less physics can't build the parent graph -> no accessory pendulums (params still exist)
    base = {r.output_param for r in generate_physics(stack, params)}
    assert "ParamAcc0" not in base and "ParamAcc1" not in base
