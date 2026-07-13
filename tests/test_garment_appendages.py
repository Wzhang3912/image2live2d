"""P4b — garment appendage sway. A clothing part that hangs free (cape, long sleeve, coattail) gets a
body-driven pendulum, while a bodice glued to the torso stays rigid — the two are told apart by the P1
dynamics free-edge score run on the mesh silhouette (no alpha), not by name or role.
"""

from __future__ import annotations

from pathlib import Path

from image2live2d.core.physics import generate_physics
from image2live2d.core.rig import author_rig, select_template
from image2live2d.core.structure import (
    DynamicsVerdict,
    analyze_meshes,
    build_rig_graph,
    garment_appendages,
)
from image2live2d.core.types import Layer, LayerStack
from image2live2d.irr.schema import Mesh
from image2live2d.irr.schema import SemanticRole as R


def _mesh(pid, x0, y0, x1, y1):
    verts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return Mesh(part_id=pid, vertices=verts, uvs=[(0.0, 0.0)] * 4, triangles=[(0, 1, 2), (0, 2, 3)])


# torso in the centre; a cape that hangs wider and lower into void (free edge); a bodice fully inside the
# torso silhouette (every edge glued -> no free edge); a proper waist hem (owned by the skirt planner).
_PARTS = [
    ("torso", R.torso, (0.35, 0.30, 0.65, 0.75)),
    ("cape", R.clothing, (0.25, 0.10, 0.75, 0.55)),
    ("bodice", R.clothing, (0.37, 0.45, 0.63, 0.72)),
]


def _scene(parts=_PARTS):
    layers, meshes = [], []
    for i, (pid, role, box) in enumerate(parts):
        layers.append(Layer(id=pid, semantic_role=role, texture_path=Path(f"{pid}.png"),
                            draw_order=i * 10, width=64, height=64))
        meshes.append(_mesh(pid, *box))
    return LayerStack(layers=layers, canvas_width=64, canvas_height=64), meshes


def test_free_edge_separates_cape_from_bodice():
    stack, meshes = _scene()
    dyn = {d.part_id: d for d in analyze_meshes(stack, meshes)}
    assert dyn["cape"].verdict is not DynamicsVerdict.rigid    # hangs into void -> swings
    assert dyn["bodice"].verdict is DynamicsVerdict.rigid      # glued to the torso -> rigid
    assert dyn["cape"].free_edge_ratio > dyn["bodice"].free_edge_ratio


def test_only_the_free_edged_garment_becomes_an_appendage():
    stack, meshes = _scene()
    graph = build_rig_graph(stack, meshes)
    specs = {s.part_id: s for s in garment_appendages(stack, meshes, graph)}
    assert set(specs) == {"cape"}                              # bodice excluded, torso isn't clothing
    assert specs["cape"].param_id == "ParamCloth0"
    assert specs["cape"].driver == "ParamBodyAngleX"           # a garment rides the body


def test_author_sways_the_cape_only():
    stack, meshes = _scene()
    params = {p.id: p for p in author_rig(stack, meshes, select_template(stack)).parameters}
    assert "ParamCloth0" in params

    def parts_moved(pid):
        return {k for kf in params[pid].keyforms for k, offs in kf.mesh_offsets.items()
                if any(dx or dy for dx, dy in offs)}

    assert parts_moved("ParamCloth0") == {"cape"}             # the cape's free hem swings, nothing else


def test_physics_wires_a_body_driven_cloth_pendulum():
    stack, meshes = _scene()
    params = author_rig(stack, meshes, select_template(stack)).parameters
    rigs = {r.output_param: r for r in generate_physics(stack, params, meshes=meshes)}
    assert "ParamCloth0" in rigs
    assert rigs["ParamCloth0"].driver_param == "ParamBodyAngleX"
    assert rigs["ParamCloth0"].model.value == "spring_pendulum"     # cloth sheet, not a stiff pendulum
    # mesh-less physics can't score the silhouette -> no garment pendulums (the param may still exist)
    base = {r.output_param for r in generate_physics(stack, params)}
    assert "ParamCloth0" not in base


def test_a_rigid_only_wardrobe_adds_nothing():
    # torso + bodice only: no free-edged garment -> no ParamCloth params at all (byte-identical wardrobe)
    stack, meshes = _scene([_PARTS[0], _PARTS[2]])
    params = {p.id for p in author_rig(stack, meshes, select_template(stack)).parameters}
    assert not any(p.startswith("ParamCloth") for p in params)


def test_skirt_hem_is_left_to_the_skirt_planner():
    # a proper waist hem is skirtable -> owned by skirt_zones, never a garment appendage
    stack, meshes = _scene([_PARTS[0], ("skirt", R.clothing, (0.30, 0.10, 0.70, 0.42))])
    graph = build_rig_graph(stack, meshes)
    assert garment_appendages(stack, meshes, graph) == []
