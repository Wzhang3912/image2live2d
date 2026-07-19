"""Auto-physics emission: SimplePhysics nodes + an anchor bound to the driver param.

Schema verified against nijilive source (core/nodes/drivers/simplephysics.d, core/param/binding.d,
core/nodes/node.d transform targets). Structure is checked here; the actual swing magnitude can only
be judged in the nijilive runtime."""

from __future__ import annotations

from pathlib import Path

from image2live2d.backends.nijilive.puppet import build_puppet
from image2live2d.core.assemble import assemble_rig
from image2live2d.core.mesh import grid_mesh
from image2live2d.core.physics import generate_physics
from image2live2d.core.rig import author_rig, select_template
from image2live2d.core.types import Layer, LayerStack
from image2live2d.irr.schema import SemanticRole as R


def _hair_rig():
    parts = [("face_base", R.face_base, (0.2, 0.1, 0.8, 0.9)),
             ("hair_front", R.hair_front, (0.2, 0.6, 0.8, 0.98))]
    layers, meshes = [], []
    for i, (pid, role, rect) in enumerate(parts):
        layers.append(Layer(id=pid, semantic_role=role, texture_path=Path(f"{pid}.png"),
                            draw_order=i * 10, width=64, height=64))
        meshes.append(grid_mesh(pid, rect, lambda u, v: 255, grid=2))
    stack = LayerStack(layers=layers, canvas_width=64, canvas_height=64)
    auth = author_rig(stack, meshes, select_template(stack))
    phys = generate_physics(stack, auth.parameters)
    return assemble_rig(name="h", source=None, stack=stack, meshes=meshes,
                        deformers=auth.deformers, parameters=auth.parameters, physics=phys,
                        archetype="portrait_front")


def test_simplephysics_node_emitted_and_wired():
    build = build_puppet(_hair_rig())
    children = build.puppet["nodes"]["children"]
    anchors = [c for c in children if c["type"] == "Node" and c["name"].startswith("physics_anchor")]
    # Two anchors: the head-turn anchor drives the horizontal sway, and a separate ParamAngleY anchor
    # drives the vertical nod-bounce (its anchor moves on t.y). They are grouped by driver set.
    assert len(anchors) == 2
    param_by_name = {p["name"]: p for p in build.puppet["param"]}

    def sp_for(output_name):
        want = param_by_name[output_name]["uuid"]
        for a in anchors:
            for n in a["children"]:
                if n["type"] == "SimplePhysics" and n["param"] == want:
                    return a, n
        raise AssertionError(f"no SimplePhysics drives {output_name}")

    sway_anchor, sp = sp_for("ParamHairFront")
    assert sp["model_type"] == "Pendulum" and sp["map_mode"] == "AngleLength"
    assert sp["length"] > 0 and len(sp["output_scale"]) == 2
    assert "ParamAngleY" not in sway_anchor["name"]                 # sway is not pitch-driven

    bounce_anchor, _ = sp_for("ParamHairFrontV")                    # the nod-bounce exists...
    assert "ParamAngleY" in bounce_anchor["name"]                  # ...and its anchor is pitch-driven


def test_driver_param_has_transform_binding_to_anchor():
    build = build_puppet(_hair_rig())
    children = build.puppet["nodes"]["children"]
    anchor = next(c for c in children if c["name"].startswith("physics_anchor"))
    ax = next(p for p in build.puppet["param"] if p["name"] == "ParamAngleX")

    tbs = [b for b in ax["bindings"] if b["param_name"] == "transform.t.x"]
    assert len(tbs) == 1
    tb = tbs[0]
    assert tb["node"] == anchor["uuid"]
    # binding grid matches the param's x axis points (nijilive enforces this), scalar values[x][y]
    assert len(tb["values"]) == len(ax["axis_points"][0])
    assert all(len(row) == 1 and isinstance(row[0], float) for row in tb["values"])
    # ParamAngleX now drives the head GROUP's rotation (one rigid unit) instead of per-part head
    # deforms — the anchor transform binding coexists with the head-group rotation binding.
    assert any(b["param_name"] == "transform.r.y" for b in ax["bindings"])


def test_no_physics_nodes_without_hair():
    # example rig (face + mouth, no hair) -> no anchors
    from image2live2d.irr.example import build_example_rig
    build = build_puppet(build_example_rig())
    assert not [c for c in build.puppet["nodes"]["children"] if c["name"].startswith("physics_anchor")]
