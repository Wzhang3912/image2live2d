"""Phase 4 exit gate — same IRR emits equivalently on Route B (nijilive) and Route A (Live2D).

The IRR is the contract; both emitters must represent the same params, the same physics driver->output
pairs, and the same animation targets. Catching divergence here is independent of the (gated) .moc3."""

from __future__ import annotations

from pathlib import Path

from image2live2d.backends.live2d.physics3 import physics3
from image2live2d.backends.live2d.motion3 import motion3
from image2live2d.backends.nijilive.puppet import build_puppet
from image2live2d.core.assemble import assemble_rig
from image2live2d.core.mesh import grid_mesh
from image2live2d.core.motion import generate_idle
from image2live2d.core.physics import generate_physics
from image2live2d.core.rig import author_rig, select_template
from image2live2d.core.types import Layer, LayerStack
from image2live2d.irr.schema import SemanticRole as R


def _rig():
    parts = [("face_base", R.face_base, (0.2, 0.5, 0.8, 0.95)),
             ("eye_l", R.eye_l, (0.30, 0.70, 0.45, 0.78)),
             ("eye_r", R.eye_r, (0.55, 0.70, 0.70, 0.78)),
             ("mouth", R.mouth, (0.42, 0.55, 0.58, 0.63)),
             ("hair_front", R.hair_front, (0.2, 0.75, 0.8, 0.98)),
             ("hair_back", R.hair_back, (0.2, 0.55, 0.8, 0.95)),
             ("torso", R.torso, (0.35, 0.20, 0.65, 0.55)),
             ("clothing", R.clothing, (0.30, 0.05, 0.70, 0.30))]
    layers, meshes = [], []
    for i, (pid, role, rect) in enumerate(parts):
        layers.append(Layer(id=pid, semantic_role=role, texture_path=Path(f"{pid}.png"),
                            draw_order=i * 10, width=64, height=64))
        meshes.append(grid_mesh(pid, rect, lambda u, v: 255, grid=2))
    stack = LayerStack(layers=layers, canvas_width=64, canvas_height=64)
    auth = author_rig(stack, meshes, select_template(stack))
    phys = generate_physics(stack, auth.parameters)
    anims = generate_idle(auth.parameters)
    return assemble_rig(name="x", source=None, stack=stack, meshes=meshes,
                        deformers=auth.deformers, parameters=auth.parameters, physics=phys,
                        animations=anims)


def _niji_output_drivers(puppet) -> dict[str, set[str]]:
    """output_param -> set of driver params, reconstructed from nijilive anchors.

    Each anchor is bound (transform.t.x) by one or more driver params; its child SimplePhysics nodes
    output to params. So an output's drivers = the params binding its anchor."""
    name_by_uuid = {p["uuid"]: p["name"] for p in puppet["param"]}
    # which params bind each anchor uuid via transform.t.x
    anchor_drivers: dict[int, set[str]] = {}
    for p in puppet["param"]:
        for b in p["bindings"]:
            if b["param_name"].startswith("transform.t."):  # t.x (sway) or t.y (bob)
                anchor_drivers.setdefault(b["node"], set()).add(p["name"])
    out: dict[str, set[str]] = {}
    for child in puppet["nodes"]["children"]:
        if not child["name"].startswith("physics_anchor_"):
            continue
        drivers = anchor_drivers.get(child["uuid"], set())
        for sp in child["children"]:
            out[name_by_uuid[sp["param"]]] = drivers
    return out


def test_physics_pairs_match_across_backends():
    rig = _rig()
    niji = _niji_output_drivers(build_puppet(rig).puppet)
    live2d = {
        s["Output"][0]["Destination"]["Id"]: {i["Source"]["Id"] for i in s["Input"]}
        for s in physics3(rig)["PhysicsSettings"]
    }
    irr = {ph.output_param: set(ph.all_drivers()) for ph in rig.physics}
    assert niji == live2d == irr      # same output->drivers mapping on both backends and the IRR
    assert irr                         # non-empty (hair + skirt zones)
    # skirt zones are multi-driven (body sway + lean) — "all lower body affects the skirt"
    assert any(len(d) > 1 for d in irr.values())


def test_animation_targets_match_across_backends():
    rig = _rig()
    puppet = build_puppet(rig).puppet
    name_by_uuid = {p["uuid"]: p["name"] for p in puppet["param"]}

    niji_idle = puppet["animations"]["idle"]
    niji_targets = {name_by_uuid[l["uuid"]] for l in niji_idle["lanes"]}
    niji_counts = {name_by_uuid[l["uuid"]]: len(l["keyframes"]) for l in niji_idle["lanes"]}

    anim = next(a for a in rig.animations if a.name == "idle")
    live2d_targets = {c["Id"] for c in motion3(anim)["Curves"]}
    irr_targets = {l.param_id for l in anim.lanes}

    assert niji_targets == live2d_targets == irr_targets
    # same number of keyframes per lane on the nijilive side as the IRR
    for lane in anim.lanes:
        assert niji_counts[lane.param_id] == len(lane.keyframes)
