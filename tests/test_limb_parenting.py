"""P4 — kinematic parenting for garments: a sleeve/cuff rides its ARM, not the body.

A clothing appendage that sits *predominantly over an arm* (a sleeve, a cuff) should lag and flare off
that arm's articulation, so its pendulum is driven by ``ParamArm*`` rather than ``ParamBodyAngle*``. The
binding is geometric (footprint overlap), gated strictly so an ordinary torso garment stays on the body
(byte-identical wardrobe). Verified from the RigGraph parent through the garment spec to the physics rig.
"""

from __future__ import annotations

from pathlib import Path

from image2live2d.core.landmark import Landmarks
from image2live2d.core.physics import generate_physics
from image2live2d.core.rig import author_rig, select_template
from image2live2d.core.structure import (
    ARM_L,
    BODY,
    build_rig_graph,
    garment_appendages,
)
from image2live2d.core.types import Layer, LayerStack
from image2live2d.irr.schema import Mesh
from image2live2d.irr.schema import SemanticRole as R


def _mesh(pid, x0, y0, x1, y1):
    verts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return Mesh(part_id=pid, vertices=verts, uvs=[(0.0, 0.0)] * 4, triangles=[(0, 1, 2), (0, 2, 3)])


# torso centre-right, a left arm down the side, and a sleeve over the upper arm that flares out into
# void (free edge -> swingable). Its bottom sits above the waist line so it reads as upper-body clothing
# (a sleeve, not a skirt hem — the skirt planner only claims low hems). A bodice over the torso is the
# byte-safety control that must stay body-driven.
_PARTS = [
    ("torso", R.torso, (0.40, 0.20, 0.60, 0.75)),
    ("arm_l", R.arm_l, (0.15, 0.20, 0.32, 0.72)),
    ("sleeve", R.clothing, (0.08, 0.46, 0.34, 0.74)),
    ("bodice", R.clothing, (0.42, 0.46, 0.58, 0.72)),
]


def _scene(parts=_PARTS):
    layers, meshes = [], []
    for i, (pid, role, box) in enumerate(parts):
        layers.append(Layer(id=pid, semantic_role=role, texture_path=Path(f"{pid}.png"),
                            draw_order=i * 10, width=64, height=64))
        meshes.append(_mesh(pid, *box))
    return LayerStack(layers=layers, canvas_width=64, canvas_height=64), meshes


def test_sleeve_over_arm_binds_to_that_arm():
    stack, meshes = _scene()
    graph = build_rig_graph(stack, meshes)
    assert graph.parent_of("sleeve") == ARM_L        # mostly over the arm -> rides the arm
    assert graph.parent_of("bodice") == BODY         # over the torso -> stays on the body


def test_torso_garment_without_arms_is_unchanged():
    # no arm meshes -> nothing to bind to; every clothing part stays on the body (byte-identical).
    stack, meshes = _scene([_PARTS[0], _PARTS[2], _PARTS[3]])   # torso + "sleeve" + bodice, no arm
    graph = build_rig_graph(stack, meshes)
    assert graph.parent_of("sleeve") == BODY
    assert graph.parent_of("bodice") == BODY


def test_garment_appendage_sleeve_is_arm_driven():
    stack, meshes = _scene()
    specs = {s.part_id: s for s in garment_appendages(stack, meshes, build_rig_graph(stack, meshes))}
    assert "sleeve" in specs                          # free-edged garment over the arm
    assert specs["sleeve"].driver == "ParamArmLA"     # swings with the shoulder, not the body
    assert "ParamArmLB" in specs["sleeve"].extra_drivers   # elbow bend enriches the sway


def test_physics_wires_the_sleeve_to_the_arm_swing():
    stack, meshes = _scene()
    # arm articulation needs a landmark joint; with it, ParamArmLA exists so the sleeve pendulum wires.
    lm = Landmarks(joints={"arm_l": (0.22, 0.65)})
    params = author_rig(stack, meshes, select_template(stack), landmarks=lm).parameters
    ids = {p.id for p in params}
    assert {"ParamArmLA", "ParamCloth0"} <= ids
    rigs = {r.output_param: r for r in generate_physics(stack, params, meshes=meshes)}
    assert rigs["ParamCloth0"].driver_param == "ParamArmLA"   # the sleeve lags the arm, not the torso
