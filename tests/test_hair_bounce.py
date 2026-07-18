"""Nod -> hair bounce (vertical hair physics).

A horizontal sway param can't express a nod: a "Y" pitch input is inert for an angle output (it slides
the anchor down its own string), so for months a nod moved 0 of the hair chains — the hair only rode
the head-turn deformation, which reads stiff. The fix is a per-role VERTICAL bounce output param
(ParamHair*V) on its own pendulum, fed pitch as an Angle input (tips gravity -> swings -> settles). These
pin that the bounce param exists, deforms vertically, and is wired so a nod drives it and a turn doesn't.
"""

from __future__ import annotations

from pathlib import Path

from image2live2d.backends.live2d.physics3 import physics3
from image2live2d.core.assemble import assemble_rig
from image2live2d.core.mesh import grid_mesh
from image2live2d.core.physics import generate_physics
from image2live2d.core.rig import author_rig, select_template
from image2live2d.core.types import Layer, LayerStack
from image2live2d.irr.schema import SemanticRole as R


def _hair_rig(parts=None):
    parts = parts or [("face_base", R.face_base, (0.2, 0.1, 0.8, 0.9)),
                      ("hair_front", R.hair_front, (0.2, 0.62, 0.8, 0.98)),
                      ("hair_back", R.hair_back, (0.2, 0.55, 0.8, 0.95))]
    layers, meshes = [], []
    for i, (pid, role, rect) in enumerate(parts):
        layers.append(Layer(id=pid, semantic_role=role, texture_path=Path(f"{pid}.png"),
                            draw_order=i * 10, width=64, height=64))
        meshes.append(grid_mesh(pid, rect, lambda u, v: 255, grid=3))
    stack = LayerStack(layers=layers, canvas_width=64, canvas_height=64)
    auth = author_rig(stack, meshes, select_template(stack))
    phys = generate_physics(stack, auth.parameters, meshes=meshes)
    return assemble_rig(name="h", source=None, stack=stack, meshes=meshes, deformers=auth.deformers,
                        parameters=auth.parameters, physics=phys, archetype="portrait_front")


def _span(offsets):
    xs = [dx for dx, _ in offsets]
    ys = [dy for _, dy in offsets]
    return max(xs) - min(xs), max(ys) - min(ys)


def test_each_hair_role_gets_a_vertical_bounce_param():
    rig = _hair_rig()
    ids = {p.id for p in rig.parameters}
    assert "ParamHairFrontV" in ids and "ParamHairBackV" in ids
    assert "ParamHairSideV" not in ids                              # no hair_side part -> no param


def test_bounce_keyform_is_vertical_sway_keyform_is_horizontal():
    """The whole point: the bounce moves the hair in Y (down), the sway moves it in X. Otherwise a nod
    and a turn would drive the same deformation."""
    rig = _hair_rig()
    bounce = next(p for p in rig.parameters if p.id == "ParamHairFrontV")
    sway = next(p for p in rig.parameters if p.id == "ParamHairFront")
    kf_b = max(bounce.keyforms, key=lambda k: k.value)
    kf_s = max(sway.keyforms, key=lambda k: k.value)
    offs_b = next(iter(kf_b.mesh_offsets.values()))
    offs_s = next(iter(kf_s.mesh_offsets.values()))
    bx, by = _span(offs_b)
    sx, sy = _span(offs_s)
    assert by > 1e-4 and by > 5 * bx        # bounce is (near-)pure vertical
    assert sx > 1e-4 and sx > 5 * sy        # sway is (near-)pure horizontal


def test_bounce_chain_is_pitch_driven():
    rig = _hair_rig()
    bounce = [ph for ph in rig.physics if ph.output_param.endswith("V")]
    assert bounce, "no bounce chain generated"
    for ph in bounce:
        assert ph.driver_param == "ParamAngleY" and ph.pitch_angle is True


def test_sway_chain_drops_pitch():
    """Pitch must NOT be a driver on the horizontal sway chains — it is inert there and only bloated the
    setting (and, fed as X/Angle, would swing the hair sideways on a nod)."""
    rig = _hair_rig()
    for ph in rig.physics:
        if ph.output_param in ("ParamHairFront", "ParamHairBack"):
            assert "ParamAngleY" not in ph.all_drivers()


def test_cubism_emits_pitch_as_angle_for_bounce_only():
    """physics3: the bounce chain feeds ParamAngleY as an Angle input (tips gravity), while the sway
    chain drops it entirely (a Y input is a no-op for an angle output)."""
    rig = _hair_rig()
    by_out = {s["Output"][0]["Destination"]["Id"]: s for s in physics3(rig)["PhysicsSettings"]}
    b = by_out["ParamHairFrontV"]
    types = {i["Source"]["Id"]: i["Type"] for i in b["Input"]}
    assert types.get("ParamAngleY") == "Angle"
    s = by_out["ParamHairFront"]
    assert "ParamAngleY" not in {i["Source"]["Id"] for i in s["Input"]}
