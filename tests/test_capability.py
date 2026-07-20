"""Rig capability report (core.qa.capability).

An honest, render-free summary of what a finished puppet does: which axes articulate, whether the mouth
can open, how many physics chains are live, and which clips were dropped for missing parts. These pin
that a degraded rig (no arms, a lip line with no interior) is *reported* as such rather than shipping a
confident-looking dud, and that an emitter-synthesised head turn is NOT mistaken for a dead axis.
"""

from __future__ import annotations

from pathlib import Path

from image2live2d.core.assemble import assemble_rig
from image2live2d.core.mesh import grid_mesh
from image2live2d.core.motion import generate_all
from image2live2d.core.physics import generate_physics
from image2live2d.core.qa.capability import rig_capabilities
from image2live2d.core.rig import author_rig, select_template
from image2live2d.core.types import Layer, LayerStack
from image2live2d.irr.schema import SemanticRole as R


def _rig(parts):
    layers, meshes = [], []
    for i, (pid, role, rect) in enumerate(parts):
        layers.append(Layer(id=pid, semantic_role=role, texture_path=Path(f"{pid}.png"),
                            draw_order=i * 10, width=64, height=64))
        meshes.append(grid_mesh(pid, rect, lambda u, v: 255, grid=2))
    stack = LayerStack(layers=layers, canvas_width=64, canvas_height=64)
    auth = author_rig(stack, meshes, select_template(stack))
    phys = generate_physics(stack, auth.parameters)
    anims = generate_all(auth.parameters, phys)
    return assemble_rig(name="t", source=None, stack=stack, meshes=meshes, deformers=auth.deformers,
                        parameters=auth.parameters, physics=phys, archetype="fullbody",
                        animations=anims)


# A full-body figure with both arms and both legs separated.
_FULL = [("face_base", R.face_base, (0.30, 0.55, 0.70, 0.80)),
         ("eye_l", R.eye_l, (0.36, 0.66, 0.46, 0.72)),
         ("eye_r", R.eye_r, (0.54, 0.66, 0.64, 0.72)),
         ("mouth", R.mouth, (0.45, 0.58, 0.55, 0.61)),
         ("mouth_cavity", R.mouth_cavity, (0.46, 0.575, 0.54, 0.605)),
         ("arm_l", R.arm_l, (0.20, 0.30, 0.30, 0.55)),
         ("arm_r", R.arm_r, (0.70, 0.30, 0.80, 0.55)),
         ("leg_l", R.leg_l, (0.40, 0.05, 0.48, 0.30)),
         ("leg_r", R.leg_r, (0.52, 0.05, 0.60, 0.30))]

# A gowned figure: no arms or legs separated (fused into the dress), and a bare lip line with no cavity.
_GOWN = [("face_base", R.face_base, (0.30, 0.55, 0.70, 0.80)),
         ("eye_l", R.eye_l, (0.36, 0.66, 0.46, 0.72)),
         ("eye_r", R.eye_r, (0.54, 0.66, 0.64, 0.72)),
         ("mouth", R.mouth, (0.45, 0.58, 0.55, 0.61)),
         ("clothing", R.clothing, (0.20, 0.05, 0.80, 0.55))]


def test_full_body_reports_all_articulation():
    cap = rig_capabilities(_rig(_FULL))
    assert cap.has("arm_left") and cap.has("arm_right")
    assert cap.has("leg_left") and cap.has("leg_right")
    assert cap.has("blink_left") and cap.has("blink_right")
    assert cap.has("mouth_open")            # has a cavity -> can truly open
    assert not cap.notes                    # nothing degraded to surface


def test_head_turn_is_reported_present_despite_no_keyform_offsets():
    """The head turn is synthesised by the emitters, so ParamAngleX/Y/Z carry no IRR offsets. It must
    still read as a live capability — testing mesh offsets here would call a working turn dead."""
    cap = rig_capabilities(_rig(_FULL))
    assert cap.has("head_turn")


def test_gown_reports_missing_arms_and_legs():
    cap = rig_capabilities(_rig(_GOWN))
    assert not cap.has("arm_left") and not cap.has("arm_right")
    assert not cap.has("leg_left") and not cap.has("leg_right")
    assert any("no arm articulation" in n for n in cap.notes)
    assert any("no leg articulation" in n for n in cap.notes)


def test_lip_line_without_interior_cannot_open():
    cap = rig_capabilities(_rig(_GOWN))
    assert not cap.has("mouth_open")        # a mouth param exists, but no cavity to open into
    assert any("mouth cannot open" in n for n in cap.notes)


def test_clips_for_missing_limbs_are_reported_suppressed():
    """The arm drive clips are dropped when a character has no arms; the report names that gap."""
    cap = rig_capabilities(_rig(_GOWN))
    assert "arms_raise" in cap.clips_suppressed
    assert "arms_swing" in cap.clips_suppressed
    # a character WITH arms keeps the arm clips
    full = rig_capabilities(_rig(_FULL))
    assert "arms_raise" not in full.clips_suppressed
    assert "arms_swing" not in full.clips_suppressed


def test_report_serialises_to_dict():
    d = rig_capabilities(_rig(_FULL)).to_dict()
    assert d["capabilities"]["arm_left"] is True
    assert set(d) == {"name", "parts", "params", "capabilities", "physics", "clips", "notes"}
