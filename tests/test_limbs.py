"""Limb de-cardboarding: See-through bundles both arms in one layer and both legs+socks in another,
so they can't move independently and read as a rigid board. The decomposer now splits each bundle
L/R (geometrically), the landmark extractor emits shoulder/elbow/wrist (hip/knee/ankle) joints, and
the rig authors a whole-limb swing + a gap-free lower-segment bend about the elbow/knee."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PIL")
pytest.importorskip("numpy")

from PIL import Image

from image2live2d.core.decompose.sources import _limb_split
from image2live2d.core.landmark import Landmarks
from image2live2d.core.mesh import grid_mesh
from image2live2d.core.rig import author_rig, select_template
from image2live2d.core.types import Layer, LayerStack
from image2live2d.irr.schema import SemanticRole as R


def _canvas(blobs, size=200):
    """Full-canvas RGBA with opaque rectangles at ``blobs`` = [(x0,y0,x1,y1)] in pixel coords."""
    from PIL import ImageDraw
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for b in blobs:
        d.rectangle(b, fill=(180, 150, 140, 255))
    return im


# --------------------------------------------------------------------------------------------------
# _limb_split geometric detection
# --------------------------------------------------------------------------------------------------
def test_split_arms_two_side_blobs():
    # two tall blobs out to the sides, upper body (not reaching the floor) -> arms
    im = _canvas([(40, 60, 75, 150), (125, 60, 160, 150)], size=200)
    out = _limb_split(im)
    assert out is not None
    lrole, rrole, limg, rimg = out
    assert {lrole, rrole} == {R.arm_l, R.arm_r}


def test_split_legs_two_central_blobs_reaching_floor():
    # two central vertical blobs descending to the bottom edge -> legs
    im = _canvas([(82, 100, 97, 198), (103, 100, 118, 198)], size=200)
    out = _limb_split(im)
    assert out is not None
    lrole, rrole, _, _ = out
    assert {lrole, rrole} == {R.leg_l, R.leg_r}


def test_split_single_blob_is_not_a_limb():
    # one connected blob (a torso/skirt/top) is never a limb bundle
    assert _limb_split(_canvas([(70, 60, 130, 150)])) is None


def test_split_two_tiny_blobs_are_not_limbs():
    # two small high blobs (e.g. earrings) are too short vertically to be limbs
    assert _limb_split(_canvas([(60, 30, 70, 45), (130, 30, 140, 45)])) is None


def test_left_right_assignment_by_screen_side():
    # screen-left blob = character's own right (_r); screen-right = character's left (_l)
    im = _canvas([(40, 60, 75, 150), (125, 60, 160, 150)], size=200)
    lrole, rrole, limg, rimg = _limb_split(im)
    assert lrole == R.arm_r and rrole == R.arm_l          # first tuple element is the screen-left cut
    import numpy as np
    left_has = (np.array(limg)[:, :, 3] > 0).any(axis=0)
    assert left_has[:100].any() and not left_has[100:].any()  # screen-left image keeps only left blob


# --------------------------------------------------------------------------------------------------
# rig: swing + elbow/knee bend authored from joints, isolated per limb, gap-free
# --------------------------------------------------------------------------------------------------
def _limb_stack():
    parts = [
        ("torso", R.torso, (0.40, 0.45, 0.60, 0.70)),
        ("arm_l", R.arm_l, (0.60, 0.40, 0.70, 0.72)),
        ("arm_r", R.arm_r, (0.30, 0.40, 0.40, 0.72)),
        ("leg_l", R.leg_l, (0.50, 0.02, 0.58, 0.45)),
        ("leg_r", R.leg_r, (0.42, 0.02, 0.50, 0.45)),
    ]
    layers, meshes = [], []
    for i, (pid, role, rect) in enumerate(parts):
        layers.append(Layer(id=pid, semantic_role=role, texture_path=Path(f"{pid}.png"),
                            draw_order=i * 10, width=64, height=64))
        meshes.append(grid_mesh(pid, rect, lambda u, v: 255, grid=3))
    return LayerStack(layers=layers, canvas_width=64, canvas_height=64), meshes


def _joints():
    # shoulder/hip (top), elbow/knee (mid), wrist/ankle (end) for each limb
    j = {}
    for role, top, mid, end, x in (("arm_l", 0.72, 0.56, 0.40, 0.65), ("arm_r", 0.72, 0.56, 0.40, 0.35),
                                   ("leg_l", 0.45, 0.24, 0.02, 0.54), ("leg_r", 0.45, 0.24, 0.02, 0.46)):
        j[role] = (x, top)
        j[f"{role}_mid"] = (x, mid)
        j[f"{role}_end"] = (x, end)
    return j


def test_limb_swing_and_bend_params_authored():
    stack, meshes = _limb_stack()
    auth = author_rig(stack, meshes, select_template(stack), landmarks=Landmarks(joints=_joints()))
    ids = {p.id for p in auth.parameters}
    assert {"ParamArmLA", "ParamArmRA", "ParamLegLA", "ParamLegRA"} <= ids   # whole-limb swing
    assert {"ParamArmLB", "ParamArmRB", "ParamLegLB", "ParamLegRB"} <= ids   # elbow/knee bend


def test_bend_only_moves_its_own_limb_and_lower_segment():
    stack, meshes = _limb_stack()
    auth = author_rig(stack, meshes, select_template(stack), landmarks=Landmarks(joints=_joints()))
    bend = next(p for p in auth.parameters if p.id == "ParamArmLB")
    kf = next(k for k in bend.keyforms if k.value == bend.max)
    # only arm_l is moved
    moved = {pid for pid, offs in kf.mesh_offsets.items() if any(dx or dy for dx, dy in offs)}
    assert moved == {"arm_l"}
    # within arm_l, vertices at/above the elbow stay put; below the elbow move (gap-free ramp)
    mesh = next(m for m in meshes if m.part_id == "arm_l")
    offs = kf.mesh_offsets["arm_l"]
    top = [abs(offs[i][0]) + abs(offs[i][1]) for i, (_, y) in enumerate(mesh.vertices) if y > 0.60]
    bot = [abs(offs[i][0]) + abs(offs[i][1]) for i, (_, y) in enumerate(mesh.vertices) if y < 0.45]
    assert max(top) < 1e-9 and max(bot) > 1e-3


def test_limb_params_exempt_from_deform_cap():
    # the generic backstop clamps ordinary params but must leave limb swing/bend alone (a long limb's
    # wrist can legitimately travel past the cap; each limb is bounded by its own degree limit instead)
    from image2live2d.core.rig.author import _cap_offsets
    from image2live2d.irr.schema import Keyform
    from image2live2d.irr.params import make_parameter

    def _param(pid):
        p = make_parameter(pid)
        p.keyforms = [Keyform(value=p.max, mesh_offsets={"x": [(0.9, 0.0)]})]  # way over cap
        return p

    limb, other = _param("ParamArmLB"), _param("ParamHairFront")
    _cap_offsets([limb, other], 0.28)
    assert limb.keyforms[0].mesh_offsets["x"][0][0] == pytest.approx(0.9)   # limb untouched
    assert other.keyforms[0].mesh_offsets["x"][0][0] == pytest.approx(0.28)  # ordinary param clamped


def test_a_shoe_rides_the_leg_it_sits_under():
    """The second half of the leg-disconnect: a shoe is a separate part, so leg articulation used to
    move the leg and leave the shoe on the floor. A part at a limb's distal end now moves with it."""
    parts = [
        ("leg_l", R.leg_l, (0.50, 0.10, 0.58, 0.45)),
        ("leg_r", R.leg_r, (0.42, 0.10, 0.50, 0.45)),
        ("shoe_l", R.clothing, (0.49, 0.02, 0.59, 0.12)),   # under leg_l's foot
        ("shoe_r", R.clothing, (0.41, 0.02, 0.51, 0.12)),   # under leg_r's foot
        ("skirt", R.clothing, (0.38, 0.45, 0.62, 0.60)),    # well above the legs — must NOT ride them
    ]
    layers, meshes = [], []
    for i, (pid, role, rect) in enumerate(parts):
        layers.append(Layer(id=pid, semantic_role=role, texture_path=Path(f"{pid}.png"),
                            draw_order=i * 10, width=64, height=64))
        meshes.append(grid_mesh(pid, rect, lambda u, v: 255, grid=3))
    stack = LayerStack(layers=layers, canvas_width=64, canvas_height=64)

    auth = author_rig(stack, meshes, select_template(stack), landmarks=None)
    swing = next(p for p in auth.parameters if p.id == "ParamLegLA")
    kf = next(k for k in swing.keyforms if k.value == swing.max)
    moved = {pid for pid, offs in kf.mesh_offsets.items() if any(dx or dy for dx, dy in offs)}
    assert "shoe_l" in moved       # the shoe swings with its leg
    assert "shoe_r" not in moved   # the other leg's shoe does not
    assert "skirt" not in moved    # and the skirt, far above the feet, is untouched


def test_limbs_authored_from_mesh_without_landmark_joints():
    # Limbs no longer depend on landmark joints: the joint is derived from each limb's own split mesh
    # (the landmark silhouette is of both limbs at once — see author._limb_joints). So articulation is
    # authored whether landmarks carry joints, are empty, or are absent entirely.
    stack, meshes = _limb_stack()
    for lm in (Landmarks(joints={}), None):
        auth = author_rig(stack, meshes, select_template(stack), landmarks=lm)
        ids = {p.id for p in auth.parameters}
        assert {"ParamArmLA", "ParamArmLB", "ParamLegLA", "ParamLegRA"} <= ids
