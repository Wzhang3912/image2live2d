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


def test_limb_swing_is_mirror_symmetric():
    """+param must lift each arm OUTWARD — away from the midline — on both sides, so driving both arms
    to +max raises them symmetrically instead of lifting one and dropping the other. Before this, the
    two arms rotated the same absolute direction, and you could not raise both at once."""
    parts = [("torso", R.torso, (0.42, 0.45, 0.58, 0.70)),
             ("arm_l", R.arm_l, (0.28, 0.40, 0.40, 0.72)),   # screen-left arm
             ("arm_r", R.arm_r, (0.60, 0.40, 0.72, 0.72))]   # screen-right arm
    layers, meshes = [], []
    for i, (pid, role, rect) in enumerate(parts):
        layers.append(Layer(id=pid, semantic_role=role, texture_path=Path(f"{pid}.png"),
                            draw_order=i * 10, width=64, height=64))
        meshes.append(grid_mesh(pid, rect, lambda u, v: 255, grid=3))
    stack = LayerStack(layers=layers, canvas_width=64, canvas_height=64)
    params = author_rig(stack, meshes, select_template(stack), landmarks=None).parameters
    by_id = {p.id: p for p in params}
    mesh_by = {m.part_id: m for m in meshes}

    def hand_dx(part_id: str, param: str) -> float:
        """Mean x-shift of the arm's hand (its lowest row) when ``param`` is driven to +max."""
        m = mesh_by[part_id]
        bot = min(y for _, y in m.vertices)
        kf = next(k for k in by_id[param].keyforms if k.value == by_id[param].max)
        off = kf.mesh_offsets[part_id]
        dxs = [off[i][0] for i, (_, y) in enumerate(m.vertices) if y <= bot + 0.02]
        return sum(dxs) / len(dxs)

    left = hand_dx("arm_l", "ParamArmLA")     # screen-left arm, its own +swing
    right = hand_dx("arm_r", "ParamArmRA")    # screen-right arm, its own +swing
    assert left < 0 and right > 0, (left, right)   # each hand swings OUTWARD, opposite directions


def test_legs_swing_splays_the_feet_and_never_crosses():
    """The reported bug: two close-together legs swung in opposite phase rotate toward each other and
    cross into an X. The clip now splays them outward, and the swing is small enough that even the
    ping-pong's other extreme stays short of crossing. Property: the screen-left leg's foot never ends
    up right of the screen-right leg's foot."""
    from image2live2d.core.motion import generate_drives

    # legs spaced like a real standing character (feet ~0.09 of canvas apart) — the reduced swing has
    # to stay short of crossing even as the ping-pong swings them toward each other
    parts = [("leg_l", R.leg_l, (0.42, 0.05, 0.48, 0.45)),
             ("leg_r", R.leg_r, (0.52, 0.05, 0.58, 0.45))]
    layers, meshes = [], []
    for i, (pid, role, rect) in enumerate(parts):
        layers.append(Layer(id=pid, semantic_role=role, texture_path=Path(f"{pid}.png"),
                            draw_order=i * 10, width=64, height=64))
        meshes.append(grid_mesh(pid, rect, lambda u, v: 255, grid=3))
    stack = LayerStack(layers=layers, canvas_width=64, canvas_height=64)
    params = author_rig(stack, meshes, select_template(stack), landmarks=None).parameters
    by_id = {p.id: p for p in params}
    mesh_by = {m.part_id: m for m in meshes}

    # the signed fraction the legs_swing clip drives each param to (its keyed extreme is full-range;
    # the clip reaches |frac| of it, and the keyforms are linear from the default, so scale by |frac|)
    pose = next(a for a in generate_drives(params) if a.name == "legs_swing")
    fracs = {ln.param_id: max((kf.value for kf in ln.keyframes), key=abs) / 10.0 for ln in pose.lanes}

    def foot_x(part_id: str, flip: float) -> float:
        """Mean x of the part's lowest row (its foot) at the clip pose (flip=+1) or its opposite."""
        m = mesh_by[part_id]
        bot = min(y for _, y in m.vertices)
        off = [0.0] * len(m.vertices)
        for pid, frac in fracs.items():
            p = by_id.get(pid)
            if not p:
                continue
            val = p.max if frac * flip > 0 else p.min
            kf = next((k for k in p.keyforms if k.value == val), None)
            if kf and part_id in kf.mesh_offsets:
                for i, (dx, _dy) in enumerate(kf.mesh_offsets[part_id]):
                    off[i] += dx * abs(frac)                 # the clip reaches |frac| of the extreme
        foot = [x + off[i] for i, (x, y) in enumerate(m.vertices) if y <= bot + 0.02]
        return sum(foot) / len(foot)

    def rest_foot_x(part_id: str) -> float:
        m = mesh_by[part_id]
        bot = min(y for _, y in m.vertices)
        foot = [x for x, y in m.vertices if y <= bot + 0.02]
        return sum(foot) / len(foot)

    # the clip is one-directional (bidirectional=False): it reaches the splay pose and returns to
    # neutral, never the mirror (inward) pose. So check the pose it actually visits.
    l_splay, r_splay = foot_x("leg_l", 1.0), foot_x("leg_r", 1.0)
    l_rest, r_rest = rest_foot_x("leg_l"), rest_foot_x("leg_r")
    assert l_splay < r_splay, "feet crossed at the splay pose"                  # never an X
    assert (r_splay - l_splay) > (r_rest - l_rest), "feet did not splay apart"  # widened, not narrowed


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


# --------------------------------------------------------------------------------------------------
# Limb joints sit on the limb's silhouette axis, not its bbox centre (backlog T10)
# --------------------------------------------------------------------------------------------------
def _angled_arm_mesh():
    """An arm that attaches at a shoulder and hangs down and OUTWARD — the shape of a real arm, and the
    one the bbox centre gets wrong. Two stacked blocks: an upper arm at the shoulder, a forearm swung
    out to the side. The bbox spans both, so its x-centre lands in the notch between them."""
    from image2live2d.irr.schema import Mesh

    verts, uvs, tris = [], [], []
    for x0, y0, x1, y1 in ((0.40, 0.60, 0.48, 0.80),      # upper arm, at the shoulder
                           (0.48, 0.40, 0.72, 0.60)):     # forearm, swung out to the side
        b = len(verts)
        verts += [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        uvs += [(0.0, 0.0)] * 4
        tris += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
    return Mesh(part_id="arm_l", vertices=verts, uvs=uvs, triangles=tris)


def test_the_shoulder_sits_on_the_limbs_own_axis_not_its_bbox_centre():
    """Measured on the 8 real characters: with the bbox centre, 16 of 34 limb pivots landed OUTSIDE the
    limb they rotate (both arms on 7 of 8). A rotation about a point in empty space swings the limb wide
    instead of turning it in place. The cross-section centroid puts all 34 inside."""
    from image2live2d.core.rig.author import _limb_joints

    mesh = _angled_arm_mesh()
    shoulder, _, _ = _limb_joints([mesh])
    assert shoulder[1] == pytest.approx(0.80)                 # at the top of the limb
    assert 0.40 <= shoulder[0] <= 0.48, "shoulder must land on the upper arm, not out in the notch"
    bbox_centre_x = (0.40 + 0.72) / 2.0                       # 0.56 — inside neither block at that height
    assert not (0.40 <= bbox_centre_x <= 0.48), "the old pivot really did miss the limb"


def test_every_joint_follows_the_limb_where_it_bends():
    """The wrist/ankle anchors the parts that ride the limb's end (a shoe, a cuff), so it has to follow
    the limb out to where it actually ends rather than staying on the bbox's midline."""
    from image2live2d.core.rig.author import _limb_joints

    _, elbow, wrist = _limb_joints([_angled_arm_mesh()])
    assert wrist[1] == pytest.approx(0.40)                    # the bottom of the limb
    assert 0.48 <= wrist[0] <= 0.72, "wrist must sit on the forearm it ends in"
    assert elbow[1] == pytest.approx(0.60)                    # halfway down
