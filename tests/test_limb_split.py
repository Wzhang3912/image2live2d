"""Splitting parts that are really a mirrored pair (core.structure.limbs).

A decomposer returns what is visually contiguous, not what is anatomically separate. On a real
character it handed back both arms as one layer labelled ``accessory``, both eyebrows as one layer
labelled ``eyebrow_l``, and both ears as ``ear_l``. Each breaks the rig differently: arms in one mesh
can only move as a rigid sheet (they read as cardboard), and a missing ``eyebrow_r`` leaves
``ParamBrowRY`` driving nothing at all.
"""

from __future__ import annotations

from pathlib import Path

from image2live2d.core.structure import split_bundled_pairs
from image2live2d.core.types import Layer, LayerStack
from image2live2d.irr.schema import Mesh
from image2live2d.irr.schema import SemanticRole as R


def _lobes(pid: str, boxes) -> Mesh:
    """A mesh made of one quad per box — disconnected boxes become separate components."""
    verts, uvs, tris = [], [], []
    for x0, y0, x1, y1 in boxes:
        b = len(verts)
        verts += [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        uvs += [(0.0, 0.0)] * 4
        tris += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
    return Mesh(part_id=pid, vertices=verts, uvs=uvs, triangles=tris)


def _scene(parts):
    """parts: list of (id, role, [boxes])."""
    layers, meshes = [], []
    for i, (pid, role, boxes) in enumerate(parts):
        layers.append(Layer(id=pid, semantic_role=role, texture_path=Path(f"{pid}.png"),
                            draw_order=i, width=64, height=64))
        meshes.append(_lobes(pid, boxes))
    return LayerStack(layers=layers, canvas_width=64, canvas_height=64), meshes


def _roles(stack):
    return {ly.id: ly.semantic_role for ly in stack.layers}


# --- rule 1: a one-sided role holding both sides is the pair ---------------------------------------
def test_both_brows_labelled_eyebrow_l_become_a_left_and_a_right():
    """Real geometry: the decomposer put BOTH brows in one layer called `eyebrow_l`, so `eyebrow_r`
    never existed and ParamBrowRY drove nothing."""
    stack, meshes = _scene([
        ("face", R.face_base, [(0.44, 0.81, 0.54, 0.93)]),
        ("brows", R.eyebrow_l, [(0.45, 0.88, 0.48, 0.89), (0.50, 0.88, 0.53, 0.89)]),
    ])
    created = split_bundled_pairs(stack, meshes)
    roles = _roles(stack)
    assert set(created) == {"01_eyebrow_l", "01_eyebrow_r"}
    assert roles["01_eyebrow_l"] is R.eyebrow_l
    assert roles["01_eyebrow_r"] is R.eyebrow_r
    assert "brows" not in roles                       # the bundle is gone, replaced by its halves
    # each half carries only its own lobe
    by_id = {m.part_id: m for m in meshes}
    assert max(x for x, _ in by_id["01_eyebrow_l"].vertices) <= 0.48
    assert min(x for x, _ in by_id["01_eyebrow_r"].vertices) >= 0.50


def test_a_twin_that_already_exists_is_not_stolen():
    """If eyebrow_r is a real separate part, an eyebrow_l with two lobes is NOT the pair — never invent
    a second right brow on top of the one that exists. It still splits (rule 3), but keeps its role."""
    stack, meshes = _scene([
        ("face", R.face_base, [(0.40, 0.81, 0.60, 0.93)]),
        # two lobes that do straddle the midline, so only the existing twin stops rule 1 firing
        ("browl", R.eyebrow_l, [(0.42, 0.88, 0.46, 0.89), (0.54, 0.88, 0.58, 0.89)]),
        ("browr", R.eyebrow_r, [(0.47, 0.85, 0.53, 0.86)]),
    ])
    split_bundled_pairs(stack, meshes)
    assert sorted(r.value for r in _roles(stack).values()) == ["eyebrow_l", "eyebrow_l", "eyebrow_r",
                                                              "face_base"]


# --- rule 2: the junk-drawer arms -----------------------------------------------------------------
def _body_with_arms(arm_role=R.accessory):
    return _scene([
        ("face", R.face_base, [(0.44, 0.81, 0.54, 0.93)]),
        ("neck", R.neck, [(0.47, 0.79, 0.51, 0.85)]),
        ("torso", R.clothing, [(0.43, 0.64, 0.55, 0.81)]),
        ("legs", R.clothing, [(0.41, 0.13, 0.58, 0.54)]),
        ("arms", arm_role, [(0.28, 0.49, 0.46, 0.78), (0.52, 0.49, 0.70, 0.78)]),
    ])


def test_a_wide_lateral_pair_in_the_junk_drawer_is_the_arms():
    """Real geometry: both arms arrived as one `accessory`. As one mesh they can only move as a rigid
    sheet — a left and a right arm swing oppositely about different shoulders."""
    stack, meshes = _body_with_arms()
    created = split_bundled_pairs(stack, meshes)
    roles = _roles(stack)
    assert set(created) >= {"04_arm_l", "04_arm_r"}
    assert roles["04_arm_l"] is R.arm_l and roles["04_arm_r"] is R.arm_r


def test_earrings_are_not_mistaken_for_arms():
    """A mirrored pair inside the head's column is jewellery, not a limb. A false positive here would
    give an earring shoulder and elbow articulation."""
    stack, meshes = _scene([
        ("face", R.face_base, [(0.44, 0.81, 0.54, 0.93)]),
        ("neck", R.neck, [(0.47, 0.79, 0.51, 0.85)]),
        ("torso", R.clothing, [(0.43, 0.64, 0.55, 0.81)]),
        ("earrings", R.accessory, [(0.45, 0.82, 0.46, 0.84), (0.53, 0.82, 0.54, 0.84)]),
    ])
    split_bundled_pairs(stack, meshes)
    assert all(r is not R.arm_l and r is not R.arm_r for r in _roles(stack).values())


# --- rule 2b: the arms of a character who is not standing to attention ------------------------------
# Every scene below uses the real model-space geometry of `wikipetan_mop` (a character mopping a floor:
# one arm raised to the mop handle, the other down, a mop spanning most of the canvas). Three separate
# rules each assumed the arms-at-sides pose that all eight of the original test characters happen to
# share, and they failed in series, so each of these tests fails on its own before its fix.
# head box (face+neck union) and body box match the measured ones: (0.452, 0.691, 0.623, 0.902) and
# (0.097, 0.012, 0.909, 0.975). Both matter — the shoulder ceiling is measured from the head's base and
# scaled by the body's height, so a fixture with no legs puts the ceiling below the raised arm.
_MOP_BODY = [
    ("face", R.face_base, [(0.452, 0.691, 0.623, 0.902)]),
    ("neck", R.neck, [(0.50, 0.691, 0.57, 0.74)]),
    ("torso", R.clothing, [(0.44, 0.30, 0.66, 0.69)]),
    ("legs", R.clothing, [(0.44, 0.012, 0.66, 0.30)]),
]


def test_a_held_prop_does_not_drag_the_midline_off_the_body():
    """The midline sorts the lobes into a left and a right, so it must be the *character's* centre. Taken
    from the union of every part it is not: the mop spans x 0.10-0.91 and pulls it to 0.503 while the
    body sits right of it, so both arms (cx 0.519 and 0.743) land on the same side and the pair is
    rejected as "not a left and a right". The head cannot be pushed sideways by something she holds."""
    stack, meshes = _scene([
        *_MOP_BODY,
        ("mop", R.other, [(0.097, 0.012, 0.909, 0.30)]),
        ("arms", R.accessory, [(0.466, 0.386, 0.572, 0.688), (0.651, 0.578, 0.835, 0.770)]),
    ])
    split_bundled_pairs(stack, meshes)
    assert set(_roles(stack).values()) >= {R.arm_l, R.arm_r}


def test_arms_at_different_heights_are_still_a_pair():
    """One arm raised to the mop handle, one down: the centroids differ by 0.137, well past the 0.10 that
    keeps *facial* twins level. Eyes and brows are level by anatomy; limbs are not, and a pose moves one
    without the other. Facial twins must keep the strict tolerance, so this loosens only for limbs."""
    stack, meshes = _scene([
        *_MOP_BODY,
        ("arms", R.accessory, [(0.466, 0.386, 0.572, 0.688), (0.651, 0.578, 0.835, 0.770)]),
    ])
    split_bundled_pairs(stack, meshes)
    assert set(_roles(stack).values()) >= {R.arm_l, R.arm_r}


def test_an_arm_in_front_of_the_body_is_not_jewellery():
    """Arms folded across the chest sit inside the head's *column* — but below the chin, where that
    column is the torso's, not the head's. The earring rule tested the column alone, so a raised forearm
    (x 0.466-0.572, inside the head column 0.452-0.623) read as an earring and the character shipped with
    no arms. Jewellery is beside the head; these lobes are entirely below it."""
    stack, meshes = _scene([
        *_MOP_BODY,
        ("arms", R.accessory, [(0.46, 0.36, 0.52, 0.66), (0.55, 0.36, 0.61, 0.66)]),
    ])
    split_bundled_pairs(stack, meshes)
    assert set(_roles(stack).values()) >= {R.arm_l, R.arm_r}


def test_headwear_is_not_mistaken_for_arms():
    """Regression (backlog T10): a wide-brimmed hat clears every other arm test — its two lobes are a
    mirrored pair, they sit outside the head's narrow column, they are tall enough, and they are nowhere
    near the waist. The rule had a floor but no ceiling, so `blondedrills`' headwear was split into
    arm_l/arm_r up at the top of the canvas (y 0.84-0.99) and the real arms then had to share a shoulder
    pivot with a hat, dragging it 0.2 body-heights above where a shoulder is. An arm attaches AT the
    shoulder, so it cannot start above one."""
    stack, meshes = _scene([
        ("face", R.face_base, [(0.44, 0.81, 0.54, 0.93)]),
        ("neck", R.neck, [(0.47, 0.79, 0.51, 0.85)]),
        ("torso", R.clothing, [(0.43, 0.64, 0.55, 0.81)]),
        ("legs", R.clothing, [(0.41, 0.13, 0.58, 0.54)]),
        # the brim: a mirrored pair, outside the head's column, tall, and well above the waist
        ("hat", R.accessory, [(0.30, 0.84, 0.44, 0.99), (0.54, 0.84, 0.68, 0.99)]),
    ])
    split_bundled_pairs(stack, meshes)
    assert all(r is not R.arm_l and r is not R.arm_r for r in _roles(stack).values())


def test_shoes_are_not_mistaken_for_arms():
    """A mirrored pair down at the feet is footwear. Arms hang from the shoulders."""
    stack, meshes = _scene([
        ("face", R.face_base, [(0.44, 0.81, 0.54, 0.93)]),
        ("neck", R.neck, [(0.47, 0.79, 0.51, 0.85)]),
        ("shoes", R.clothing, [(0.44, 0.03, 0.48, 0.17), (0.50, 0.03, 0.55, 0.17)]),
    ])
    split_bundled_pairs(stack, meshes)
    assert all(r is not R.arm_l and r is not R.arm_r for r in _roles(stack).values())


# --- rule 3: everything else just stops being one rigid sheet --------------------------------------
def test_a_paired_ornament_splits_but_keeps_its_role():
    """Two earrings in one part share one pendulum and swing in lockstep. Split, they each get their
    own mesh and their own dangle — but they are still accessories."""
    stack, meshes = _scene([
        ("face", R.face_base, [(0.44, 0.81, 0.54, 0.93)]),
        ("neck", R.neck, [(0.47, 0.79, 0.51, 0.85)]),
        ("earrings", R.accessory, [(0.45, 0.82, 0.46, 0.84), (0.53, 0.82, 0.54, 0.84)]),
    ])
    created = split_bundled_pairs(stack, meshes)
    assert set(created) == {"02_accessory_l", "02_accessory_r"}
    assert all(_roles(stack)[c] is R.accessory for c in created)


# --- guards ---------------------------------------------------------------------------------------
def test_a_single_blob_is_left_alone():
    stack, meshes = _scene([
        ("face", R.face_base, [(0.44, 0.81, 0.54, 0.93)]),
        ("torso", R.clothing, [(0.43, 0.64, 0.55, 0.81)]),
    ])
    assert split_bundled_pairs(stack, meshes) == []
    assert len(stack.layers) == 2


def test_a_speckle_beside_a_blob_is_not_a_pair():
    """Two components are not automatically a left and a right — a decomposition artefact next to a
    real part must not spawn a phantom limb."""
    stack, meshes = _scene([
        ("face", R.face_base, [(0.44, 0.81, 0.54, 0.93)]),
        ("neck", R.neck, [(0.47, 0.79, 0.51, 0.85)]),
        # a big left blob and a one-quad speckle on the right: sizes are nowhere near comparable
        ("thing", R.accessory, [(0.28, 0.49, 0.46, 0.78)]),
    ])
    assert split_bundled_pairs(stack, meshes) == []


def test_two_lobes_on_the_same_side_are_not_a_pair():
    stack, meshes = _scene([
        ("face", R.face_base, [(0.44, 0.81, 0.54, 0.93)]),
        ("neck", R.neck, [(0.47, 0.79, 0.51, 0.85)]),
        ("both_left", R.accessory, [(0.10, 0.49, 0.20, 0.78), (0.25, 0.49, 0.35, 0.78)]),
    ])
    split_bundled_pairs(stack, meshes)
    assert all(r is not R.arm_l for r in _roles(stack).values())


# --- fused legs: connected components can't help, so cut along the crotch seam ---------------------
def _legs_mesh(pid="legs", *, crotch_v=0.35, gap=(0.44, 0.56), rect=(0.40, 0.10, 0.60, 0.55)):
    """Two legs fused at the hips: solid above the crotch, with a gap between them below it.

    ``grid_mesh`` drops the transparent cells, so the gap becomes a real hole in the lattice — which is
    exactly the handle the seam finder uses. ``v`` runs top->bottom.
    """
    from image2live2d.core.mesh import grid_mesh

    def alpha(u, v):
        return 0 if v > crotch_v and gap[0] < u < gap[1] else 255

    return grid_mesh(pid, rect, alpha, grid=16)


def _legs_scene(mesh, role=R.clothing):
    layers = [
        Layer(id="face", semantic_role=R.face_base, texture_path=Path("f.png"),
              draw_order=0, width=64, height=64),
        Layer(id=mesh.part_id, semantic_role=role, texture_path=Path("l.png"),
              draw_order=1, width=64, height=64),
    ]
    meshes = [_lobes("face", [(0.45, 0.80, 0.55, 0.95)]), mesh]
    return LayerStack(layers=layers, canvas_width=64, canvas_height=64), meshes


def test_legs_fused_at_the_hips_are_cut_into_a_left_and_a_right():
    """Connected components cannot separate the legs — the thighs touch, so both legs are one blob.
    The gap that opens below the crotch is the handle."""
    from image2live2d.core.structure import split_bundled_pairs, split_fused_legs

    mesh = _legs_mesh()
    assert len(split_bundled_pairs(*_legs_scene(mesh))) == 0     # components genuinely can't do it

    stack, meshes = _legs_scene(_legs_mesh())
    created = split_fused_legs(stack, meshes)
    assert set(created) == {"01_leg_l", "01_leg_r"}
    roles = _roles(stack)
    assert roles["01_leg_l"] is R.leg_l and roles["01_leg_r"] is R.leg_r


def test_the_cut_loses_no_triangles():
    """Every triangle goes wholly to one side, so no hole can open along the seam."""
    from image2live2d.core.structure import split_fused_legs

    original = _legs_mesh()
    stack, meshes = _legs_scene(_legs_mesh())
    split_fused_legs(stack, meshes)
    by = {m.part_id: m for m in meshes}
    assert len(by["01_leg_l"].triangles) + len(by["01_leg_r"].triangles) == len(original.triangles)
    # ...and the halves land on opposite sides of the seam
    assert max(x for x, _ in by["01_leg_l"].vertices) <= min(x for x, _ in by["01_leg_r"].vertices)


def test_a_solid_skirt_is_not_cut():
    """A skirt has no crotch. Cutting one in half would give each side its own hip and knee."""
    from image2live2d.core.mesh import grid_mesh
    from image2live2d.core.structure import split_fused_legs

    skirt = grid_mesh("skirt", (0.36, 0.30, 0.64, 0.55), lambda u, v: 255, grid=16)
    stack, meshes = _legs_scene(skirt)
    assert split_fused_legs(stack, meshes) == []


def test_an_off_centre_slit_is_not_a_crotch():
    """A gap away from the body's midline is a slit or a fold in a garment, not the space between legs."""
    from image2live2d.core.structure import split_fused_legs

    stack, meshes = _legs_scene(_legs_mesh(gap=(0.70, 0.80)))
    assert split_fused_legs(stack, meshes) == []


def test_a_real_right_leg_stops_the_cut():
    """If leg_r already exists the part is not both legs — never invent a second one."""
    from image2live2d.core.structure import split_fused_legs

    stack, meshes = _legs_scene(_legs_mesh())
    stack.layers.append(Layer(id="realleg", semantic_role=R.leg_r, texture_path=Path("r.png"),
                              draw_order=2, width=64, height=64))
    meshes.append(_lobes("realleg", [(0.60, 0.10, 0.66, 0.50)]))
    assert split_fused_legs(stack, meshes) == []


# --- arms the decomposer mislabelled as legs -------------------------------------------------------
def _armleg_scene(limb_boxes, role=R.leg_r):
    """A head, a lower body reaching the feet, and one limb — so head_box/body_box (and the feet
    reference) are all defined. The limb at `limb_boxes` is labelled a leg."""
    return _scene([
        ("00_face_base", R.face_base, [(0.44, 0.80, 0.54, 0.95)]),   # head near the top (y-up)
        ("02_clothing", R.clothing, [(0.40, 0.02, 0.60, 0.50)]),     # torso->feet, sets the body bottom
        ("01_leg", role, limb_boxes),
    ])


def test_a_leg_labelled_part_at_the_shoulder_becomes_an_arm():
    """See-through labels a slim character's arms `leg_l`/`leg_r`. An arm attaches at the shoulder (top
    at the head's base) and stops mid-body (never reaches the feet), so it must be re-roled to an arm."""
    from image2live2d.core.structure import reassign_arm_mislabeled_as_leg

    # slender limb: top 0.79 (just below head base 0.80), bottom 0.45 (mid-body), taller than wide
    stack, meshes = _armleg_scene([(0.30, 0.45, 0.40, 0.79)], role=R.leg_r)
    changed = reassign_arm_mislabeled_as_leg(stack, meshes)
    assert changed == ["01_leg"]
    assert _roles(stack)["01_leg"] is R.arm_r          # side preserved: leg_r -> arm_r


def test_a_real_leg_stays_a_leg():
    """A leg reaches the floor; it must not be re-roled to an arm."""
    from image2live2d.core.structure import reassign_arm_mislabeled_as_leg

    # limb runs from the hip (0.45) down to the feet (0.04), near the midline
    stack, meshes = _armleg_scene([(0.44, 0.04, 0.52, 0.45)], role=R.leg_r)
    assert reassign_arm_mislabeled_as_leg(stack, meshes) == []
    assert _roles(stack)["01_leg"] is R.leg_r


def test_a_limb_that_rises_above_the_head_is_not_an_arm():
    """Drill-hair a decomposer also mislabels `leg` rises past the crown. An arm never rises above the
    head, so the shoulder test rejects it — it is left a leg rather than given arm articulation."""
    from image2live2d.core.structure import reassign_arm_mislabeled_as_leg

    # top 0.98 sits above the head top (0.95)
    stack, meshes = _armleg_scene([(0.30, 0.82, 0.42, 0.98)], role=R.leg_r)
    assert reassign_arm_mislabeled_as_leg(stack, meshes) == []
    assert _roles(stack)["01_leg"] is R.leg_r


def test_a_wide_garment_blob_is_not_an_arm():
    """An arm is slender. A wide low blob (a garment) mislabelled a leg must not gain arm articulation."""
    from image2live2d.core.structure import reassign_arm_mislabeled_as_leg

    # wider than tall, in the mid-body
    stack, meshes = _armleg_scene([(0.30, 0.55, 0.62, 0.72)], role=R.leg_l)
    assert reassign_arm_mislabeled_as_leg(stack, meshes) == []


def test_no_reassign_without_a_head():
    """With no face to place the shoulder line, we do not guess."""
    from image2live2d.core.structure import reassign_arm_mislabeled_as_leg

    stack, meshes = _scene([("01_leg", R.leg_r, [(0.30, 0.45, 0.40, 0.79)])])
    assert reassign_arm_mislabeled_as_leg(stack, meshes) == []
