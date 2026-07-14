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
