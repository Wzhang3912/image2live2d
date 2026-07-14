"""Face z-order normalisation (core.structure.zorder).

The decomposer buried the eyebrows under ``face_base`` on a real character, so driving ``ParamBrowLY``
across its whole range changed zero pixels. These pin the two lift rules and, just as importantly,
that a correctly-ordered stack is left alone (the golden suite depends on it).
"""

from __future__ import annotations

from pathlib import Path

from image2live2d.core.structure import normalize_face_zorder
from image2live2d.core.types import Layer, LayerStack
from image2live2d.irr.schema import Mesh
from image2live2d.irr.schema import SemanticRole as R


def _mesh(pid, x0, y0, x1, y1) -> Mesh:
    verts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return Mesh(part_id=pid, vertices=verts, uvs=[(0.0, 0.0)] * 4, triangles=[(0, 1, 2), (0, 2, 3)])


def _scene(parts):
    """parts: list of (id, role, draw_order, (x0, y0, x1, y1))."""
    layers, meshes = [], []
    for pid, role, order, box in parts:
        layers.append(Layer(id=pid, semantic_role=role, texture_path=Path(f"{pid}.png"),
                            draw_order=order, width=64, height=64))
        meshes.append(_mesh(pid, *box))
    layers.sort(key=lambda ly: ly.draw_order)
    return LayerStack(layers=layers, canvas_width=64, canvas_height=64), meshes


def _order(stack):
    return {ly.id: ly.draw_order for ly in stack.layers}


def test_brow_buried_under_the_face_is_lifted_above_it():
    """Real geometry (char_fixed_v10): eyebrow drew at 7, face_base at 10 — under the skin, so the
    brow was invisible and ParamBrowLY moved zero pixels."""
    stack, meshes = _scene([
        ("brow", R.eyebrow_l, 7, (0.45, 0.88, 0.53, 0.89)),
        ("face", R.face_base, 10, (0.44, 0.81, 0.54, 0.93)),
        ("eye", R.eye_l, 13, (0.45, 0.85, 0.48, 0.88)),
    ])
    moved = normalize_face_zorder(stack, meshes)
    assert "brow" in moved
    assert _order(stack)["brow"] > _order(stack)["face"]


def test_brow_is_lifted_above_a_covering_fringe():
    """Hiyori (shipping commercial rig) draws its brow meshes at render order 130-131 while the bangs
    sit at 57/103 — brows read *through* the fringe, else every expression clip is dead weight."""
    stack, meshes = _scene([
        ("face", R.face_base, 10, (0.44, 0.81, 0.54, 0.93)),
        ("brow", R.eyebrow_l, 11, (0.45, 0.88, 0.53, 0.89)),
        ("bangs", R.hair_front, 20, (0.43, 0.81, 0.55, 0.98)),   # covers the brow entirely
    ])
    normalize_face_zorder(stack, meshes)
    assert _order(stack)["brow"] > _order(stack)["bangs"]


def test_a_fringe_that_does_not_cover_the_brow_leaves_it_alone():
    """Only *actual* occlusion lifts a part — side hair that misses the brow must not reorder it."""
    stack, meshes = _scene([
        ("face", R.face_base, 10, (0.44, 0.81, 0.54, 0.93)),
        ("brow", R.eyebrow_l, 11, (0.45, 0.88, 0.53, 0.89)),
        ("bangs", R.hair_front, 20, (0.20, 0.40, 0.30, 0.98)),   # off to the side, no overlap
    ])
    moved = normalize_face_zorder(stack, meshes)
    assert moved == []
    assert _order(stack)["brow"] == 11


def test_a_correctly_ordered_face_is_untouched():
    """The sparse/sample pipeline already stacks the face correctly; it must come out byte-identical."""
    stack, meshes = _scene([
        ("face", R.face_base, 10, (0.40, 0.60, 0.60, 0.90)),
        ("brow", R.eyebrow_l, 11, (0.43, 0.80, 0.50, 0.82)),
        ("eye", R.eye_l, 12, (0.43, 0.72, 0.48, 0.78)),
        ("mouth", R.mouth, 13, (0.47, 0.64, 0.53, 0.67)),
    ])
    before = _order(stack)
    assert normalize_face_zorder(stack, meshes) == []
    assert _order(stack) == before


def test_features_under_the_skin_all_surface():
    stack, meshes = _scene([
        ("mouth", R.mouth, 1, (0.47, 0.64, 0.53, 0.67)),
        ("eye", R.eye_l, 2, (0.43, 0.72, 0.48, 0.78)),
        ("face", R.face_base, 10, (0.40, 0.60, 0.60, 0.90)),
    ])
    normalize_face_zorder(stack, meshes)
    o = _order(stack)
    assert o["mouth"] > o["face"] and o["eye"] > o["face"]
