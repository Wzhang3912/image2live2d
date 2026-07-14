"""Face z-order normalisation — make the expressive parts of the face actually visible.

The decomposer's depth model is not trustworthy for the face. On a real character it ordered
``eyebrow`` *below* ``face_base``, so the brows were painted under the skin and could never be seen:
driving ``ParamBrowLY`` through its whole range changed **zero pixels**. The same class of error is
already corrected for head ornaments by ``_lift_occluded_accessories`` in the pipeline; this module
does it for the face, where the stakes are higher — a brow that cannot be seen makes the brow
parameter and every expression clip that uses it (angry/sad/surprise) dead weight.

Two rules, both conditional on *actual* occlusion so a correctly-ordered stack is left untouched:

1. **A face feature never hides under the face.** Skin is opaque; a feature drawn below ``face_base``
   is invisible by construction, so lift it above.
2. **Eyebrows read through the fringe.** Anime rigs draw the brows *over* the front hair so the
   expression stays legible under a heavy fringe. This is not a guess: in Hiyori (a shipping
   commercial model) the brow meshes render at order 130-131 while the bangs are at 57 and 103.
   We only lift a brow when the front hair genuinely covers it.

Both rules preserve relative order within the parts they move, and only ever raise a part, so a stack
that is already correct comes out unchanged (the golden suite pins this).
"""

from __future__ import annotations

from ..types import LayerStack
from ...irr.schema import Mesh, SemanticRole, Vec2

# Parts that live on the face and must read above the skin.
FACE_FEATURES: frozenset[SemanticRole] = frozenset({
    SemanticRole.eyebrow_l, SemanticRole.eyebrow_r,
    SemanticRole.eye_l, SemanticRole.eye_r,
    SemanticRole.eye_white_l, SemanticRole.eye_white_r,
    SemanticRole.pupil_l, SemanticRole.pupil_r,
    SemanticRole.nose, SemanticRole.mouth, SemanticRole.blush,
})

BROWS: frozenset[SemanticRole] = frozenset({SemanticRole.eyebrow_l, SemanticRole.eyebrow_r})

# A part counts as "covered" by another only if a real share of its footprint is behind it — a stray
# pixel of overlap is not occlusion.
_COVER_MIN = 0.25


def _bbox(verts: list[Vec2]) -> tuple[float, float, float, float]:
    xs = [x for x, _ in verts]
    ys = [y for _, y in verts]
    return min(xs), min(ys), max(xs), max(ys)


def _covered_frac(box: tuple[float, float, float, float], other: tuple[float, float, float, float]) -> float:
    """Fraction of ``box``'s area that lies inside ``other``."""
    bx0, by0, bx1, by1 = box
    ox0, oy0, ox1, oy1 = other
    iw = max(0.0, min(bx1, ox1) - max(bx0, ox0))
    ih = max(0.0, min(by1, oy1) - max(by0, oy0))
    return (iw * ih) / max((bx1 - bx0) * (by1 - by0), 1e-12)


def normalize_face_zorder(stack: LayerStack, meshes: list[Mesh]) -> list[str]:
    """Raise face features that the decomposer buried. Mutates ``stack``; returns the ids it moved.

    A feature is lifted only when it is *both* drawn below an occluder *and* actually covered by it,
    so a stack the decomposer got right is returned untouched.
    """
    box_by_part = {m.part_id: _bbox(m.vertices) for m in meshes}
    by_role: dict[SemanticRole, list] = {}
    for layer in stack.layers:
        by_role.setdefault(layer.semantic_role, []).append(layer)

    def occluders(role: SemanticRole) -> list:
        return [ly for ly in by_role.get(role, []) if ly.id in box_by_part]

    moved: list[str] = []

    def lift_above(layer, tops: list) -> bool:
        """Raise ``layer`` just above the highest of ``tops`` that covers it and outranks it."""
        box = box_by_part.get(layer.id)
        if box is None:
            return False
        blocking = [t for t in tops
                    if t.draw_order > layer.draw_order
                    and _covered_frac(box, box_by_part[t.id]) >= _COVER_MIN]
        if not blocking:
            return False
        layer.draw_order = max(t.draw_order for t in blocking) + 1
        return True

    # 1. No face feature hides under the skin.
    face_base = occluders(SemanticRole.face_base)
    for layer in stack.layers:
        if layer.semantic_role in FACE_FEATURES and lift_above(layer, face_base):
            moved.append(layer.id)

    # 2. Brows read through the fringe (Hiyori draws them above the bangs).
    hair_front = occluders(SemanticRole.hair_front)
    for layer in stack.layers:
        if layer.semantic_role in BROWS and lift_above(layer, hair_front):
            if layer.id not in moved:
                moved.append(layer.id)

    stack.layers.sort(key=lambda ly: ly.draw_order)
    return moved
