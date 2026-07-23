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
    SemanticRole.nose, SemanticRole.mouth, SemanticRole.mouth_cavity, SemanticRole.blush,
})

BROWS: frozenset[SemanticRole] = frozenset({SemanticRole.eyebrow_l, SemanticRole.eyebrow_r})

LEGS: frozenset[SemanticRole] = frozenset({SemanticRole.leg_l, SemanticRole.leg_r})

# The top slice of a leg where See-through's cut-off seam sits (it severs the leg at the hemline).
_LEG_TOP_BAND = 0.30

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

    def lift_group(roles: frozenset[SemanticRole], tops: list) -> None:
        """Raise every part in ``roles`` that ``tops`` both outranks and actually covers.

        Lifted parts are re-stacked in their existing relative order and given *distinct* draw orders.
        Order matters among them — the lips have to read on top of the cavity painted behind them — so
        a shared "just above the occluder" value would leave that to a tie-break.
        """
        nxt: int | None = None
        for layer in sorted(stack.layers, key=lambda ly: ly.draw_order):
            if layer.semantic_role not in roles:
                continue
            box = box_by_part.get(layer.id)
            if box is None:
                continue
            blocking = [t.draw_order for t in tops
                        if t.draw_order > layer.draw_order
                        and _covered_frac(box, box_by_part[t.id]) >= _COVER_MIN]
            if not blocking:
                continue
            base = max(blocking) + 1
            nxt = base if nxt is None else max(nxt + 1, base)
            layer.draw_order = nxt
            if layer.id not in moved:
                moved.append(layer.id)

    # 1. No face feature hides under the skin.
    lift_group(FACE_FEATURES, occluders(SemanticRole.face_base))
    # 2. Brows read through the fringe (Hiyori draws them above the bangs).
    lift_group(BROWS, occluders(SemanticRole.hair_front))

    stack.layers.sort(key=lambda ly: ly.draw_order)
    return moved


def normalize_leg_zorder(stack: LayerStack, meshes: list[Mesh]) -> list[str]:
    """Tuck a leg behind a body garment that covers its cut-off top. Mutates ``stack``; returns moved ids.

    See-through severs a leg at the hemline, leaving a flat top edge, and sometimes also orders that leg
    *in front of* the skirt — so the raw seam paints over the garment (the bare-thigh 'white/brown cut'
    artifact). Dropping the leg just behind such a garment lets the skirt cover the seam, while the lower
    leg still shows below the hem where the garment is transparent.

    Conservative like the face rules: only fires when a garment *already* sits behind the leg **and**
    actually covers the leg's top band, so a leg meant to be in front (its top not covered) is untouched.
    """
    box_by_part = {m.part_id: _bbox(m.vertices) for m in meshes}
    legs = [ly for ly in stack.layers if ly.semantic_role in LEGS and ly.id in box_by_part]
    garments = [ly for ly in stack.layers
                if ly.semantic_role == SemanticRole.clothing and ly.id in box_by_part]

    new_key: dict[str, float] = {}
    for leg in legs:
        lx0, ly0, lx1, ly1 = box_by_part[leg.id]
        top_band = (lx0, ly1 - _LEG_TOP_BAND * (ly1 - ly0), lx1, ly1)
        coverers = [g.draw_order for g in garments
                    if g.draw_order < leg.draw_order
                    and _covered_frac(top_band, box_by_part[g.id]) >= _COVER_MIN]
        if coverers:
            new_key[leg.id] = min(coverers) - 0.5      # just behind the lowest garment covering its top

    if not new_key:
        return []
    # Renumber to consecutive ints in the desired order — the moved legs slot in just behind their
    # garment, every other part keeps its relative order, so nothing else changes and there are no ties.
    order_key = lambda ly: new_key.get(ly.id, float(ly.draw_order))  # noqa: E731
    for i, ly in enumerate(sorted(stack.layers, key=order_key)):
        ly.draw_order = i
    stack.layers.sort(key=lambda ly: ly.draw_order)
    return list(new_key)
