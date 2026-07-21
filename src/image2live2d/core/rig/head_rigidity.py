"""Protected-region rigidity for the head-turn warp (shared by the moc3 + cmo3 backends).

A head-turn squash foreshortens the *whole* face uniformly — measured through the native Cubism core,
every facial feature narrows ~9% at the ±30° yaw extreme (eyes 0.914, mouth 0.909, nose 0.899, face
0.909). Anime Live2D convention — and every rival we studied (see the ``rival-code-harvest`` memory) —
keeps the focal features rigid so they don't shrink as the head turns: the eyes fully (weight 1.0), the
nose and mouth partially (0.30). This also pulls the moc3/cmo3 turn *toward* nijilive, whose head node
rotates as a rigid unit (features never narrow at all).

The mechanism is backend-neutral: each backend builds its own warp grid and its own pseudo-3D squash,
so this only decides, per grid control point, **how rigid** it should be and **which feature centroid**
it rides. The backend then blends its own squash target for that point toward a rigid, translate-only
target — moving the whole protected region by its centroid's displacement, which preserves the region's
internal spacing (that is what "rigid" means here). Roll is exempt: an in-plane rotation already
preserves shape, so a rolled point never needs protecting.
"""

from __future__ import annotations

import math

# Semantic-role name -> rigidity weight in [0, 1]. A feature the head-turn squash should NOT shrink:
# 1.0 = fully rigid (translates with the head, never narrows), 0.0 = fully squashed (unlisted roles).
# The whole eye cluster (white/lid/pupil/closed) rides together so it stays a coherent shape.
PROTECT: dict[str, float] = {
    "eye_l": 1.0, "eye_r": 1.0,
    "eye_white_l": 1.0, "eye_white_r": 1.0,
    "eye_closed_l": 1.0, "eye_closed_r": 1.0,
    "pupil_l": 1.0, "pupil_r": 1.0,
    "nose": 0.30,
    "mouth": 0.30,
}

Bbox = tuple[float, float, float, float]        # (x0, y0, x1, y1)
Region = tuple[float, float, float, float, float]  # (x0, y0, x1, y1, weight)


def regions_from(role_bboxes: dict[str, Bbox]) -> list[Region]:
    """Turn a ``role -> bbox`` map into weighted protection regions, keeping only protected roles with a
    real bbox. ``role_bboxes`` values may be ``None`` (role absent) and are skipped."""
    out: list[Region] = []
    for role, w in PROTECT.items():
        b = role_bboxes.get(role)
        if b is not None:
            out.append((b[0], b[1], b[2], b[3], w))
    return out


# Rigidity ramps to 0 over a margin band this fraction of the bbox's larger side, outside it. Smaller =
# tighter protection (sharper grid transition, more residual squash on the feature); larger = softer but
# it bleeds rigidity into the surrounding face. Tuned on the native core (see RIVAL_HARVEST_BACKLOG T5).
_MARGIN_FRAC = 0.5


def _falloff(x: float, y: float, x0: float, y0: float, x1: float, y1: float) -> float:
    """1.0 inside the bbox, ramping linearly to 0 over a margin band outside it — so rigidity fades
    smoothly into the surrounding face instead of kinking the grid."""
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    d = math.hypot(dx, dy)
    if d <= 0.0:
        return 1.0
    margin = _MARGIN_FRAC * max(x1 - x0, y1 - y0, 1e-6)
    if margin <= 0.0:
        return 0.0
    return max(0.0, 1.0 - d / margin)


def rigidity_field(
    points: list[tuple[float, float]], regions: list[Region],
) -> list[tuple[float, float, float]]:
    """For each ``(x, y)`` control point, the ``(weight, cx, cy)`` of the protecting region it rides:
    ``weight`` in [0, 1] and ``(cx, cy)`` that region's centroid. A point inside/near several regions
    takes the one with the strongest influence (``weight * falloff``); a point near none gets
    ``(0.0, x, y)`` — fully squashed, its own position as a harmless centroid.
    """
    out: list[tuple[float, float, float]] = []
    for x, y in points:
        best_w, best_c = 0.0, (x, y)
        for x0, y0, x1, y1, w in regions:
            infl = w * _falloff(x, y, x0, y0, x1, y1)
            if infl > best_w:
                best_w = infl
                best_c = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        out.append((best_w, best_c[0], best_c[1]))
    return out
