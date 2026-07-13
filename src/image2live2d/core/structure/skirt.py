"""Skirt / cloth-hem planning (P3) — geometry-derived pendulum material for a garment's hem zones.

Before P3 the skirt used three fixed L/C/R zones with hardcoded pendulum material, so a floor-length
dress and a mini-skirt swung with the *same* pendulum length. Here each zone's mass/length is derived
from the garment's actual geometry (``material_from_geometry``): a longer hem → a longer, slower
pendulum (bigger arc, more follow-through); more fabric → more mass (more lag). The base per-zone
tuning is the pre-P3 constants, anchored to a reference-sized garment (factor 1.0), so a typical skirt
keeps today's feel and only unusual garments scale.

The zone *count* now scales with hem width (P3b): a reference-width hem keeps the three overlapping
L/C/R windows (byte-identical), while a markedly wider hem breaks into more evenly-tiled interior lobes
(``ParamSkirtC1``, ``ParamSkirtC2`` …) so a full skirt ripples in more independent zones. Both
``author_rig`` (windows) and ``generate_physics`` (material) consume this one planner so they never
drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..types import LayerStack
from ...irr.schema import Mesh, SemanticRole, Vec2

# A clothing part is treated as a swingable skirt hem unless its geometry says otherwise (thresholds
# moved verbatim from rig.author; model space is y up, normalized to the canvas).
_FOOTWEAR_TOP_Y = 0.28   # top below this -> footwear at the feet, not a hem
_CLOTH_HEM_MIN_Y = 0.20  # bundled skirt+legs (waist -> feet) if it starts low AND reaches the waist
_CLOTH_WAIST_Y = 0.45    # a part sitting entirely at/above this is a top/shirt (rides the body rigidly)

# Reference garment (normalized) at which the base tuning holds; real garments scale relative to it.
_REF_HANG = 0.22
_REF_AREA = 0.09

# Zone *count* scales with hem width (P3b). A reference-width hem (~_REF_SPAN of the canvas) ripples in
# 3 lobes — the pre-P3b fixed L/C/R; a markedly wider hem breaks into more independent lobes, one extra
# per ~_SPAN_PER_ZONE of added width, capped at _MAX_ZONES. Exactly 3 zones reproduces the old layout
# (centres, windows, drivers, material) byte-for-byte, so every reference-width garment is unchanged.
_REF_SPAN = 0.40        # a typical skirt spans ~40% of the canvas -> 3 zones
_SPAN_PER_ZONE = 0.16   # each additional ~16% of width adds a hem lobe
_MAX_ZONES = 7          # cap: even a full-width gown ripples in a bounded number of lobes

# Edge vs interior base tuning + drivers (were the per-zone _SKIRT_ZONES constants). Edge zones couple
# to the near leg; interior zones carry more fabric (heavier, longer) and couple to body lean/twist.
_EDGE_BASE = (1.5, 0.28, 1.3)
_INTERIOR_BASE = (1.8, 0.25, 1.5)
_EDGE_DRIVERS_L = ["ParamLegLA", "ParamBodyAngleZ"]
_EDGE_DRIVERS_R = ["ParamLegRA", "ParamBodyAngleZ"]
_INTERIOR_DRIVERS = ["ParamBodyAngleZ", "ParamBodyAngleY"]

# The base 3 (catalog) ids; wide hems mint extra interior ids (ParamSkirtC1, C2 … — see params.py).
SKIRT_PARAM_IDS: tuple[str, ...] = ("ParamSkirtL", "ParamSkirtC", "ParamSkirtR")


def _interior_param_id(k: int) -> str:
    """kth interior zone id: first = ``ParamSkirtC`` (so a 3-zone skirt stays byte-identical), extras
    suffixed ``ParamSkirtC1``, ``ParamSkirtC2`` … (same first-is-base convention as hair strands)."""
    return "ParamSkirtC" if k == 0 else f"ParamSkirtC{k}"


def _zone_count(span: float) -> int:
    """Number of hem lobes for a garment of horizontal ``span`` (normalized to the canvas). 3 up to the
    reference width, then +1 per _SPAN_PER_ZONE of extra width, capped — never below 3 (byte-identical)."""
    if span <= _REF_SPAN:
        return 3
    return min(3 + int((span - _REF_SPAN) / _SPAN_PER_ZONE), _MAX_ZONES)


def _zone_layout(n: int) -> list[tuple[str, list[str], tuple[float, float, float]]]:
    """(param id, drivers, base material) for each of ``n`` zones, left→right. The two ends are the leg-
    coupled edges (L, R); everything between is a body-coupled interior. ``n == 3`` yields exactly the
    old L / C / R layout."""
    out: list[tuple[str, list[str], tuple[float, float, float]]] = []
    interior_k = 0
    for i in range(n):
        if i == 0:
            out.append(("ParamSkirtL", list(_EDGE_DRIVERS_L), _EDGE_BASE))
        elif i == n - 1:
            out.append(("ParamSkirtR", list(_EDGE_DRIVERS_R), _EDGE_BASE))
        else:
            out.append((_interior_param_id(interior_k), list(_INTERIOR_DRIVERS), _INTERIOR_BASE))
            interior_k += 1
    return out


@dataclass
class ZoneSpec:
    """One skirt hem zone: its output param, its window (for the sway keyform), its lower-body drivers,
    and its geometry-scaled pendulum material."""

    param_id: str
    center_x: float
    half_width: float
    extra_drivers: list[str] = field(default_factory=list)
    mass: float = 1.0
    drag: float = 0.25
    length: float = 1.3


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _bbox(verts: list[Vec2]) -> tuple[float, float, float, float]:
    xs = [x for x, _ in verts]
    ys = [y for _, y in verts]
    return min(xs), min(ys), max(xs), max(ys)


def material_from_geometry(
    base: tuple[float, float, float], hang: float, area: float,
    *, ref_hang: float = _REF_HANG, ref_area: float = _REF_AREA,
) -> tuple[float, float, float]:
    """Scale a base ``(mass, drag, length)`` by a garment's geometry.

    ``length`` grows with how far the hem hangs (bigger arc), ``mass`` with fabric area (more lag),
    ``drag`` falls as it lengthens (longer cloth is floppier). Factors are clamped so a pathological
    garment can't explode the sim; a reference-sized garment gives factor 1.0 → the base unchanged.
    """
    m0, d0, l0 = base
    hf = _clamp(hang / ref_hang, 0.4, 2.5) if ref_hang > 0 else 1.0
    af = _clamp(area / ref_area, 0.4, 2.5) if ref_area > 0 else 1.0
    length = l0 * hf
    mass = m0 * (0.5 + 0.5 * af)           # area influence, damped so it stays sane
    drag = d0 / hf
    return (mass, drag, length)


def _skirtable(mesh: Mesh) -> bool:
    x0, y0, x1, y1 = _bbox(mesh.vertices)   # y up: y0 bottom, y1 top
    if y1 < _FOOTWEAR_TOP_Y:
        return False                        # footwear (entirely at the feet)
    if y0 < _CLOTH_HEM_MIN_Y and y1 >= _CLOTH_WAIST_Y:
        return False                        # bundled skirt+legs (waist -> feet)
    if y0 >= _CLOTH_WAIST_Y:
        return False                        # a top/shirt: rides the body, no hem to swing
    return True


def skirt_cloth(stack: LayerStack, meshes: list[Mesh]) -> list[tuple[str, Mesh]]:
    """The clothing parts that read as a swingable skirt hem, in stack order."""
    mbp = {m.part_id: m for m in meshes}
    out: list[tuple[str, Mesh]] = []
    for ly in stack.layers:
        if ly.semantic_role is SemanticRole.clothing and ly.id in mbp and _skirtable(mbp[ly.id]):
            out.append((ly.id, mbp[ly.id]))
    return out


def skirt_zones(stack: LayerStack, meshes: list[Mesh]) -> list[ZoneSpec]:
    """Plan the hem zones for a garment: a width-driven zone count (3 for a reference hem, more for a
    wide one) with evenly-tiled overlapping windows and geometry-scaled material. Empty if there is no
    skirtable cloth. A 3-zone (reference-width) garment reproduces the old L/C/R layout exactly."""
    cloth = skirt_cloth(stack, meshes)
    if not cloth:
        return []
    boxes = [_bbox(m.vertices) for _, m in cloth]
    cx0 = min(b[0] for b in boxes)
    cx1 = max(b[2] for b in boxes)
    cy0 = min(b[1] for b in boxes)
    cy1 = max(b[3] for b in boxes)
    span = max(cx1 - cx0, 1e-6)
    hang = cy1 - cy0
    area = span * hang
    n = _zone_count(span)
    half_w = span / n                        # overlapping windows (each 2*span/n wide) for continuity
    zones: list[ZoneSpec] = []
    for i, (pid, drivers, base) in enumerate(_zone_layout(n)):
        center_x = cx0 + span * (i + 0.5) / n
        mass, drag, length = material_from_geometry(base, hang, area)
        zones.append(ZoneSpec(pid, center_x, half_w, drivers, mass, drag, length))
    return zones


def skirt_specs_from_params(param_ids) -> list[ZoneSpec]:
    """Base-material zone specs (no geometry scaling, no windows) for the skirt params already present —
    used by ``generate_physics`` when meshes aren't supplied. Emits left edge, then each present interior
    (C, C1, C2 …), then right edge — matching ``skirt_zones``' order; for the base L/C/R set that is the
    old output verbatim. Physics only needs param/drivers/material, so windows are zeroed."""
    ids = set(param_ids)
    out: list[ZoneSpec] = []
    if "ParamSkirtL" in ids:
        out.append(ZoneSpec("ParamSkirtL", 0.0, 0.0, list(_EDGE_DRIVERS_L), *_EDGE_BASE))
    k = 0
    while _interior_param_id(k) in ids:
        out.append(ZoneSpec(_interior_param_id(k), 0.0, 0.0, list(_INTERIOR_DRIVERS), *_INTERIOR_BASE))
        k += 1
    if "ParamSkirtR" in ids:
        out.append(ZoneSpec("ParamSkirtR", 0.0, 0.0, list(_EDGE_DRIVERS_R), *_EDGE_BASE))
    return out
