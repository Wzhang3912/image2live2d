"""Skirt / cloth-hem planning (P3) — geometry-derived pendulum material for a garment's hem zones.

Before P3 the skirt used three fixed L/C/R zones with hardcoded pendulum material, so a floor-length
dress and a mini-skirt swung with the *same* pendulum length. Here each zone's mass/length is derived
from the garment's actual geometry (``material_from_geometry``): a longer hem → a longer, slower
pendulum (bigger arc, more follow-through); more fabric → more mass (more lag). The base per-zone
tuning is the pre-P3 constants, anchored to a reference-sized garment (factor 1.0), so a typical skirt
keeps today's feel and only unusual garments scale.

The zone *structure* (three overlapping L/C/R windows, their centres/widths, and their lower-body
drivers) is unchanged, so the authored sway keyforms stay byte-identical; only the physics material is
geometry-driven. Both ``author_rig`` (windows) and ``generate_physics`` (material) consume this one
planner so they never drift.
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

# Per-zone base: (param id, extra drivers, base (mass, drag, length)) — the pre-P3 _SKIRT_ZONES values.
# Centre zone is heavier/longer (more fabric hangs there); side zones couple to the near leg.
_ZONE_BASE: list[tuple[str, list[str], tuple[float, float, float]]] = [
    ("ParamSkirtL", ["ParamLegLA", "ParamBodyAngleZ"], (1.5, 0.28, 1.3)),
    ("ParamSkirtC", ["ParamBodyAngleZ", "ParamBodyAngleY"], (1.8, 0.25, 1.5)),
    ("ParamSkirtR", ["ParamLegRA", "ParamBodyAngleZ"], (1.5, 0.28, 1.3)),
]
SKIRT_PARAM_IDS: tuple[str, ...] = tuple(z[0] for z in _ZONE_BASE)


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
    """Plan the L/C/R hem zones for a garment: unchanged windows + geometry-scaled material. Empty if
    there is no skirtable cloth."""
    cloth = skirt_cloth(stack, meshes)
    if not cloth:
        return []
    boxes = [_bbox(m.vertices) for _, m in cloth]
    cx0 = min(b[0] for b in boxes)
    cx1 = max(b[2] for b in boxes)
    cy0 = min(b[1] for b in boxes)
    cy1 = max(b[3] for b in boxes)
    span = max(cx1 - cx0, 1e-6)
    half_w = span / 3.0                      # overlapping windows (each ~2/3 span) for continuity
    hang = cy1 - cy0
    area = span * hang
    centers = (cx0 + span / 6.0, (cx0 + cx1) / 2.0, cx1 - span / 6.0)
    zones: list[ZoneSpec] = []
    for (pid, drivers, base), center_x in zip(_ZONE_BASE, centers):
        mass, drag, length = material_from_geometry(base, hang, area)
        zones.append(ZoneSpec(pid, center_x, half_w, drivers, mass, drag, length))
    return zones


def skirt_specs_from_params(param_ids) -> list[ZoneSpec]:
    """Base-material zone specs (no geometry scaling, no windows) for the skirt params already present —
    used by ``generate_physics`` when meshes aren't supplied. Physics only needs param/drivers/material.
    """
    ids = set(param_ids)
    out: list[ZoneSpec] = []
    for pid, drivers, (mass, drag, length) in _ZONE_BASE:
        if pid in ids:
            out.append(ZoneSpec(pid, 0.0, 0.0, drivers, mass, drag, length))
    return out
