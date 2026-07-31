"""Skirt / cloth-hem planning (P3) — geometry-derived pendulum material for a garment's hem zones.

Before P3 the skirt used three fixed L/C/R zones with hardcoded pendulum material, so a floor-length
dress and a mini-skirt swung with the *same* pendulum length. Here each zone's mass/length is derived
from the garment's actual geometry (``material_from_geometry``): a longer hem → a longer, slower
pendulum (bigger arc, more follow-through); more fabric → more mass (more lag). The base per-zone
tuning is the pre-P3 constants, anchored to a reference-sized garment (factor 1.0), so a typical skirt
keeps today's feel and only unusual garments scale.

The zone *count* now scales with hem width (P3b): a reference-width hem keeps the three overlapping
L/C/R windows (byte-identical), a markedly wider hem breaks into more evenly-tiled interior lobes
(``ParamSkirtC1``, ``ParamSkirtC2`` …) so a full skirt ripples in more independent zones, and a narrow
hem collapses to two halves or a single central sway so a thin frill doesn't carry three redundant
windows. Both ``author_rig`` (windows) and ``generate_physics`` (material) consume this one planner so
they never drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..types import LayerStack
from ...irr.schema import Mesh, SemanticRole, Vec2
from .limbs import _leg_seam

# A clothing part is treated as a swingable skirt hem unless its geometry says otherwise (thresholds
# moved verbatim from rig.author; model space is y up, normalized to the canvas).
_FOOTWEAR_TOP_Y = 0.28   # top below this -> footwear at the feet, not a hem
_CLOTH_HEM_MIN_Y = 0.20  # bundled skirt+legs (waist -> feet) if it starts low AND reaches the waist
_CLOTH_WAIST_Y = 0.45    # a part sitting entirely at/above this is a top/shirt (rides the body rigidly)

# A waist->feet clothing part is EITHER a floor-length gown / A-line skirt (which should ripple as a
# multi-zone hem) or a pair of legs / pants (which must NOT sway as cloth). They are told apart by shape:
# a gown flares OUT to a wide continuous hem, while pants taper to two narrow leg columns. We require a
# real outward flare (hem markedly wider than the thigh band) and a substantial hem, plus — as cheap
# insurance — no detectable inter-leg seam (a strap dangling down the crotch can defeat the seam test on
# its own, e.g. cargo pants, which is exactly why the flare test is the primary gate). Tuned against the
# corpus: gowns flare >=1.17 (kimono) up to 1.75, pants sit at ~0.86; narrow front panels are ~0.13-0.16
# wide and never reach the hem-width floor.
_GOWN_FLARE_MIN = 1.15       # hem width / thigh-band width; below this it tapers (pants/leggings)
_GOWN_HEM_MIN_WIDTH = 0.25   # a real gown hem is a wide sweep, not a thin apron/sash panel

# Reference garment (normalized) at which the base tuning holds; real garments scale relative to it.
_REF_HANG = 0.22
_REF_AREA = 0.09

# Zone *count* scales with hem width (P3b). A reference-width hem (~_REF_SPAN of the canvas) ripples in
# 3 lobes — the pre-P3b fixed L/C/R. Each _SPAN_PER_ZONE of width away from the reference adds or drops a
# lobe: a markedly wider hem breaks into more independent lobes (capped at _MAX_ZONES); a narrow hem
# (a pencil skirt, a thin frill) collapses to fewer — 2 halves, or a single central sway — down to
# _MIN_ZONES. Exactly 3 zones reproduces the old layout (centres, windows, drivers, material) byte-for-
# byte, and the reference band stays 3, so every reference-width garment is unchanged.
_REF_SPAN = 0.40        # a typical skirt spans ~40% of the canvas -> 3 zones
_SPAN_PER_ZONE = 0.16   # each ~16% of width away from the reference adds/drops a hem lobe
_MAX_ZONES = 7          # cap: even a full-width gown ripples in a bounded number of lobes
_MIN_ZONES = 1          # floor: a very narrow hem is one central sway, not three overlapping windows

# Edge vs interior base tuning + drivers (were the per-zone _SKIRT_ZONES constants). Edge zones couple
# to the near leg; interior zones carry more fabric (heavier, longer) and couple to body lean/twist.
# Mass/drag re-tuned against Hiyori's cloth row (RIVAL_HARVEST_BACKLOG T7): our skirt was the ONE thing we
# emitted that fell OUTSIDE the real-artist regime — raw mobility (1-drag) came out 0.61-0.65 against a
# real floor of ~0.71, so physics3.py had to clamp it, i.e. we were emitting cloth more heavily damped
# than any rig on hand. Hiyori's skirt is mobility 0.9 / delay 0.6; these bases land ~0.86 / ~0.67 after
# the geometry scaling below, in-regime and unclamped even for a short hem (the case that used to clamp).
_EDGE_BASE = (0.9, 0.10, 1.3)
_INTERIOR_BASE = (1.1, 0.09, 1.5)
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
    """Number of hem lobes for a garment of horizontal ``span`` (normalized to the canvas): 3 across the
    reference band, +1 per _SPAN_PER_ZONE wider (capped at _MAX_ZONES), -1 per _SPAN_PER_ZONE narrower
    (floored at _MIN_ZONES). The reference band [_REF_SPAN - _SPAN_PER_ZONE, _REF_SPAN + _SPAN_PER_ZONE)
    stays 3, so every current (0.40-span) test garment is byte-identical."""
    steps = int((span - _REF_SPAN) / _SPAN_PER_ZONE) if span >= _REF_SPAN else \
        -int((_REF_SPAN - span) / _SPAN_PER_ZONE)
    return max(_MIN_ZONES, min(3 + steps, _MAX_ZONES))


def _zone_layout(n: int) -> list[tuple[str, list[str], tuple[float, float, float]]]:
    """(param id, drivers, base material) for each of ``n`` zones, left→right. A single zone is one
    body-coupled central sway (``ParamSkirtC``); otherwise the two ends are the leg-coupled edges (L, R)
    and everything between is a body-coupled interior. ``n == 3`` yields exactly the old L / C / R."""
    if n == 1:
        return [("ParamSkirtC", list(_INTERIOR_DRIVERS), _INTERIOR_BASE)]
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
    # The specific skirt TIER (clothing part) this zone drives; ``None`` = the whole skirt group (the
    # mesh-less physics path). A layered skirt gives each tier its own zones so they ripple on their
    # own phase instead of the whole stack moving as one — see skirt_zones.
    part_id: str | None = None


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


def _band_width(verts: list[Vec2], y_lo: float, y_hi: float) -> float:
    """Horizontal extent of the vertices lying in the [y_lo, y_hi] band (0 if the band is empty)."""
    xs = [x for x, y in verts if y_lo <= y <= y_hi]
    return (max(xs) - min(xs)) if xs else 0.0


def _gown_hem(mesh: Mesh, body_box) -> bool:
    """True if a waist->feet clothing part reads as a floor-length gown / A-line skirt (a wide, flared,
    continuous hem) rather than legs / pants (two tapering columns). A gown's hem band is markedly wider
    than its thigh band and spans a real sweep; a seam down the midline (a crotch gap) rules it out.

    ``body_box`` is the whole-figure bounding box (as ``split_fused_legs`` uses), needed by the seam
    guard; without it we cannot safely tell a gown from bundled legs, so the caller stays conservative."""
    if body_box is None:
        return False
    verts = mesh.vertices
    ys = [y for _, y in verts]
    if not ys:
        return False
    y0, y1 = min(ys), max(ys)
    h = y1 - y0
    if h <= 0:
        return False
    bot = _band_width(verts, y0, y0 + 0.15 * h)              # hem band (near the floor)
    mid = _band_width(verts, y0 + 0.45 * h, y0 + 0.65 * h)   # thigh / mid band
    if mid <= 0 or bot < _GOWN_HEM_MIN_WIDTH:
        return False
    if bot / mid < _GOWN_FLARE_MIN:
        return False                                         # tapers / columnar -> pants, not a gown
    return _leg_seam(mesh, body_box=body_box) is None        # a real inter-leg seam -> bundled legs


def _skirtable(mesh: Mesh, *, body_box=None) -> bool:
    x0, y0, x1, y1 = _bbox(mesh.vertices)   # y up: y0 bottom, y1 top
    if y1 < _FOOTWEAR_TOP_Y:
        return False                        # footwear (entirely at the feet)
    if y0 < _CLOTH_HEM_MIN_Y and y1 >= _CLOTH_WAIST_Y:
        # Waist -> feet: bundled legs / pants, UNLESS it reads as a flared gown hem. Without a body_box
        # we can't run the gown test, so we stay conservative (reject) — keeping the pre-gown behavior
        # byte-for-byte for any caller that doesn't thread the figure box through.
        if not _gown_hem(mesh, body_box):
            return False
    if y0 >= _CLOTH_WAIST_Y:
        return False                        # a top/shirt: rides the body, no hem to swing
    return True


def skirt_cloth(stack: LayerStack, meshes: list[Mesh]) -> list[tuple[str, Mesh]]:
    """The clothing parts that read as a swingable skirt hem, in stack order. A floor-length gown / A-line
    skirt (a waist->feet clothing part that flares to a wide hem) counts too; legs / pants do not."""
    mbp = {m.part_id: m for m in meshes}
    all_verts = [v for m in meshes for v in m.vertices]
    body_box = _bbox(all_verts) if all_verts else None
    out: list[tuple[str, Mesh]] = []
    for ly in stack.layers:
        if (ly.semantic_role is SemanticRole.clothing and ly.id in mbp
                and _skirtable(mbp[ly.id], body_box=body_box)):
            out.append((ly.id, mbp[ly.id]))
    return out


def skirt_zones(stack: LayerStack, meshes: list[Mesh]) -> list[ZoneSpec]:
    """Plan the hem zones for a garment: a width-driven zone count (3 for a reference hem, more for a
    wide one) with evenly-tiled overlapping windows and geometry-scaled material. Empty if there is no
    skirtable cloth. A 3-zone (reference-width) garment reproduces the old L/C/R layout exactly."""
    cloth = skirt_cloth(stack, meshes)
    if not cloth:
        return []
    # A layered/tiered skirt (ruffles) arrives as several skirtable parts stacked vertically. The old
    # plan unioned them and drove every tier with the SAME L/C/R params, so the whole stack swung as one.
    # Instead: the PRIMARY tier (largest area — the main skirt) keeps the width-scaled L/C/R(+interior)
    # zones scoped to its own part; each EXTRA tier gets one independent central sway (ParamSkirtT1, T2…)
    # scoped to its part, so a ruffle layer ripples on its own phase. A single-tier skirt is unchanged
    # (one part = the primary, its zones scoped to it = the old output verbatim).
    def _area(m: Mesh) -> float:
        x0, y0, x1, y1 = _bbox(m.vertices)
        return (x1 - x0) * (y1 - y0)
    primary = max(range(len(cloth)), key=lambda i: _area(cloth[i][1]))
    zones: list[ZoneSpec] = []
    tier_k = 0
    for i, (pid, m) in enumerate(cloth):
        x0, y0, x1, y1 = _bbox(m.vertices)
        span = max(x1 - x0, 1e-6)
        hang = y1 - y0
        area = span * hang
        if i == primary:
            n = _zone_count(span)
            half_w = span / n                # overlapping windows (each 2*span/n wide) for continuity
            for j, (zpid, drivers, base) in enumerate(_zone_layout(n)):
                center_x = x0 + span * (j + 0.5) / n
                mass, drag, length = material_from_geometry(base, hang, area)
                zones.append(ZoneSpec(zpid, center_x, half_w, drivers, mass, drag, length, part_id=pid))
        else:
            tier_k += 1
            mass, drag, length = material_from_geometry(_INTERIOR_BASE, hang, area)
            zones.append(ZoneSpec(f"ParamSkirtT{tier_k}", 0.5 * (x0 + x1), span,
                                  list(_INTERIOR_DRIVERS), mass, drag, length, part_id=pid))
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
    t = 1                                     # extra tiers of a layered skirt (ParamSkirtT1, T2 …)
    while f"ParamSkirtT{t}" in ids:
        out.append(ZoneSpec(f"ParamSkirtT{t}", 0.0, 0.0, list(_INTERIOR_DRIVERS), *_INTERIOR_BASE))
        t += 1
    return out
