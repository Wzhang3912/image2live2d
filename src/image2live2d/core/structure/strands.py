"""Per-strand hair planning (P2) — split hair into independent strands, one param + pendulum each.

Before P2 the rig lumped *all* parts of a hair role into ONE output param and ONE pendulum, so
twin-tails / pigtails / a ponytail + fringe moved as a single welded blob. Here each hair **part**
(layer) becomes its own strand: its own sway param and its own physics pendulum, so they swing
independently. This is the seam where intra-layer connected-component splitting will later plug in
(a single layer with two disconnected lobes → two strands); for now the unit of a strand is a part.

The first part of a role keeps the **base** param id (``ParamHairSide``) and extra parts get a numeric
suffix (``ParamHairSide2`` …), so a character with a single part per role is **unchanged**. When meshes
are available the pendulum mass/length scale with each strand's height relative to its role's mean (a
longer tail lags more → the strands visibly desync); a lone or exactly-average strand gets factor 1.0,
i.e. the role's base tuning verbatim — so single-strand output is byte-identical to before P2.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ..types import LayerStack
from ...irr.schema import Mesh, SemanticRole, Vec2

# A component smaller than this fraction of a part's vertices is treated as a stray fragment (alpha
# speckle / antialiasing island), not a real strand — its vertices fold into the nearest real lobe.
_MIN_COMPONENT_FRAC = 0.1

# --- Bottom-contour strand-tip detection ----------------------------------------------------------
# A connected hair sheet usually hangs in several distinct LOCKS — a fringe parted into strands, a
# ponytail splitting toward its tip — with no alpha gap between them, so connected-components sees ONE
# strand and the whole sheet swings as a single rigid blob. The locks show up as separate low points
# ("tips") along the sheet's BOTTOM contour. Detect them as prominent local maxima of the box-smoothed
# bottom edge and split the lobe's vertices to the nearest tip, so each lock gets its own pendulum.
# (Adapted from Anime2.5DRig's detectStrands; we read the contour off the mesh grid, not the raw alpha.)
_TIP_BINS = 64                # x-resolution of the bottom contour
_TIP_SMOOTH = 9              # box-smoothing window over the bins (kills antialiasing wobble)
_TIP_MIN_PROMINENCE = 0.18   # a tip must dip this far (fraction of the lobe height) below its saddle
_TIP_MIN_SEPARATION = 6      # bins two tips must be apart — merges locks that are basically one
_TIP_MAX = 6                 # never more than this many strands from one lobe

# Base pendulum tuning per hair role. These are exactly the pre-P2 physics._HAIR_TUNING values, so one
# strand of a role reproduces the old physics rig verbatim: back hair heavy/slow, front fringe light.
HAIR_BASE_TUNING: dict[SemanticRole, tuple[str, tuple[float, float, float]]] = {
    SemanticRole.hair_front: ("ParamHairFront", (1.1, 0.10, 1.05)),
    SemanticRole.hair_side: ("ParamHairSide", (1.4, 0.08, 1.30)),
    SemanticRole.hair_back: ("ParamHairBack", (2.0, 0.06, 1.70)),
}
HAIR_DRIVER = "ParamAngleX"  # head turn drives hair sway (yaw; pitch/roll added as extra drivers)


@dataclass
class StrandSpec:
    """One hair strand: the part it deforms, its output param id, and its pendulum material.

    ``vertex_indices`` is the subset of the part's mesh vertices this strand owns (``None`` = the whole
    mesh, the single-lobe case). When a hair layer holds two disconnected lobes (twin-tails fused into
    one part), each lobe is its own strand with its own vertex subset, so they swing independently even
    though they share a texture/mesh."""

    part_id: str
    param_id: str
    role: SemanticRole
    mass: float
    drag: float
    length: float
    vertex_indices: list[int] | None = None


def _height_of(verts: list[Vec2]) -> float:
    ys = [y for _, y in verts]
    return (max(ys) - min(ys)) if ys else 0.0


def _centroid(verts: list[Vec2]) -> Vec2:
    n = len(verts) or 1
    return (sum(x for x, _ in verts) / n, sum(y for _, y in verts) / n)


def mesh_components(mesh: Mesh) -> list[list[int]]:
    """Split a mesh's vertices into connected components (lobes) via its triangle graph.

    ``grid_mesh`` drops fully-transparent cells, so two alpha lobes separated by a gap become two
    disconnected triangle clusters — this recovers them with no alpha access. Returns one list of
    vertex indices per lobe (largest first); a single connected mesh returns ``[all indices]``. Stray
    fragments below ``_MIN_COMPONENT_FRAC`` of the vertices are folded into the nearest real lobe, so a
    speckle can't spawn a spurious strand."""
    n = len(mesh.vertices)
    if n == 0:
        return []
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for tri in mesh.triangles:
        union(tri[0], tri[1])
        union(tri[1], tri[2])

    groups: dict[int, list[int]] = defaultdict(list)
    for v in range(n):
        groups[find(v)].append(v)
    comps = sorted(groups.values(), key=len, reverse=True)
    if len(comps) == 1:
        return [list(range(n))]

    min_verts = max(3, int(_MIN_COMPONENT_FRAC * n))
    big = [c for c in comps if len(c) >= min_verts]
    if len(big) <= 1:
        return [list(range(n))]

    # Partition ALL vertices (including stray-fragment ones) to the nearest real-lobe centroid.
    cents = [_centroid([mesh.vertices[i] for i in c]) for c in big]
    labels: list[list[int]] = [[] for _ in big]
    for v in range(n):
        vx, vy = mesh.vertices[v]
        k = min(range(len(big)), key=lambda j: (vx - cents[j][0]) ** 2 + (vy - cents[j][1]) ** 2)
        labels[k].append(v)
    return labels


def _bottom_contour(verts: list[Vec2], x0: float, span: float) -> list[float | None]:
    """The lowest (max-y; our y is y-DOWN so the hair tips are at large y) point per x-bin over
    ``verts``, as a ``_TIP_BINS``-long list. Empty bins are ``None`` (filled by the caller)."""
    contour: list[float | None] = [None] * _TIP_BINS
    for x, y in verts:
        b = min(_TIP_BINS - 1, int((x - x0) / span * _TIP_BINS))
        if contour[b] is None or y > contour[b]:
            contour[b] = y
    return contour


def _smooth_filled(contour: list[float | None]) -> list[float]:
    """Fill empty bins by nearest-neighbour hold, then box-smooth by ``_TIP_SMOOTH`` — a clean 1-D
    bottom edge to find tips on."""
    filled: list[float] = []
    last = next((c for c in contour if c is not None), 0.0)
    for c in contour:
        last = c if c is not None else last
        filled.append(last)
    k = _TIP_SMOOTH
    out = []
    for i in range(len(filled)):
        lo, hi = max(0, i - k // 2), min(len(filled), i + k // 2 + 1)
        out.append(sum(filled[lo:hi]) / (hi - lo))
    return out


def _tip_bins(contour: list[float]) -> list[int]:
    """Bins that are prominent local maxima (hair tips) of the smoothed bottom contour: a peak whose
    drop to the higher of its flanking valleys is at least ``_TIP_MIN_PROMINENCE`` of the contour's
    total height. Peaks closer than ``_TIP_MIN_SEPARATION`` bins are merged (the lower one drops)."""
    lo, hi = min(contour), max(contour)
    height = hi - lo
    if height <= 0:
        return []
    peaks = [i for i in range(1, len(contour) - 1)
             if contour[i] >= contour[i - 1] and contour[i] > contour[i + 1]]
    prominent = []
    for i in peaks:
        left = min(contour[:i]) if i else contour[i]
        right = min(contour[i + 1:]) if i + 1 < len(contour) else contour[i]
        if (contour[i] - max(left, right)) >= _TIP_MIN_PROMINENCE * height:
            prominent.append(i)
    # merge near-duplicates, keeping the lower-hanging (larger y) tip
    prominent.sort()
    merged: list[int] = []
    for i in prominent:
        if merged and i - merged[-1] < _TIP_MIN_SEPARATION:
            if contour[i] > contour[merged[-1]]:
                merged[-1] = i
        else:
            merged.append(i)
    # keep the deepest-hanging tips if there are more than the cap
    merged.sort(key=lambda i: -contour[i])
    return sorted(merged[:_TIP_MAX])


def split_lobe_by_tips(mesh: Mesh, indices: list[int]) -> list[list[int]]:
    """Split one connected hair lobe into per-lock strands by its bottom-contour tips (see the block
    comment above). Returns ``[indices]`` unchanged when fewer than two prominent tips are found — a
    round bun or a single lock is never force-split. Otherwise partitions every vertex to the nearest
    tip in x, so each lock owns a contiguous slice of the sheet."""
    verts = [mesh.vertices[i] for i in indices]
    xs = [x for x, _ in verts]
    x0, x1 = min(xs), max(xs)
    span = x1 - x0
    if span <= 1e-6 or len(indices) < 2 * _TIP_MIN_SEPARATION:
        return [indices]
    tip_bins = _tip_bins(_smooth_filled(_bottom_contour(verts, x0, span)))
    if len(tip_bins) < 2:
        return [indices]
    tip_xs = [x0 + (b + 0.5) / _TIP_BINS * span for b in tip_bins]
    groups: list[list[int]] = [[] for _ in tip_xs]
    for i in indices:
        vx = mesh.vertices[i][0]
        k = min(range(len(tip_xs)), key=lambda j: abs(vx - tip_xs[j]))
        groups[k].append(i)
    # a tip that captured no vertices (rare, adjacent tips) is dropped; keep non-empty locks in x order
    return [g for _, g in sorted(zip(tip_xs, groups)) if g]


def strand_param_id(base: str, index: int) -> str:
    """First part keeps the base id; extras get a 1-based numeric suffix (base, base2, base3, …)."""
    return base if index == 0 else f"{base}{index + 1}"


def hair_strands(stack: LayerStack, meshes: list[Mesh]) -> list[StrandSpec]:
    """One ``StrandSpec`` per hair **strand** — a connected lobe of a hair part — in (role, stack,
    lobe) order. This is the deterministic plan both ``author_rig`` (sway keyforms) and
    ``generate_physics`` (pendulums) consume so their param ids always agree.

    A hair part with a single connected mesh yields one strand over the whole mesh (unchanged); a part
    holding two disconnected lobes (twin-tails fused into one layer) yields one strand per lobe, each
    owning its lobe's vertices. Mass/length scale with each strand's height vs its role's mean (factor
    1.0 for a lone/average strand → base tuning unchanged)."""
    mbp = {m.part_id: m for m in meshes}
    # (part_id, vertex_indices|None, height) per strand unit, grouped by role in stack order.
    units: dict[SemanticRole, list[tuple[str, list[int] | None, float]]] = defaultdict(list)
    for ly in stack.layers:
        if ly.semantic_role not in HAIR_BASE_TUNING or ly.id not in mbp:
            continue
        m = mbp[ly.id]
        # First split by connected components (alpha gaps: twin-tails fused into one layer), then split
        # each connected lobe again by its bottom-contour tips (locks with no gap between them).
        sublobes: list[list[int]] = []
        for comp in mesh_components(m):
            sublobes.extend(split_lobe_by_tips(m, comp))
        if len(sublobes) <= 1:
            units[ly.semantic_role].append((ly.id, None, _height_of(m.vertices)))
        else:
            for sub in sublobes:
                h = _height_of([m.vertices[i] for i in sub])
                units[ly.semantic_role].append((ly.id, sub, h))

    specs: list[StrandSpec] = []
    for role, (base, (m0, d0, l0)) in HAIR_BASE_TUNING.items():
        role_units = units.get(role)
        if not role_units:
            continue
        heights = [h for _, _, h in role_units]
        mean = sum(heights) / len(heights)
        for i, (pid, indices, h) in enumerate(role_units):
            f = (h / mean) if mean > 0 else 1.0
            specs.append(StrandSpec(part_id=pid, param_id=strand_param_id(base, i), role=role,
                                    mass=m0 * f, drag=d0, length=l0 * f, vertex_indices=indices))
    return specs


def hair_specs_from_params(param_ids) -> list[StrandSpec]:
    """Reconstruct strand specs (base tuning, no geometry scaling) from the hair param ids already in a
    parameter set — used by ``generate_physics`` when meshes aren't supplied. Yields base then suffixed
    ids per role in (front, side, back) order, matching ``hair_strands``' emission order."""
    ids = set(param_ids)
    specs: list[StrandSpec] = []
    for role, (base, (m0, d0, l0)) in HAIR_BASE_TUNING.items():
        i = 0
        while True:
            pid = strand_param_id(base, i)
            if pid not in ids:
                break
            specs.append(StrandSpec(part_id="", param_id=pid, role=role,
                                    mass=m0, drag=d0, length=l0))
            i += 1
    return specs
