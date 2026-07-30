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
# The bottom edge must actually undulate — vary by at least _TIP_MIN_RELIEF of the strand's full height —
# before we look for tips at all. A flat rectangle bottom varies only by float noise (~1e-16), and
# dividing prominence by that noise turned rounding wobble into spurious tips that split a clean lobe
# differently on Python 3.10 vs 3.12. This absolute floor (vs the stable strand height) gates that out;
# once the edge genuinely undulates, tip PROMINENCE is measured against its own relief, so a shallow but
# real set of locks still splits.
_TIP_MIN_RELIEF = 0.06       # bottom-edge undulation, as a fraction of strand height, to look for tips
_TIP_MIN_PROMINENCE = 0.18   # a tip must dip this far (fraction of the bottom-edge relief) below saddle
_TIP_MIN_SEPARATION = 6      # bins two tips must be apart — merges locks that are basically one
_TIP_MAX = 6                 # never more than this many strands from one lobe

# --- Vertical sub-strand columns ------------------------------------------------------------------
# Even after tip-splitting, See-through usually hands hair back as WHOLE CURTAINS (a full-width back
# sheet, a fringe) with no alpha gaps and no distinct bottom tips — so a lobe is one strand and the whole
# sheet swings on ONE pendulum, reading like a rigid board. A real hair curtain has many physics chains
# across its width. So a lobe wider than a couple of column-widths is split into overlapping vertical
# COLUMNS: each column is its own pendulum, but the per-vertex sway weights form a partition of unity
# (linear blend between adjacent column centres), so the sheet deforms smoothly — a travelling ripple,
# not hard columns that tear. A small per-column length/mass gradient desyncs them so the ripple flows.
# Only the wide, free-HANGING curtains get columns: back hair and side hair. The front fringe frames the
# face and should sway as one piece (columns would fragment a bang line), and an already tip-split lock
# (a tail/ponytail) is its own strand and isn't fragmented further.
_COLUMN_ROLES = (SemanticRole.hair_back, SemanticRole.hair_side)
_COLUMN_WIDTH = 0.13         # target column width (model units); a lobe wider than ~2x this is split
_COLUMN_MAX = 5              # never more than this many columns from one lobe
_COLUMN_DESYNC = 0.28        # per-column length/mass spread (gradient across x) so columns flow, not lockstep


def _column_count(width: float) -> int:
    """How many vertical columns a lobe of this x-width gets: 1 until it's ~2 column-widths, then one
    per column-width, capped. Keeps a narrow tail/fringe a single strand (unchanged)."""
    if width < 2.0 * _COLUMN_WIDTH:
        return 1
    return min(_COLUMN_MAX, max(2, round(width / _COLUMN_WIDTH)))


def _column_weights(mesh: Mesh, indices: list[int], n: int) -> list[list[float]]:
    """``n`` per-vertex weight arrays (over the WHOLE mesh) partitioning the lobe ``indices`` into
    overlapping vertical columns. Column centres are evenly spaced across the lobe's x-extent; each
    vertex splits its weight linearly between the two nearest centres (a hat-function partition of
    unity), clamped at the ends. Vertices outside ``indices`` get 0 in every column."""
    xs = [mesh.vertices[i][0] for i in indices]
    x0, x1 = min(xs), max(xs)
    span = max(x1 - x0, 1e-9)
    centres = [x0 + (k + 0.5) / n * span for k in range(n)]
    weights = [[0.0] * len(mesh.vertices) for _ in range(n)]
    for i in indices:
        x = mesh.vertices[i][0]
        if x <= centres[0]:
            weights[0][i] = 1.0
        elif x >= centres[-1]:
            weights[-1][i] = 1.0
        else:
            k = max(j for j in range(n) if centres[j] <= x)
            t = (x - centres[k]) / (centres[k + 1] - centres[k])
            weights[k][i] = 1.0 - t
            weights[k + 1][i] = t
    # Drop any column that captured no vertices (a sparse mesh can leave a middle column empty). Safe:
    # a vertex only ever weights the two columns bracketing it, so both are non-empty — dropping an
    # all-zero column keeps the remaining weights a partition of unity.
    return [w for w in weights if any(v > 0.0 for v in w)]

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
    # Per-vertex sway weight over the WHOLE part mesh (0..1), for a *vertical sub-strand column* of a wide
    # sheet. See core.structure.strands._column_weights: a curtain wider than a couple of column-widths is
    # split into overlapping columns whose weights form a partition of unity, so each column gets its own
    # pendulum yet the sheet deforms smoothly (a travelling ripple) with no tear line between columns.
    # ``None`` = the strand sways uniformly over its ``vertex_indices`` (a lone lobe/tail), unchanged.
    weights: list[float] | None = None


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


def _tip_bins(contour: list[float], ref_height: float) -> list[int]:
    """Bins that are prominent local maxima (hair tips) of the smoothed bottom contour: a peak whose
    drop to the higher of its flanking valleys is at least ``_TIP_MIN_PROMINENCE`` of ``ref_height``
    (the lobe's full top-to-bottom height). Peaks closer than ``_TIP_MIN_SEPARATION`` bins are merged.

    A flat bottom edge (relief below ``_TIP_MIN_RELIEF`` of ``ref_height``) yields no tips regardless of
    its float-noise range — see the constant's note. Once the edge undulates, prominence is measured
    against that relief so a shallow-but-real set of locks still splits."""
    lo, hi = min(contour), max(contour)
    relief = hi - lo
    if ref_height <= 0 or relief < _TIP_MIN_RELIEF * ref_height:
        return []
    min_prom = _TIP_MIN_PROMINENCE * relief
    peaks = [i for i in range(1, len(contour) - 1)
             if contour[i] >= contour[i - 1] and contour[i] > contour[i + 1]]
    prominent = []
    for i in peaks:
        left = min(contour[:i]) if i else contour[i]
        right = min(contour[i + 1:]) if i + 1 < len(contour) else contour[i]
        if (contour[i] - max(left, right)) >= min_prom:
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
    ys = [y for _, y in verts]
    x0, x1 = min(xs), max(xs)
    span = x1 - x0
    lobe_height = max(ys) - min(ys)
    if span <= 1e-6 or len(indices) < 2 * _TIP_MIN_SEPARATION:
        return [indices]
    tip_bins = _tip_bins(_smooth_filled(_bottom_contour(verts, x0, span)), lobe_height)
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
    # (part_id, sway_indices|None, weights|None, height, desync) per strand unit, grouped by role.
    units: dict[SemanticRole, list[tuple[str, list[int] | None, list[float] | None, float, float]]] = \
        defaultdict(list)
    for ly in stack.layers:
        if ly.semantic_role not in HAIR_BASE_TUNING or ly.id not in mbp:
            continue
        m = mbp[ly.id]
        # First split by connected components (alpha gaps: twin-tails fused into one layer), then split
        # each connected lobe by its bottom-contour tips (locks with no gap), then split a wide lobe into
        # overlapping vertical COLUMNS (a curtain rippling on several pendulums, not one rigid board).
        sublobes: list[list[int]] = []
        for comp in mesh_components(m):
            sublobes.extend(split_lobe_by_tips(m, comp))
        multi = len(sublobes) > 1
        may_column = (ly.semantic_role in _COLUMN_ROLES) and not multi   # only a single wide curtain
        for sub in sublobes:
            h = _height_of([m.vertices[i] for i in sub])
            idx = sub if multi else None                 # bounce reference; None = the whole single lobe
            xs = [m.vertices[i][0] for i in sub]
            n = _column_count(max(xs) - min(xs)) if may_column else 1
            cols = _column_weights(m, sub, n) if n > 1 else []
            if len(cols) <= 1:                           # narrow, or collapsed to one non-empty column
                units[ly.semantic_role].append((ly.id, idx, None, h, 1.0))
            else:
                nc = len(cols)
                for k, w in enumerate(cols):
                    desync = 1.0 + _COLUMN_DESYNC * (k / (nc - 1) - 0.5)   # gradient -> travelling ripple
                    units[ly.semantic_role].append((ly.id, idx, w, h, desync))

    specs: list[StrandSpec] = []
    for role, (base, (m0, d0, l0)) in HAIR_BASE_TUNING.items():
        role_units = units.get(role)
        if not role_units:
            continue
        heights = [h for _, _, _, h, _ in role_units]
        mean = sum(heights) / len(heights)
        for i, (pid, indices, weights, h, desync) in enumerate(role_units):
            f = (h / mean) if mean > 0 else 1.0
            specs.append(StrandSpec(
                part_id=pid, param_id=strand_param_id(base, i), role=role,
                mass=m0 * f * desync, drag=d0, length=l0 * f * desync,
                vertex_indices=indices, weights=weights))
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
