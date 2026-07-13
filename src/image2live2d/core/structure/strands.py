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
        comps = mesh_components(m)
        if len(comps) <= 1:
            units[ly.semantic_role].append((ly.id, None, _height_of(m.vertices)))
        else:
            for comp in comps:
                h = _height_of([m.vertices[i] for i in comp])
                units[ly.semantic_role].append((ly.id, comp, h))

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
