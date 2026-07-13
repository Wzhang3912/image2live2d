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

from dataclasses import dataclass

from ..types import LayerStack
from ...irr.schema import Mesh, SemanticRole

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
    """One hair strand: the part it deforms, its output param id, and its pendulum material."""

    part_id: str
    param_id: str
    role: SemanticRole
    mass: float
    drag: float
    length: float


def _height(m: Mesh) -> float:
    ys = [y for _, y in m.vertices]
    return (max(ys) - min(ys)) if ys else 0.0


def strand_param_id(base: str, index: int) -> str:
    """First part keeps the base id; extras get a 1-based numeric suffix (base, base2, base3, …)."""
    return base if index == 0 else f"{base}{index + 1}"


def hair_strands(stack: LayerStack, meshes: list[Mesh]) -> list[StrandSpec]:
    """One ``StrandSpec`` per meshed hair part, in (role, stack) order — the deterministic plan both
    ``author_rig`` (sway keyforms) and ``generate_physics`` (pendulums) consume so their param ids
    always agree. Mass/length scale with each strand's height vs its role's mean (factor 1.0 for a
    lone/average strand → base tuning unchanged)."""
    mbp = {m.part_id: m for m in meshes}
    specs: list[StrandSpec] = []
    for role, (base, (m0, d0, l0)) in HAIR_BASE_TUNING.items():
        parts = [ly.id for ly in stack.layers if ly.semantic_role == role and ly.id in mbp]
        if not parts:
            continue
        heights = [_height(mbp[p]) for p in parts]
        mean = sum(heights) / len(heights)
        for i, pid in enumerate(parts):
            f = (heights[i] / mean) if mean > 0 else 1.0
            specs.append(StrandSpec(part_id=pid, param_id=strand_param_id(base, i), role=role,
                                    mass=m0 * f, drag=d0, length=l0 * f))
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
