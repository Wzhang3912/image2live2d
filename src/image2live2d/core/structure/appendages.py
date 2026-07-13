"""Appendage sway (P4) — dangling accessories get secondary motion, driven by their kinematic parent.

Accessories (earrings, charms, ribbons, a hair bow, a waist tassel) already *follow* the head/body
turn rigidly (they're folded into the turn group). P4 adds a gentle pendulum on top, so a dangling
ornament **swings** with secondary motion — driven by whichever structural group the RigGraph bound it
to: a head ornament swings with the head turn (``ParamAngleX``), a waist charm with the body
(``ParamBodyAngleX``). The parent decision is exactly the graph's accessory binding, so turn-follow and
sway-drive always agree.

Mesh-based (no alpha): the planner reads the graph's parent + the part's mesh, so it works in the pure
pipeline and tests. Distinguishing a swingable clothing appendage (sleeve, cape) from a rigid top needs
the alpha free-edge signal (the P1 dynamics score) — that's a follow-up; P4 covers accessories, which
are ornaments by role and safe to give a gentle, bounded sway.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..types import LayerStack
from ...irr.schema import Mesh, SemanticRole
from .graph import BODY, HEAD, RigGraph

# Gentle light pendulum base for an ornament (mass, drag, length): quick to react, short arc, quick to
# settle — an accessory is small and should read as a subtle dangle, never a flapping flag.
_ACC_TUNING = (0.9, 0.14, 0.9)

# Per parent group: (primary driver, extra drivers). A head ornament rides the head turn; a body
# ornament the body sway. Extras (pitch/roll) enrich the motion and are present-gated downstream.
_PARENT_CFG: dict[str, tuple[str, tuple[str, ...]]] = {
    HEAD: ("ParamAngleX", ("ParamAngleY", "ParamAngleZ")),
    BODY: ("ParamBodyAngleX", ("ParamBodyAngleY", "ParamBodyAngleZ")),
}


@dataclass
class AppendageSpec:
    """One dangling accessory: its output param, the parent motion that drives its pendulum, and the
    pendulum material."""

    part_id: str
    param_id: str
    driver: str
    extra_drivers: list[str] = field(default_factory=list)
    mass: float = _ACC_TUNING[0]
    drag: float = _ACC_TUNING[1]
    length: float = _ACC_TUNING[2]


def accessory_appendages(stack: LayerStack, meshes: list[Mesh], graph: RigGraph) -> list[AppendageSpec]:
    """One ``AppendageSpec`` per meshed accessory that the graph bound to a head/body group, in stack
    order. Param ids are ``ParamAcc0``, ``ParamAcc1``, … (only parented accessories consume an index,
    so ``author_rig`` and ``generate_physics`` — both calling this — always agree). An accessory with
    no head/body to ride is skipped (nothing to drive its sway)."""
    meshed = {m.part_id for m in meshes}
    specs: list[AppendageSpec] = []
    n = 0
    for ly in stack.layers:
        if ly.semantic_role is not SemanticRole.accessory or ly.id not in meshed:
            continue
        cfg = _PARENT_CFG.get(graph.parent_of(ly.id))
        if cfg is None:
            continue
        driver, extras = cfg
        m0, d0, l0 = _ACC_TUNING
        specs.append(AppendageSpec(ly.id, f"ParamAcc{n}", driver, list(extras), m0, d0, l0))
        n += 1
    return specs
