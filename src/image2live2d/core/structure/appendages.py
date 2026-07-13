"""Appendage sway — dangling parts get secondary motion, driven by their kinematic parent.

Two planners share one ``AppendageSpec`` shape and one driver map:

* **accessories** (P4) — earrings, charms, ribbons, a hair bow, a waist tassel — already *follow* the
  head/body turn rigidly; P4 adds a gentle pendulum so the ornament **swings** as secondary motion,
  driven by whichever structural group the RigGraph bound it to (a head ornament with the head turn
  ``ParamAngleX``, a waist charm with the body ``ParamBodyAngleX``). Accessories are ornaments by role,
  so all of them are safe to give a bounded sway — no free-edge test needed.

* **garment appendages** (P4b) — a cape, a long sleeve, a coattail. These are ``clothing``, not
  accessories, and unlike a skirt hem they hang from the torso/shoulders, not the waist. The hard part
  is telling a *swingable* one from a rigid bodice/top: both are clothing. That is exactly what the P1
  dynamics score decides — a garment with a real **free edge** (a boundary that opens into void, not one
  glued to the torso) reads as ``gentle``/``dynamic`` and gets a body-driven pendulum; a bodice glued to
  the torso stays ``rigid`` and is left alone. The free-edge signal comes from the mesh silhouette
  (``analyze_meshes``), so this stays deterministic and Pillow-free. Skirt hems are owned by the skirt
  planner and excluded here.

Mesh-based throughout (no alpha PNG): the planners read the graph's parent + the part's mesh, so they
work in the pure pipeline and tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..types import LayerStack
from ...irr.schema import Mesh, SemanticRole
from .dynamics import DynamicsVerdict, analyze_meshes
from .graph import BODY, HEAD, RigGraph
from .skirt import _skirtable

# Gentle light pendulum base for an ornament (mass, drag, length): quick to react, short arc, quick to
# settle — an accessory is small and should read as a subtle dangle, never a flapping flag.
_ACC_TUNING = (0.9, 0.14, 0.9)

# A garment appendage (cape/sleeve/coattail) is a larger, heavier sheet than an ornament: more mass and
# a longer arc, floppier (higher drag) so it lags and settles like fabric rather than a light trinket.
_GARMENT_TUNING = (1.2, 0.20, 1.1)

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


def garment_appendages(
    stack: LayerStack, meshes: list[Mesh], graph: RigGraph, *, dynamics=None,
) -> list[AppendageSpec]:
    """One ``AppendageSpec`` per swingable clothing appendage (cape, long sleeve, coattail), in stack
    order. A candidate is a meshed ``clothing`` part that the skirt planner doesn't own (not a hem); it
    becomes an appendage only if the P1 dynamics score reads its silhouette as non-rigid — i.e. it has a
    real free edge that hangs into void, the cue that separates a cape from a bodice glued to the torso.
    Each gets a body-driven pendulum (``ParamCloth0``, ``ParamCloth1`` …); a rigid top is left alone.

    ``dynamics`` (a ``{part_id: PartDynamics}`` map) can be passed to reuse a single mesh analysis across
    author_rig + generate_physics; otherwise it is computed here. The mesh scan only runs when at least
    one non-skirt clothing candidate exists, so hair/accessory-only characters pay nothing."""
    mesh_by_part = {m.part_id: m for m in meshes}
    candidates = [ly for ly in stack.layers
                  if ly.semantic_role is SemanticRole.clothing and ly.id in mesh_by_part
                  and not _skirtable(mesh_by_part[ly.id])]
    if not candidates:
        return []
    dyn = dynamics if dynamics is not None else {d.part_id: d for d in analyze_meshes(stack, meshes)}
    specs: list[AppendageSpec] = []
    n = 0
    for ly in candidates:
        d = dyn.get(ly.id)
        if d is None or d.verdict is DynamicsVerdict.rigid:
            continue                                # a bodice/top glued to the torso: no free edge, no sway
        driver, extras = _PARENT_CFG.get(graph.parent_of(ly.id) or BODY)   # clothing rides the body
        m0, d0, l0 = _GARMENT_TUNING
        specs.append(AppendageSpec(ly.id, f"ParamCloth{n}", driver, list(extras), m0, d0, l0))
        n += 1
    return specs
