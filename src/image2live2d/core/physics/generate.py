"""Stage 5 — Physics. Procedurally generate pendulum physics for hair/cloth.

Each ``PhysicsRig`` is a Live2D-style input->output pendulum: a driver parameter (head/body motion)
swings an output parameter (``ParamHairFront`` etc.) through a spring/damper. The corresponding
*output* parameters and their sway deformation are authored in ``rig.author_rig``; this stage just
wires the pendulums. We only emit a rig when both its driver and output parameters already exist, so
the result always satisfies the IRR's referential integrity.
"""

from __future__ import annotations

from ..structure.appendages import accessory_appendages, garment_appendages
from ..structure.graph import build_rig_graph
from ..structure.skirt import skirt_specs_from_params, skirt_zones
from ..structure.strands import HAIR_BASE_TUNING, hair_specs_from_params, hair_strands
from ..types import LayerStack
from ...irr.schema import Mesh, Parameter, PhysicsModel, PhysicsRig, SemanticRole

# Per-strand hair tuning (base values + geometry scaling) now lives in core.structure.strands so
# author_rig and this stage agree on the strand param ids. Back hair is heavier/slower, front fringe
# lighter/snappier: higher mass -> more lag behind the head, lower drag -> longer follow-through.
# Per-role vertical hair-bounce output params (see core.rig.author._hair_bounce). Kept here so the
# physics stage and the authoring stage agree on the ids without a cross-import.
_HAIR_BOUNCE_OUT = {
    SemanticRole.hair_front: "ParamHairFrontV",
    SemanticRole.hair_side: "ParamHairSideV",
    SemanticRole.hair_back: "ParamHairBackV",
}

_HEAD_DRIVER = "ParamAngleX"  # head turn drives hair sway

_BODY_DRIVER = "ParamBodyAngleX"

# Skirt is modelled as a springy multi-zone cloth: each hem zone is a SpringPendulum driven by ALL
# the relevant lower-body motion — body sway (primary) plus the nearest leg and body lean — so any
# lower-body interaction swings the cloth. Zone windows, drivers, and geometry-scaled material all come
# from core.structure.skirt (see skirt_zones / skirt_specs_from_params).


def generate_physics(
    stack: LayerStack, parameters: list[Parameter], *, meshes: list[Mesh] | None = None,
) -> list[PhysicsRig]:
    """Create pendulum ``PhysicsRig`` entries for each hair/cloth output parameter that was authored.

    ``parameters`` is the authored parameter list (from ``author_rig``); a rig is only created when
    its driver and output parameters actually exist (keeping the IRR's referential integrity intact).
    Hair = one pendulum per strand driven by head turn; skirt = springy multi-zone cloth driven by the
    whole lower body. When ``meshes`` are supplied, each strand's mass/length is geometry-scaled (a
    longer tail lags more); without them the strand's base (role) tuning is used — identical for a
    single strand, so callers that don't thread meshes stay byte-compatible.
    """
    param_ids = {p.id for p in parameters}
    rigs: list[PhysicsRig] = []

    if _HEAD_DRIVER in param_ids:
        # Horizontal sway reacts to the head turn: yaw (X, primary) + roll (Z, tips gravity). Pitch is
        # NOT an extra driver here — a "Y" input is inert for this angle output (it slides the anchor
        # down its own string), so it only ever bloated the setting; the nod bob is a separate vertical
        # chain below. One pendulum per strand (P2) so twin-tails / fringe swing independently.
        head_extra = [d for d in ("ParamAngleZ",) if d in param_ids]
        specs = (hair_strands(stack, meshes) if meshes is not None
                 else hair_specs_from_params(param_ids))
        for s in specs:
            if s.param_id in param_ids:
                rigs.append(PhysicsRig(id=f"phys_{s.param_id}", driver_param=_HEAD_DRIVER,
                                       output_param=s.param_id, extra_drivers=head_extra,
                                       mass=s.mass, drag=s.drag, length=s.length))

        # Vertical bounce: one pendulum per hair ROLE, driven by pitch. ParamAngleY tips the strand's
        # gravity (pitch_angle=True -> emitted as an Angle input), so a nod swings it and it settles,
        # driving the role's ParamHair*V output straight down. This is the motion the horizontal sway
        # chains structurally cannot produce. Role tuning matches the sway strands (back heavy/slow).
        for role, (base, (mass, drag, length)) in HAIR_BASE_TUNING.items():
            out = _HAIR_BOUNCE_OUT[role]
            if out in param_ids and "ParamAngleY" in param_ids:
                rigs.append(PhysicsRig(id=f"phys_{out}", driver_param="ParamAngleY",
                                       output_param=out, pitch_angle=True,
                                       mass=mass, drag=drag, length=length))

    # Skirt zones: primary driver = body sway (fall back to head turn if no body param at all). Material
    # is geometry-scaled when meshes are threaded through (a longer/wider garment swings bigger/slower),
    # else base tuning — identical for a reference-sized garment, so mesh-less callers stay compatible.
    primary = _BODY_DRIVER if _BODY_DRIVER in param_ids else (
        _HEAD_DRIVER if _HEAD_DRIVER in param_ids else None
    )
    if primary:
        zones = (skirt_zones(stack, meshes) if meshes is not None
                 else skirt_specs_from_params(param_ids))
        for z in zones:
            if z.param_id not in param_ids:
                continue
            drivers = [e for e in z.extra_drivers if e in param_ids and e != primary]
            rigs.append(PhysicsRig(
                id=f"phys_{z.param_id}", driver_param=primary, output_param=z.param_id,
                extra_drivers=drivers, model=PhysicsModel.spring_pendulum,
                mass=z.mass, drag=z.drag, length=z.length,
            ))

    # Accessory dangles: one pendulum per ornament, driven by the parent (head/body) the graph bound
    # it to. Needs meshes to build the graph; the ids/drivers match author_rig's accessory sway params.
    if meshes is not None:
        graph = build_rig_graph(stack, meshes)
        for a in accessory_appendages(stack, meshes, graph):
            if a.param_id not in param_ids or a.driver not in param_ids:
                continue
            extras = [e for e in a.extra_drivers if e in param_ids]
            rigs.append(PhysicsRig(id=f"phys_{a.param_id}", driver_param=a.driver,
                                   output_param=a.param_id, extra_drivers=extras,
                                   mass=a.mass, drag=a.drag, length=a.length))
        # Garment appendages (cape/sleeve/coattail): a body-driven springy cloth pendulum, like the
        # skirt but hanging from the torso/shoulders. Only the free-edged garments reach here.
        for g in garment_appendages(stack, meshes, graph):
            if g.param_id not in param_ids or g.driver not in param_ids:
                continue
            extras = [e for e in g.extra_drivers if e in param_ids]
            rigs.append(PhysicsRig(id=f"phys_{g.param_id}", driver_param=g.driver,
                                   output_param=g.param_id, extra_drivers=extras,
                                   model=PhysicsModel.spring_pendulum,
                                   mass=g.mass, drag=g.drag, length=g.length))
    return rigs
