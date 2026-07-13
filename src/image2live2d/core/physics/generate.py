"""Stage 5 — Physics. Procedurally generate pendulum physics for hair/cloth.

Each ``PhysicsRig`` is a Live2D-style input->output pendulum: a driver parameter (head/body motion)
swings an output parameter (``ParamHairFront`` etc.) through a spring/damper. The corresponding
*output* parameters and their sway deformation are authored in ``rig.author_rig``; this stage just
wires the pendulums. We only emit a rig when both its driver and output parameters already exist, so
the result always satisfies the IRR's referential integrity.
"""

from __future__ import annotations

from ..structure.strands import hair_specs_from_params, hair_strands
from ..types import LayerStack
from ...irr.schema import Mesh, Parameter, PhysicsModel, PhysicsRig

# Per-strand hair tuning (base values + geometry scaling) now lives in core.structure.strands so
# author_rig and this stage agree on the strand param ids. Back hair is heavier/slower, front fringe
# lighter/snappier: higher mass -> more lag behind the head, lower drag -> longer follow-through.
_HEAD_DRIVER = "ParamAngleX"  # head turn drives hair sway

_BODY_DRIVER = "ParamBodyAngleX"

# Skirt is modelled as a springy multi-zone cloth: each hem zone is a SpringPendulum driven by ALL
# the relevant lower-body motion — body sway (primary) plus the nearest leg and body lean — so any
# lower-body interaction swings the cloth. (output_param, extra_drivers, (mass, drag, length)).
# Center is heavier/longer (more fabric); side zones lighter and coupled to their leg.
_SKIRT_ZONES: list[tuple[str, list[str], tuple[float, float, float]]] = [
    ("ParamSkirtL", ["ParamLegLA", "ParamBodyAngleZ"], (1.5, 0.28, 1.3)),
    ("ParamSkirtC", ["ParamBodyAngleZ", "ParamBodyAngleY"], (1.8, 0.25, 1.5)),
    ("ParamSkirtR", ["ParamLegRA", "ParamBodyAngleZ"], (1.5, 0.28, 1.3)),
]


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
        # Hair reacts to the WHOLE head turn: yaw (X, primary) + pitch (Y) + roll (Z). The emitter
        # maps pitch to vertical anchor motion so a nod bobs the hair, not just a side sway. One
        # pendulum per strand (P2) so twin-tails / fringe swing independently.
        head_extra = [d for d in ("ParamAngleY", "ParamAngleZ") if d in param_ids]
        specs = (hair_strands(stack, meshes) if meshes is not None
                 else hair_specs_from_params(param_ids))
        for s in specs:
            if s.param_id in param_ids:
                rigs.append(PhysicsRig(id=f"phys_{s.param_id}", driver_param=_HEAD_DRIVER,
                                       output_param=s.param_id, extra_drivers=head_extra,
                                       mass=s.mass, drag=s.drag, length=s.length))

    # Skirt zones: primary driver = body sway (fall back to head turn if no body param at all).
    primary = _BODY_DRIVER if _BODY_DRIVER in param_ids else (
        _HEAD_DRIVER if _HEAD_DRIVER in param_ids else None
    )
    if primary:
        for output, extras, (mass, drag, length) in _SKIRT_ZONES:
            if output not in param_ids:
                continue
            drivers = [e for e in extras if e in param_ids and e != primary]
            rigs.append(PhysicsRig(
                id=f"phys_{output}", driver_param=primary, output_param=output,
                extra_drivers=drivers, model=PhysicsModel.spring_pendulum,
                mass=mass, drag=drag, length=length,
            ))
    return rigs
