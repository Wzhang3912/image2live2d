"""Stage 5 — Physics. Procedurally generate pendulum physics for hair/cloth.

Each ``PhysicsRig`` is a Live2D-style input->output pendulum: a driver parameter (head/body motion)
swings an output parameter (``ParamHairFront`` etc.) through a spring/damper. The corresponding
*output* parameters and their sway deformation are authored in ``rig.author_rig``; this stage just
wires the pendulums. We only emit a rig when both its driver and output parameters already exist, so
the result always satisfies the IRR's referential integrity.
"""

from __future__ import annotations

from ..types import LayerStack
from ...irr.schema import Parameter, PhysicsModel, PhysicsRig

# output param -> (mass, drag, length): back hair is heavier/slower, front fringe lighter/snappier.
# Tuned for VISIBLE secondary motion (the hair reads as "alive"): higher mass -> more lag behind the head,
# lower drag -> the strand keeps swinging and settles slowly (follow-through), longer length -> bigger arc.
_HAIR_TUNING: dict[str, tuple[float, float, float]] = {
    "ParamHairFront": (1.1, 0.10, 1.05),   # light fringe: quick to react but with a visible settle
    "ParamHairSide": (1.4, 0.08, 1.30),
    "ParamHairBack": (2.0, 0.06, 1.70),    # heavy back hair: big, slow, long-settling swing
}
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


def generate_physics(stack: LayerStack, parameters: list[Parameter]) -> list[PhysicsRig]:
    """Create pendulum ``PhysicsRig`` entries for each hair/cloth output parameter that was authored.

    ``parameters`` is the authored parameter list (from ``author_rig``); a rig is only created when
    its driver and output parameters actually exist (keeping the IRR's referential integrity intact).
    Hair = rigid pendulums driven by head turn; skirt = springy multi-zone cloth driven by the whole
    lower body.
    """
    param_ids = {p.id for p in parameters}
    rigs: list[PhysicsRig] = []

    if _HEAD_DRIVER in param_ids:
        # Hair reacts to the WHOLE head turn: yaw (X, primary) + pitch (Y) + roll (Z). The emitter
        # maps pitch to vertical anchor motion so a nod bobs the hair, not just a side sway.
        head_extra = [d for d in ("ParamAngleY", "ParamAngleZ") if d in param_ids]
        for output, (mass, drag, length) in _HAIR_TUNING.items():
            if output in param_ids:
                rigs.append(PhysicsRig(id=f"phys_{output}", driver_param=_HEAD_DRIVER,
                                       output_param=output, extra_drivers=head_extra,
                                       mass=mass, drag=drag, length=length))

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
