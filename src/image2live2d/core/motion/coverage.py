"""Stage 5b — Motion. **Coverage**: does the shipped motion actually exercise the rig we built?

This exists because of a specific, embarrassing failure. We spent four PRs tuning hair, accessory and
cloth pendulums against a real pro rig, emitted the model, and watched it in Cubism Viewer — and the
idle drove 6 of 31 parameters and never once touched ``ParamAngleX/Y/Z``. Five of the eight physics
chains had a driver that was flat-lined for the entire loop. The physics was *correct* and *connected
to nothing*, and no test could tell, because every test we had asked "is the rig well-formed?" and none
asked "does the motion move it?".

So: three properties, checked from the IRR, no runtime required.

* **Every parameter is driven.** A parameter that no clip ever moves is dead weight — it will never be
  seen unless a human happens to drag that slider.
* **Every physics chain is excited.** A pendulum is driven by its input parameter *changing*. A driver
  that never moves, or moves by a hair's width of its range, is a pendulum that will never swing —
  which looks exactly like a pendulum that doesn't work.
* **No clip keys a physics output.** ``ParamHairFront`` is *written* by the pendulum every frame. A
  keyframe on it is a lane arguing with the simulation, and the simulation wins — so the lane is dead
  and, worse, it reads like the hair is animated when it is not.

Coverage is deliberately measured over the *natural* motion (idle, expressions, interaction clips) and
**not** over the diagnostic ``sweep``, which drives everything by construction. A parameter whose only
exercise is the sweep has no motion that uses it, and that is a finding, not a pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...irr.schema import Animation, Parameter, PhysicsRig

# A driver has to move by a real share of its own range to swing anything. The idle's body sway was
# 0.5 degrees on a +/-10 parameter — 5% of the range, technically "driven", visibly nothing. Below this
# fraction we call the chain unexcited, because on screen it is.
MIN_SWING_FRAC = 0.10


@dataclass
class MotionCoverage:
    """What the motion set does and does not reach."""

    driven: dict[str, float] = field(default_factory=dict)   # param id -> swing, as a frac of its range
    undriven: list[str] = field(default_factory=list)        # no clip moves it at all
    unexcited: list[str] = field(default_factory=list)       # physics chains whose driver barely moves
    keyed_outputs: list[str] = field(default_factory=list)   # a clip keying a physics output

    @property
    def ok(self) -> bool:
        return not (self.undriven or self.unexcited or self.keyed_outputs)

    def format(self) -> str:
        if self.ok:
            return f"motion coverage OK — {len(self.driven)} parameters driven"
        lines = []
        if self.undriven:
            lines.append(f"undriven params ({len(self.undriven)}): {', '.join(self.undriven)}")
        if self.unexcited:
            lines.append(f"unexcited physics ({len(self.unexcited)}): {', '.join(self.unexcited)}")
        if self.keyed_outputs:
            lines.append(f"clips keying a physics output: {', '.join(self.keyed_outputs)}")
        return "\n".join(lines)


def motion_coverage(
    parameters: list[Parameter],
    physics: list[PhysicsRig],
    animations: list[Animation],
) -> MotionCoverage:
    """Measure what ``animations`` actually move. Pure function of the IRR."""
    ranges = {p.id: max(p.max - p.min, 1e-9) for p in parameters}
    outputs = {r.output_param for r in physics}

    swing: dict[str, float] = {}
    keyed_outputs: list[str] = []
    for anim in animations:
        for lane in anim.lanes:
            if lane.param_id in outputs and lane.param_id not in keyed_outputs:
                keyed_outputs.append(lane.param_id)
            values = [kf.value for kf in lane.keyframes]
            if not values:
                continue
            # a flat lane is not motion: holding a parameter at one value drives nothing
            travel = (max(values) - min(values)) / ranges.get(lane.param_id, 1.0)
            swing[lane.param_id] = max(swing.get(lane.param_id, 0.0), travel)

    driven = {pid: s for pid, s in swing.items() if s > 0.0}

    # Physics outputs are excluded from "undriven": they are *supposed* to be moved by the pendulum,
    # not by a lane. They are covered iff their chain is excited, which is the next check.
    undriven = [p.id for p in parameters
                if p.id not in outputs and p.max > p.min and driven.get(p.id, 0.0) <= 0.0]

    unexcited: list[str] = []
    for rig in physics:
        drivers = [rig.driver_param, *rig.extra_drivers]
        if not any(driven.get(d, 0.0) >= MIN_SWING_FRAC for d in drivers):
            unexcited.append(rig.output_param)

    return MotionCoverage(
        driven=driven, undriven=undriven, unexcited=unexcited, keyed_outputs=keyed_outputs,
    )
