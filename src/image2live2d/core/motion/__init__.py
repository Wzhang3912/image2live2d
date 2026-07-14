"""Stage 5b — Motion. Everything that *moves* the rig once it is built.

Four pieces, all authoring backend-neutral ``Animation``s into the IRR:

* :mod:`.idle` — the looping idle (blink, breath, head drift, body sway). The head drift matters more
  than it sounds: it is what excites the hair and accessory pendulums at rest.
* :mod:`.expressions` — the expression sheet (smile / surprise / sad / angry).
* :mod:`.drive` — the **drive sheet**: interaction clips that isolate each axis of the rig, plus a
  ``sweep`` that walks every parameter through its range. This is the motion you *verify* with.
* :mod:`.coverage` — checks the shipped motion actually exercises the rig: every parameter driven,
  every physics chain excited, no clip fighting a pendulum by keying its output.
"""

from __future__ import annotations

from .coverage import MIN_SWING_FRAC, MotionCoverage, motion_coverage
from .drive import DRIVE_NAMES, SWEEP_NAME, generate_drives, generate_sweep
from .expressions import EXPRESSION_NAMES, generate_expressions
from .idle import (
    FPS,
    generate_idle,
    IDLE_FRAMES,
)
from ...irr.schema import Animation, Parameter, PhysicsRig


def generate_all(
    parameters: list[Parameter], physics: list[PhysicsRig] | None = None,
) -> list[Animation]:
    """The complete motion set for a rig: idle + expressions + drive sheet + sweep.

    One entry point on purpose. The web app used to author its own idle-plus-expressions by hand, which
    is how the bundle you downloaded from the browser ended up with different motion from the one the
    CLI emitted — and the drive clips, the whole point of which is that a human can *see* the rig, would
    have been missing from the only build a human actually opens.
    """
    return (
        generate_idle(parameters)
        + generate_expressions(parameters)
        + generate_drives(parameters, physics)
        + generate_sweep(parameters, physics)
    )


__all__ = [
    "generate_all",
    "FPS",
    "generate_idle",
    "IDLE_FRAMES",
    "generate_expressions",
    "EXPRESSION_NAMES",
    "generate_drives",
    "generate_sweep",
    "DRIVE_NAMES",
    "SWEEP_NAME",
    "motion_coverage",
    "MotionCoverage",
    "MIN_SWING_FRAC",
]
