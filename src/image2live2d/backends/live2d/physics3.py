"""IRR ``Rig.physics`` -> Cubism ``.physics3.json`` (open JSON).

Each IRR ``PhysicsRig`` (a driver-param -> output-param pendulum) becomes one Cubism
``PhysicsSetting`` with an **Angle** input from the driver and an **Angle** output to the output
param, plus a 2-vertex pendulum (fixed root + swinging tip) whose mobility/delay/acceleration are
derived from the rig's mass/drag/length.

The exact swing feel (the tuning constants below) can only be judged in a Live2D runtime — like the
nijilive physics constants, these are first-pass values. The *structure* is what's verified here.
"""

from __future__ import annotations

from ...irr.schema import Rig

PHYSICS_VERSION = 3

# Tuning (first-pass; verify in a Live2D runtime).
_INPUT_WEIGHT = 60.0       # Cubism input weight (how strongly the driver swings the pendulum)
_OUTPUT_WEIGHT = 100.0     # Cubism output weight
_LENGTH_UNITS = 12.0       # IRR pendulum length (~1) -> Cubism position units
_NORM = {                  # standard Cubism normalization ranges
    "Position": {"Minimum": -10.0, "Default": 0.0, "Maximum": 10.0},
    "Angle": {"Minimum": -10.0, "Default": 0.0, "Maximum": 10.0},
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _vertices(mass: float, drag: float, length: float) -> list[dict]:
    """A fixed root + a swinging tip. Mobility falls with drag; delay/acceleration scale with mass.

    The clamp ranges are **calibrated to real pro physics3.json** (Hiyori + Akari, read via
    tools/feel_parity.py): Mobility 0.71-1.00, Delay 0.60-1.00, Acceleration 0.80-3.00 (typically
    1.0-2.0), Radius = tip Y. The pre-calibration values fell outside these (Delay up to 1.2,
    Acceleration down to 0.5 — under-driven hair, Mobility down to 0.61 — over-damped cloth), so a real
    Live2D runtime would swing our parts more weakly/stiffly than an artist's. Keep these in step with
    nijilive's puppet.py pendulum (see the FEEL-PARITY note there)."""
    mobility = _clamp(1.0 - drag, 0.72, 0.99)            # real Hiyori/Akari union: >= ~0.71
    delay = _clamp(0.4 + 0.4 * mass, 0.60, 1.00)         # real: 0.60-1.00 (never > 1.0)
    accel = _clamp(1.5 / max(mass, 0.1), 1.00, 2.00)     # real: ~1.0-2.0 gravity gain (was < 1 -> weak)
    tip_y = max(0.1, length) * _LENGTH_UNITS
    return [
        {"Position": {"X": 0.0, "Y": 0.0}, "Mobility": 1.0, "Delay": 1.0,
         "Acceleration": 1.0, "Radius": 0.0},
        {"Position": {"X": 0.0, "Y": tip_y}, "Mobility": mobility, "Delay": delay,
         "Acceleration": accel, "Radius": tip_y},
    ]


def _input_type(param_id: str, *, pitch_angle: bool = False) -> str | None:
    """Cubism physics Input type for a driver param, or ``None`` if the param cannot drive a pendulum.

    Roll (…Z) tips the gravity vector ("Angle"); yaw (…X) translates the anchor sideways ("X"), and
    sideways translation is what actually swings a strand.

    **Pitch (…Y) is inert for a horizontal sway chain, and dropped there.** A "Y" input slides the
    pendulum's anchor straight *down its own string*, and the sway output is ``Type: "Angle"`` — an angle
    off the strand's rest direction. Sliding a pivot along the string it hangs from does not change that
    angle, so the input contributed exactly zero (``tools/physics_excite.py``: ``head_pitch`` moved 0 of
    8 sway chains), and neither Hiyori nor Akari emits a "Y" input.

    The honest fix for a nod-bounce is a *vertical* output param (``ParamHair*V``) on its own chain, fed
    pitch as an **Angle** input (``pitch_angle=True``): a nod tips the strand's gravity, so it swings and
    settles, and that swing maps to a straight-down hair drop. Tipping gravity is the one input that
    moves a down-hanging pendulum from a nod (validated in the simulator: Y peaks 0.0, Angle peaks 0.23
    and overshoots). Only the dedicated bounce chains set this; the sway chains still drop pitch.
    """
    if pitch_angle and param_id.endswith("Y"):
        return "Angle"
    if param_id.endswith("Z"):
        return "Angle"
    if param_id.endswith("Y"):
        return None
    return "X"


def physics3(rig: Rig) -> dict:
    """Build the ``.physics3.json`` document for ``rig`` (empty settings if it has no physics)."""
    settings: list[dict] = []
    dictionary: list[dict] = []
    total_vertices = 0

    for i, ph in enumerate(rig.physics, start=1):
        setting_id = f"PhysicsSetting{i}"
        verts = _vertices(ph.mass, ph.drag, ph.length)
        total_vertices += len(verts)
        settings.append({
            "Id": setting_id,
            # One Input per driver (primary + extras) that can actually drive a pendulum. Input TYPE is
            # critical: yaw (…X) TRANSLATES the anchor sideways, which is what swings a hanging strand;
            # roll (…Z) tips gravity (Type "Angle"). Emitting "Angle" for everything (an older bug)
            # rotated the anchor instead of moving it, and the hair never visibly swung. Pitch (…Y) is
            # dropped entirely — see _input_type: it is a mathematical no-op for an Angle output, and no
            # real rig emits one.
            "Input": [
                {"Source": {"Target": "Parameter", "Id": d}, "Weight": _INPUT_WEIGHT,
                 "Type": t, "Reflect": False}
                for d in ph.all_drivers() if (t := _input_type(d, pitch_angle=ph.pitch_angle))
            ],
            "Output": [{
                "Destination": {"Target": "Parameter", "Id": ph.output_param},
                # The swinging TIP drives the output — and it is a **0-based index into the particle
                # array**, so the tip is len(verts) - 1, not len(verts). We wrote len(verts) for months.
                # It did not crash, and that is exactly why it survived: the Cubism runtime keeps every
                # setting's particles in one contiguous buffer and does
                # `particles[VertexIndex] - particles[VertexIndex - 1]` with no bounds check, so an index
                # one past our chain quietly read *the next chain's root particle*. Every hair output was
                # being computed from a neighbouring pendulum, and the last chain read off the end of the
                # buffer entirely. The hair still moved, which is the whole trap — it moved for the wrong
                # reason. Verified against both real models on hand: Hiyori and Akari never emit a
                # VertexIndex >= len(Vertices) (Hiyori taps 1..7 of an 11-particle chain).
                "VertexIndex": len(verts) - 1,
                # Hair gets a higher output scale so the physics swing reaches a visibly larger sway (more
                # "alive"); cloth/skirt stays at unity to avoid over-driving the fabric.
                "Scale": 1.4 if ph.output_param.startswith("ParamHair") else 1.0,
                "Weight": _OUTPUT_WEIGHT,
                "Type": "Angle",
                "Reflect": False,
            }],
            "Vertices": verts,
            "Normalization": _NORM,
        })
        dictionary.append({"Id": setting_id, "Name": ph.output_param})

    # Gravity/wind: take the first rig's (all are (0,-1)/(0,0) by default).
    gx, gy = (rig.physics[0].gravity if rig.physics else (0.0, -1.0))
    wx, wy = (rig.physics[0].wind if rig.physics else (0.0, 0.0))

    return {
        "Version": PHYSICS_VERSION,
        "Meta": {
            "PhysicsSettingCount": len(settings),
            "TotalInputCount": sum(len(s["Input"]) for s in settings),
            "TotalOutputCount": sum(len(s["Output"]) for s in settings),
            "VertexCount": total_vertices,
            "EffectiveForces": {
                "Gravity": {"X": gx, "Y": gy},
                "Wind": {"X": wx, "Y": wy},
            },
            "PhysicsDictionary": dictionary,
        },
        "PhysicsSettings": settings,
    }
