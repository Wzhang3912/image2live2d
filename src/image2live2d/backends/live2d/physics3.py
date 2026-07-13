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


def _input_type(param_id: str) -> str:
    """Cubism physics Input type for a driver param: roll (…Z) rotates the anchor ("Angle"); horizontal
    (…X) and vertical (…Y) turns TRANSLATE it ("X"/"Y") — translation is what actually swings a strand."""
    if param_id.endswith("Z"):
        return "Angle"
    if param_id.endswith("Y"):
        return "Y"
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
            # one Input per driver (primary + extras) so all driving motion excites the pendulum.
            # Input TYPE is critical: X/Y angle params TRANSLATE the pendulum anchor (Type "X"/"Y"), which
            # actually swings a hanging strand; only roll (…Z) ROTATES it (Type "Angle"). Emitting "Angle"
            # for everything (the old bug) rotated the anchor instead of moving it -> the pendulum was
            # barely excited and the hair never visibly swung. Convention verified against Hiyori.
            "Input": [
                {"Source": {"Target": "Parameter", "Id": d}, "Weight": _INPUT_WEIGHT,
                 "Type": _input_type(d), "Reflect": False}
                for d in ph.all_drivers()
            ],
            "Output": [{
                "Destination": {"Target": "Parameter", "Id": ph.output_param},
                "VertexIndex": len(verts),  # 1-based: the swinging tip drives the output
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
