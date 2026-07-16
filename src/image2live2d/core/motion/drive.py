"""Stage 5b — Motion. The **drive sheet**: clips whose job is to *excite the rig*.

A puppet is only as verifiable as the motion you have to look at it with. The idle we shipped moved 6
of a real character's 31 parameters and never touched ``ParamAngleX/Y/Z`` — which is what every hair
and accessory pendulum is driven by. So five of eight physics chains had a **flat-lined driver**: you
could have deleted the entire physics block and the idle would have rendered identically. Watching it
proved nothing, and the bugs the human found in Cubism Viewer had to be found by dragging sliders by
hand.

This module authors the motion that a rig has to survive:

* **Interaction clips** — one per axis of the rig (head yaw/pitch/roll, body sway, arms, legs, talk,
  look, brows). Each one isolates a part of the rig, so when something looks wrong you already know
  which parameter did it.
* **A sweep clip** — every parameter, one after another, through its full range. Press play once and
  the whole rig has been exercised in a known order.

Two design rules do most of the work:

1. **A pendulum is excited by velocity, not position.** A slow ease into a pose lets the hair track the
   head exactly, and glued-on hair looks identical to physically-simulated hair. So every clip is shaped
   **snap → hold → snap back → hold → release → settle**: the snap injects the impulse, and the *hold*
   is when you see the follow-through. The long settle at the tail — driver at rest, hair still moving —
   is the single most diagnostic stretch of frames in the whole sheet. Nothing there means nothing
   works.
2. **Never key a physics output.** ``ParamHairFront`` and friends are *written* by the pendulum; a
   keyframe on one is motion fighting physics, and physics wins. They are excluded by construction,
   and ``motion.coverage`` fails the rig if one ever slips in.

Clips go to the extremes of each range on purpose. Extremes are where an auto-rigged deformation is
least trustworthy — the cardboard-arm stretch was found at the end of ``ParamAngleX``'s travel, not in
the middle of it.

Like the idle and the expression sheet, every lane is present-gated (a bare portrait gets the face
clips and no limb clips) and clamped to its parameter's own range.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...irr.schema import (
    AnimKeyframe,
    Animation,
    AnimationLane,
    InterpolateMode,
    Parameter,
    PhysicsRig,
)

FPS = 60.0

# The impulse. Fast — this is the whole point: a pendulum integrates the *rate* its driver changes at,
# so 8 frames (~0.13 s) to the pose swings the hair, where a 1-second ease would not.
_SNAP = 8
# The hold. Long enough for a pendulum to lag, overshoot and come to rest while the driver stands still,
# which is the only interval in which physics and no-physics look different.
_HOLD = 40
# The tail. Driver back at neutral, model still settling — if the hair is frozen here, it is not rigged.
_SETTLE = 60


@dataclass(frozen=True)
class Drive:
    """One interaction clip: a pose, ping-ponged.

    ``pose`` maps a standard parameter id to a **signed fraction of its range**: ``+1.0`` drives it to
    its maximum, ``-0.5`` half-way to its minimum. Opposite signs within one pose are what make
    ``arms_swing`` a swing rather than a shrug — the left arm goes forward as the right goes back.
    """

    pose: dict[str, float]
    cycles: int = 1               # ping-pongs before the settle; >1 = a repeated, shaking motion
    hold: int = _HOLD             # frames held at each extreme
    settle: int = _SETTLE         # frames at neutral at the end, watching the physics come to rest
    note: str = ""                # what this clip is *for* — what you are supposed to be looking at
    # Whether the swing reverses through the *opposite* pose or just returns to neutral. Most motions
    # are symmetric (a head yaws both ways). But some are one-directional: legs splayed outward look
    # like a stance, while the mirror pose swings them *inward* — two close-together legs then cross
    # into an X. A one-directional drive splays out and comes back, never through the crossing pose.
    bidirectional: bool = True


# The sheet. Each clip isolates one axis of the rig so a defect is attributable: if the hair only fails
# on `head_roll`, that is a very different bug from failing on all three.
_DRIVES: dict[str, Drive] = {
    # --- head: the driver of every hair and head-accessory pendulum -------------------------------
    "head_yaw": Drive({"ParamAngleX": 1.0},
                      note="hair sways side to side and lags the turn; earrings swing"),
    "head_pitch": Drive({"ParamAngleY": 1.0},
                        note="a nod bobs the hair vertically (the emitter maps pitch to anchor Y)"),
    "head_roll": Drive({"ParamAngleZ": 1.0},
                       note="a tilt rolls the hair; the fringe should not slide off the forehead"),
    # The hardest thing you can ask of a hair pendulum: reverse the driver before the previous swing
    # has settled. Under-damped hair whips; over-damped hair reads as a helmet. Both show up here and
    # nowhere else.
    "head_shake": Drive({"ParamAngleX": 0.8}, cycles=4, hold=4,
                        note="rapid reversals — hair should lag and trail, not snap rigidly along"),

    # --- body: the driver of the skirt / cloth zones ----------------------------------------------
    "body_sway": Drive({"ParamBodyAngleX": 1.0, "ParamBodyAngleZ": 0.5},
                       note="skirt/cloth zones swing and settle; the hem should lag the hips"),
    "body_bow": Drive({"ParamBodyAngleY": 1.0},
                      note="lean in/out — cloth should fall, not shear with the torso"),

    # --- limbs: the de-cardboard check ------------------------------------------------------------
    # Both arms used to arrive as a single layer, so they could only ever move as one sheet. These two
    # clips are how you confirm they were separated: if the arms move together here, the split failed.
    # Both arms to +max = both lift outward together (mirror-symmetric convention), a symmetric raise.
    # One-directional: raise and return, not down through the inward/crossed mirror pose.
    "arms_raise": Drive({"ParamArmLA": 1.0, "ParamArmLB": 0.7,
                         "ParamArmRA": 1.0, "ParamArmRB": 0.7},
                        bidirectional=False,
                        note="both arms lift outward together — sleeves must ride their own arm"),
    "arms_swing": Drive({"ParamArmLA": 1.0, "ParamArmRA": -1.0},
                        note="opposite phase — proves left and right are separate parts"),
    # Both legs to +max = splay OUTWARD (the limb convention is mirror-symmetric: +param lifts/splays
    # each side away from the midline). Two close-together legs rotated toward each other would cross
    # into an X, so this splays them to a widening stance and returns — never through the crossing pose.
    "legs_swing": Drive({"ParamLegLA": 1.0, "ParamLegLB": 0.6,
                         "ParamLegRA": 1.0, "ParamLegRB": 0.6},
                        bidirectional=False,
                        note="both legs splay outward and back (never through the crossing pose) — "
                             "proves the legs were cut at the crotch seam"),

    # --- face -------------------------------------------------------------------------------------
    "talk": Drive({"ParamMouthOpenY": 1.0}, cycles=3, hold=10, settle=20,
                  note="the mouth must actually open — a cavity, teeth and tongue behind the lips"),
    "smirk": Drive({"ParamMouthForm": 1.0},
                   note="corners up and down without the lips tearing from the face"),
    "look": Drive({"ParamEyeBallX": 1.0, "ParamEyeBallY": 1.0},
                  note="pupils travel inside the eye and never cross the lid"),
    "blink": Drive({"ParamEyeLOpen": -1.0, "ParamEyeROpen": -1.0}, cycles=2, hold=6, settle=20,
                   note="the eye squashes shut but never vanishes (a full collapse zeroes its area)"),
    "brows": Drive({"ParamBrowLY": 1.0, "ParamBrowRY": 1.0},
                   note="both brows read *through* the fringe — the right one used to drive nothing"),
}

DRIVE_NAMES: tuple[str, ...] = tuple(_DRIVES)

# The diagnostic clip: every parameter, one at a time. Named separately because it is not *motion* —
# it is an inspection tool, and `motion.coverage` deliberately does not count it as coverage. A rig
# whose only exercise of a parameter is the sweep has no natural motion that uses it, and we want to
# know that rather than have the sweep paper over it.
SWEEP_NAME = "sweep"


def generate_drives(
    parameters: list[Parameter], physics: list[PhysicsRig] | None = None,
) -> list[Animation]:
    """One ``Animation`` per interaction clip whose parameters the character actually has.

    Clips whose parameters are all absent are skipped (a portrait rig gets no ``legs_swing``), so the
    sheet scales down to a bare face and up to a full body without configuration.
    """
    by_id = _drivable(parameters, physics)
    anims: list[Animation] = []
    for name, drive in _DRIVES.items():
        lanes, length = _clip_lanes(drive, by_id)
        if not lanes:
            continue
        anims.append(Animation(name=name, fps=FPS, length=length, loop=False, lanes=lanes))
    return anims


def generate_sweep(
    parameters: list[Parameter], physics: list[PhysicsRig] | None = None,
) -> list[Animation]:
    """A single clip that walks **every** drivable parameter through its full range, in order.

    Each parameter gets the same snap/hold shape as an interaction clip — neutral, max, min, neutral —
    and the next one only starts once the previous is home, so at any instant exactly one parameter is
    moving and whatever you are looking at is attributable to it. This is the clip to scrub through in
    Cubism Viewer when you want to see the entire rig, and the one to leave running when you want a
    physics chain to betray itself.
    """
    by_id = _drivable(parameters, physics)
    if not by_id:
        return []

    span = 3 * _SNAP + 2 * _HOLD          # neutral -> max -> hold -> min -> hold -> neutral
    lanes: list[AnimationLane] = []
    for i, param in enumerate(by_id.values()):
        base = i * span
        frames = [
            (base, param.default),
            (base + _SNAP, _at(param, 1.0)),
            (base + _SNAP + _HOLD, _at(param, 1.0)),
            (base + 2 * _SNAP + _HOLD, _at(param, -1.0)),
            (base + 2 * _SNAP + 2 * _HOLD, _at(param, -1.0)),
            (base + span, param.default),
        ]
        lanes.append(AnimationLane(
            param_id=param.id,
            keyframes=[AnimKeyframe(frame=f, value=v) for f, v in frames],
            interpolation=InterpolateMode.cubic,
        ))
    length = len(by_id) * span + _SETTLE
    return [Animation(name=SWEEP_NAME, fps=FPS, length=length, loop=False, lanes=lanes)]


def _drivable(
    parameters: list[Parameter], physics: list[PhysicsRig] | None,
) -> dict[str, Parameter]:
    """Every parameter a clip is allowed to key, in authored order.

    Excludes **physics outputs**: those are written by the pendulum each frame, so a keyframe on one is
    a lane arguing with the simulation. It also excludes degenerate parameters (``min == max``), which
    cannot be driven anywhere by definition.
    """
    outputs = {r.output_param for r in (physics or [])}
    return {p.id: p for p in parameters if p.id not in outputs and p.max > p.min}


def _clip_lanes(drive: Drive, by_id: dict[str, Parameter]) -> tuple[list[AnimationLane], int]:
    """Lanes for one ping-ponged clip, plus its length. Empty when the character has none of its
    parameters."""
    present = [(by_id[pid], frac) for pid, frac in drive.pose.items() if pid in by_id]
    if not present:
        return [], 0

    # neutral -> +pose -> hold -> -pose -> hold, repeated, then home and settle
    cycle = 2 * (_SNAP + drive.hold)
    length = drive.cycles * cycle + _SNAP + drive.settle

    lanes = []
    for param, frac in present:
        # a bidirectional clip reverses through the opposite pose; a one-directional one returns to
        # neutral instead (0.0 -> param.default), so it never passes through the mirror pose
        back = -frac if drive.bidirectional else 0.0
        frames = [(0, param.default)]
        for c in range(drive.cycles):
            base = c * cycle
            frames += [
                (base + _SNAP, _at(param, frac)),
                (base + _SNAP + drive.hold, _at(param, frac)),
                (base + 2 * _SNAP + drive.hold, _at(param, back)),
                (base + cycle, _at(param, back)),
            ]
        home = drive.cycles * cycle + _SNAP
        frames += [(home, param.default), (length, param.default)]
        lanes.append(AnimationLane(
            param_id=param.id,
            keyframes=[AnimKeyframe(frame=f, value=v) for f, v in frames],
            interpolation=InterpolateMode.cubic,
        ))
    return lanes, length


def _at(param: Parameter, frac: float) -> float:
    """The value ``frac`` of the way from the parameter's default to one of its ends.

    Signed and asymmetric on purpose: parameters are not centred on their default (``ParamEyeLOpen``
    is 0..1 with a default of 1), so ``+1.0`` means "as far as this parameter goes upward" and
    ``-1.0`` "as far as it goes downward", whatever those distances happen to be.
    """
    end = param.max if frac >= 0.0 else param.min
    return param.default + abs(frac) * (end - param.default)
