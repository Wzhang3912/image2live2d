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

**Two motion styles.** The snap → hold → settle shape above is an *impulse* — it is the right tool for
a head clip, where the frozen hold is precisely when you watch the hair lag and catch up. But an arm or
a leg has no hair to trail: the limb itself is the only thing on screen, and a snap-to-pose then a
40-frame freeze reads as a robot arm, not a character. So the body/limb clips use two *baked* styles
instead — a curve sampled densely (every couple of frames) into keyframes, so the motion is smooth under
any interpolation and what the diagnostic renderer draws is what the runtime plays:

* *oscillate* — a continuous eased sine (``body_sway``, ``body_bow``, ``legs_sway``, ``arms_swing``):
  the character sways without ever stopping dead, the peak velocity at each zero-crossing still swings
  the cloth, and an integer number of cycles makes the loop seamless.
* *gesture* — an eased raise with a little overshoot, a *living* hold (a faint breath, not a freeze) and
  a settling follow-through (``arms_raise``, ``legs_swing``): a deliberate one-directional action that
  never passes through the crossing/mirror pose.

Both keep the diagnostic guarantees: they reach the extremes, they carry enough velocity to excite the
pendulums, and the left/right sides still move in opposite phase where that is the point of the clip.

Like the idle and the expression sheet, every lane is present-gated (a bare portrait gets the face
clips and no limb clips) and clamped to its parameter's own range.
"""

from __future__ import annotations

import math
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

# Overlapping action (the Disney principle): a limb should not swing as one rigid board — its LOWER
# segment (knee/elbow) trails the upper (hip/shoulder) and catches up a beat later. Without this a leg
# "just moves to the position" and reads mechanical. For every limb UPPER joint a clip drives, the paired
# LOWER joint follows with a short phase lag — reusing its existing pose value if the clip already drives
# it (so the lower simply trails), or a gentle synthesised bend if it does not (so the limb still hinges).
_LIMB_FOLLOW: dict[str, str] = {
    "ParamLegLA": "ParamLegLB", "ParamLegRA": "ParamLegRB",
    "ParamArmLA": "ParamArmLB", "ParamArmRA": "ParamArmRB",
}
_FOLLOW_LAG = 4       # frames the lower joint trails the upper
_FOLLOW_FRAC = 0.45   # synthesised lower-joint swing as a fraction of the upper's, when not already driven

# Baked-motion styles (body/limb clips). A curve sampled every _STRIDE frames — dense enough to read
# smooth whether the runtime interpolates linearly or by bezier, so the diagnostic GIF (which samples
# straight lines) shows exactly what the .moc3 plays.
_STRIDE = 2

# oscillate: a continuous sine. A quarter-period of 12 frames -> a 48-frame (1.25 Hz) cycle; two cycles
# per clip. Peak velocity at the zero-crossings (~0.13 of range per frame) matches the old snap's impulse,
# so cloth still swings, but nothing ever holds still. Quarter-period is a multiple of _STRIDE so the
# peaks (the extremes of the range) land exactly on sampled frames.
_OSC_QUARTER = 12
_OSC_CYCLES = 2

# gesture: an eased one-directional action. rise (with a touch of overshoot) -> a living hold that breathes
# and settles off the overshoot -> an eased release -> a tail that dips and recovers (follow-through).
_G_RISE = 18
_G_HOLD = 46
_G_FALL = 18
_G_TAIL = 42
_G_OVERSHOOT = 0.07   # how far past the pose the rise carries, as a fraction of the pose
_G_BREATH = 0.025     # amplitude of the living hold's breath
_G_SETTLE_DIP = 0.04  # how far the tail dips back past neutral before recovering


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
    # How the clip is shaped over time. ``"impulse"`` is the snap/hold/settle diagnostic (the default,
    # right for the yaw clip where the frozen hold is when the hair lag is visible). ``"oscillate"`` is a
    # continuous eased sine and ``"gesture"`` is an eased one-directional action — both baked as dense
    # keyframes so a limb (or a nodding/tilting head) reads as alive rather than robotic.
    style: str = "impulse"
    # ``oscillate`` tuning: the quarter-period (frames) and cycle count. The defaults are a gentle ~1.25 Hz
    # sway; a shake overrides them to a brisk, rapid-reversal wobble.
    osc_quarter: int = _OSC_QUARTER
    osc_cycles: int = _OSC_CYCLES


# The sheet. Each clip isolates one axis of the rig so a defect is attributable: if the hair only fails
# on `head_roll`, that is a very different bug from failing on all three.
_DRIVES: dict[str, Drive] = {
    # --- head: the driver of every hair and head-accessory pendulum -------------------------------
    "head_yaw": Drive({"ParamAngleX": 1.0},
                      note="hair sways side to side and lags the turn; earrings swing"),
    "head_pitch": Drive({"ParamAngleY": 1.0}, style="oscillate",
                        note="a nod bobs the hair vertically (the emitter maps pitch to anchor Y)"),
    "head_roll": Drive({"ParamAngleZ": 1.0}, style="oscillate",
                       note="a tilt rolls the hair; the fringe should not slide off the forehead"),
    # The hardest thing you can ask of a hair pendulum: reverse the driver before the previous swing
    # has settled. Under-damped hair whips; over-damped hair reads as a helmet. A brisk continuous sine
    # (short period, several cycles) keeps the reversals rapid without ever snapping rigidly.
    "head_shake": Drive({"ParamAngleX": 0.8}, style="oscillate", osc_quarter=6, osc_cycles=4,
                        note="rapid reversals — hair should lag and trail, not snap rigidly along"),

    # --- body: the driver of the skirt / cloth zones ----------------------------------------------
    "body_sway": Drive({"ParamBodyAngleX": 1.0, "ParamBodyAngleZ": 0.5}, style="oscillate",
                       note="skirt/cloth zones swing and settle; the hem should lag the hips"),
    "body_bow": Drive({"ParamBodyAngleY": 1.0}, style="oscillate",
                      note="lean in/out — cloth should fall, not shear with the torso"),

    # --- limbs: the de-cardboard check ------------------------------------------------------------
    # Both arms used to arrive as a single layer, so they could only ever move as one sheet. These two
    # clips are how you confirm they were separated: if the arms move together here, the split failed.
    # Both arms to +max = both lift outward together (mirror-symmetric convention), a symmetric raise.
    # One-directional: raise and return, not down through the inward/crossed mirror pose.
    "arms_raise": Drive({"ParamArmLA": 1.0, "ParamArmLB": 0.7,
                         "ParamArmRA": 1.0, "ParamArmRB": 0.7},
                        bidirectional=False, style="gesture",
                        note="both arms lift outward together — sleeves must ride their own arm"),
    "arms_swing": Drive({"ParamArmLA": 1.0, "ParamArmRA": -1.0}, style="oscillate",
                        note="opposite phase — proves left and right are separate parts"),
    # Both legs to +max = splay OUTWARD (the limb convention is mirror-symmetric: +param lifts/splays
    # each side away from the midline). Two close-together legs rotated toward each other would cross
    # into an X, so this splays them to a widening stance and returns — never through the crossing pose.
    "legs_swing": Drive({"ParamLegLA": 1.0, "ParamLegLB": 0.6,
                         "ParamLegRA": 1.0, "ParamLegRB": 0.6},
                        bidirectional=False, style="gesture",
                        note="both legs splay outward and back (never through the crossing pose) — "
                             "proves the legs were cut at the crotch seam"),
    # The natural counterpart to the splay diagnostic: a weight-shift. Opposite param signs move the two
    # legs in the *same* screen direction (a lean), so the gap between the feet is preserved and they
    # can never cross — the failure mode `legs_swing` was reshaped to avoid. Small amplitude: this reads
    # as a body-language shift, not a stance.
    "legs_sway": Drive({"ParamLegLA": 0.5, "ParamLegRA": -0.5}, style="oscillate",
                       note="a gentle weight-shift — both legs lean together and the skirt hem follows; "
                            "the feet keep their spacing (parallel motion, never crossing)"),

    # --- face -------------------------------------------------------------------------------------
    "talk": Drive({"ParamMouthOpenY": 1.0}, cycles=3, hold=10, settle=20,
                  note="the mouth must actually open — a cavity, teeth and tongue behind the lips"),
    "smirk": Drive({"ParamMouthForm": 1.0},
                   note="corners up and down without the lips tearing from the face"),
    "smile": Drive({"ParamEyeLSmile": 1.0, "ParamEyeRSmile": 1.0, "ParamMouthForm": 1.0},
                   cycles=2, hold=10, settle=20,
                   note="a happy squint: the eyes close onto an upward '^' arc while the mouth curves up, "
                        "the eye never flattening to a plain blink"),
    "cheek": Drive({"ParamCheek": 1.0}, cycles=2, hold=12, settle=16,
                   note="a synthesised blush fades in on the cheeks (opacity 0 at rest) and back out — "
                        "invisible until driven, so it never alters the resting face"),
    "look": Drive({"ParamEyeBallX": 1.0, "ParamEyeBallY": 1.0},
                  note="pupils travel inside the eye and never cross the lid"),
    "blink": Drive({"ParamEyeLOpen": -1.0, "ParamEyeROpen": -1.0}, cycles=2, hold=6, settle=20,
                   note="the eye squashes shut but never vanishes (a full collapse zeroes its area)"),
    "brows": Drive({"ParamBrowLY": 1.0, "ParamBrowRY": 1.0},
                   note="both brows read *through* the fringe — the right one used to drive nothing"),
    "brow_form": Drive({"ParamBrowLForm": 1.0, "ParamBrowRForm": 1.0},
                       note="both brows TILT (rotate about their centre): +1 furrows the inner ends down "
                            "(angry), -1 lifts them (sad) — a shape change, not a raise, and never into the eye"),
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
    """Lanes for one clip, plus its length. Empty when the character has none of its parameters.

    Dispatches on the clip's ``style``: the default ``"impulse"`` is the snap/hold/settle diagnostic;
    ``"oscillate"`` and ``"gesture"`` are baked (densely sampled) natural motion for body/limb clips."""
    if drive.style in ("oscillate", "gesture"):
        return _baked_lanes(drive, by_id)

    present = [(by_id[pid], frac) for pid, frac in drive.pose.items() if pid in by_id]
    if not present:
        return [], 0

    # neutral -> +pose -> hold -> -pose -> hold, repeated, then home and settle
    cycle = 2 * (_SNAP + drive.hold)
    length = drive.cycles * cycle + _SNAP + drive.settle

    # Overlapping action: the lower limb joint trails the upper (its value the clip's own, else a
    # synthesised gentle bend) — driven below with a lag.
    followers = _followers(drive, by_id)

    def _frames(param: Parameter, frac: float, lag: int) -> list[tuple[int, float]]:
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
        if lag:
            # phase-lag the interior keyframes so the lower joint trails; keep the neutral endpoints
            # pinned at 0 and length so the loop still closes. Dedup by frame (a shift can collide).
            shifted = {0: param.default, length: param.default}
            for f, v in frames:
                if 0 < f < length:
                    shifted[min(f + lag, length - 1)] = v
            frames = sorted(shifted.items())
        return frames

    lanes = []
    for param, frac in present:
        if param.id in followers:
            continue                      # driven below, with a lag, as a trailing lower joint
        frames = _frames(param, frac, 0)
        lanes.append(AnimationLane(
            param_id=param.id,
            keyframes=[AnimKeyframe(frame=f, value=v) for f, v in frames],
            interpolation=InterpolateMode.cubic,
        ))
    for low, frac in followers.items():
        frames = _frames(by_id[low], frac, _FOLLOW_LAG)
        lanes.append(AnimationLane(
            param_id=low,
            keyframes=[AnimKeyframe(frame=f, value=v) for f, v in frames],
            interpolation=InterpolateMode.cubic,
        ))
    return lanes, length


def _followers(drive: Drive, by_id: dict[str, Parameter]) -> dict[str, float]:
    """The lower limb joints that trail their upper joint (overlapping action), mapped to the signed
    fraction to drive them at — the clip's own value if it drives one, else a gentle synthesised bend."""
    followers: dict[str, float] = {}
    for pid, frac in drive.pose.items():
        low = _LIMB_FOLLOW.get(pid)
        if low and low in by_id:
            followers[low] = drive.pose.get(low, frac * _FOLLOW_FRAC)
    return followers


def _smoother(x: float) -> float:
    """Smootherstep on [0, 1]: eases in and out with zero velocity at both ends (Perlin's 6x^5-15x^4+10x^3)."""
    x = min(1.0, max(0.0, x))
    return x * x * x * (x * (6.0 * x - 15.0) + 10.0)


def _sample_frames(length: int) -> list[int]:
    """Every _STRIDE-th frame in [0, length], always including the final frame so the lane closes."""
    frames = list(range(0, length, _STRIDE))
    if frames[-1] != length:
        frames.append(length)
    return frames


def _gesture_env(f: float) -> float:
    """The gesture envelope, peaking at exactly 1.0 (the pose extreme) and never past it: rise to the
    peak, a breathing hold that settles *down* off the peak by the overshoot, an eased release, then a
    tail that dips past neutral and recovers (the follow-through)."""
    if f <= 0.0:
        return 0.0
    rest = 1.0 - _G_OVERSHOOT                                          # the resting hold, below the peak
    if f < _G_RISE:
        return _smoother(f / _G_RISE)                                 # 0 -> 1.0 (the extreme)
    f -= _G_RISE
    if f < _G_HOLD:
        x = f / _G_HOLD
        ramp = _smoother(min(1.0, 2.0 * x))                          # 0 -> 1 over the first half
        settle = _G_OVERSHOOT * (1.0 - ramp)                         # 1.0 -> rest over the first half
        breath = _G_BREATH * math.sin(2.0 * math.pi * x) * ramp      # a faint breath, once settled (never past the peak)
        return rest + settle + breath
    f -= _G_HOLD
    if f < _G_FALL:
        return rest * (1.0 - _smoother(f / _G_FALL))                  # eased release to neutral
    f -= _G_FALL
    return -_G_SETTLE_DIP * math.sin(math.pi * min(1.0, f / _G_TAIL))  # follow-through: dip and recover


def _baked_lanes(drive: Drive, by_id: dict[str, Parameter]) -> tuple[list[AnimationLane], int]:
    """Densely-sampled natural motion: a continuous sine (``oscillate``) or an eased one-way action
    (``gesture``). Sampling every _STRIDE frames makes it smooth under any interpolation, so the diagnostic
    GIF (straight lines between anchors) and the runtime (beziers) draw the same living motion."""
    present = [(by_id[pid], frac) for pid, frac in drive.pose.items() if pid in by_id]
    if not present:
        return [], 0
    followers = _followers(drive, by_id)

    if drive.style == "oscillate":
        period = 4 * drive.osc_quarter
        length = drive.osc_cycles * period
        # peaks (the extremes of the range) land on sampled frames; a lag is a phase shift, so a follower
        # trails the upper joint yet still closes the loop (the sine is periodic over the clip length).

        def drive_at(f: int, lag: int) -> float:
            return math.sin(2.0 * math.pi * (f - lag) / period)
    else:  # gesture
        length = _G_RISE + _G_HOLD + _G_FALL + _G_TAIL

        def drive_at(f: int, lag: int) -> float:
            return _gesture_env(f - lag)

    frames = _sample_frames(length)
    lanes: list[AnimationLane] = []
    for param, frac in present:
        if param.id in followers:
            continue                       # driven below, with a lag, as a trailing lower joint
        lanes.append(_baked_lane(param, frac, frames, drive_at, 0))
    for low, frac in followers.items():
        lanes.append(_baked_lane(by_id[low], frac, frames, drive_at, _FOLLOW_LAG))
    return lanes, length


def _baked_lane(param, frac, frames, drive_at, lag) -> AnimationLane:
    # linear between dense anchors: identical in the diagnostic renderer and in a Cubism runtime, and a
    # steady velocity through each short segment keeps the pendulum excitation honest
    kfs = [AnimKeyframe(frame=f, value=_at(param, frac * drive_at(f, lag))) for f in frames]
    return AnimationLane(param_id=param.id, keyframes=kfs, interpolation=InterpolateMode.linear)


def _at(param: Parameter, frac: float) -> float:
    """The value ``frac`` of the way from the parameter's default to one of its ends.

    Signed and asymmetric on purpose: parameters are not centred on their default (``ParamEyeLOpen``
    is 0..1 with a default of 1), so ``+1.0`` means "as far as this parameter goes upward" and
    ``-1.0`` "as far as it goes downward", whatever those distances happen to be.
    """
    end = param.max if frac >= 0.0 else param.min
    return param.default + abs(frac) * (end - param.default)
