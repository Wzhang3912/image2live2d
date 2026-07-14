"""Stage 5b — Motion. Procedurally author a looping **idle** animation.

A rigged puppet sitting perfectly still reads as dead. This stage bakes a gentle, looping idle —
periodic blinks, a slow breathing bob, a drifting head and a subtle body sway — as an ``Animation`` in
the IRR so the character is *alive the moment it loads*, before any external motion clip drives it.
Lanes are only authored for parameters that actually exist (so a bare face still gets blink + breath),
and every keyframe value is clamped to its parameter's range to keep the IRR valid.

**The head has to move.** The first idle didn't turn the head at all — and every hair and
head-accessory pendulum in the rig is driven by ``ParamAngleX/Y/Z``, so at rest five of eight physics
chains had a flat-lined driver and the hair hung there like a helmet. Physics is a *driven* system:
nothing swings unless an input moves. The head drift below is what makes the character's own hair move
while it is doing nothing, and it is the reason the idle now looks alive rather than merely blinking.

The drift is a sum of slow sines at incommensurate rates (yaw once per loop, pitch twice, roll once but
out of phase), so it never lands in the same pose twice within the loop and the pendulums are
continuously re-excited rather than settling into a rhythm the eye can predict. The body follows the
head a beat late, the way a body does.

Backend-neutral: the nijilive emitter writes these as animation lanes in the ``.inp``; the Live2D
emitter writes the same data as ``.motion3.json``.
"""

from __future__ import annotations

import math

from ...irr.schema import AnimKeyframe, Animation, AnimationLane, InterpolateMode, Parameter

FPS = 60.0
IDLE_FRAMES = 360  # 6-second loop

_BREATH_AMP = 0.35  # ParamBreath swings 0 -> this -> 0 (gentle; a big bob exposes part seams)
_ARM_SWAY = 2.0     # idle shoulder drift (ParamArmLA/RA units) — tiny "settling", proves arms articulate

# Head drift, in degrees. Calm, but nowhere near zero: this is the *only* thing exciting the hair
# pendulums at rest, and an amplitude that reads as "barely moving" on the head reads as "no hair
# physics at all" three layers down the chain. Yaw carries the most (it is the primary hair driver),
# pitch the least (a nod is a small motion on a still character).
_HEAD_YAW_DEG = 6.0
_HEAD_PITCH_DEG = 3.0
_HEAD_ROLL_DEG = 4.0

# Body sway, in degrees. Drives the skirt/cloth zones — same argument as the head. The old value was
# 0.5 deg on a +/-10 parameter: 5% of range, i.e. a cloth pendulum that never visibly swung.
_BODY_SWAY_DEG = 2.5

# --- The accent, and why the drift alone is not enough -------------------------------------------
# A pendulum responds to ACCELERATION, not position. Drift the head through a slow six-second sine and
# the hair simply hangs straight down from wherever the head currently is: the relative angle the
# physics output actually measures stays near zero the whole time. Simulated against the real Cubism
# physics model, a 12-degree slow drift moves ParamHairFront by 1% of its range — alive on paper,
# invisible on screen.
#
# What excites a pendulum is a *quick* move. So the idle is a slow drift plus a periodic **accent**: a
# brief glance to one side and back, once per loop. This is also just what an idle character does — a
# person at rest drifts and occasionally looks away — so the thing that makes the physics legible is
# the same thing that makes the motion read as alive, which is a nice place to land.
_ACCENT_YAW_DEG = 9.0        # how far the glance goes, on top of the drift
_ACCENT_BODY_DEG = 3.0       # the body follows it, a little
_ACCENT_START = 190          # frame the glance begins
_ACCENT_FRAMES = 48          # 0.8s out and back — fast enough to swing hair, slow enough to look calm

# How often the drift curves are sampled into keyframes. Every 20 frames is plenty for a curve this
# slow once it is cubic-interpolated; the accent is sampled finer (below) because its whole value is in
# how sharply it turns.
_DRIFT_STEP = 20
_ACCENT_STEP = 6


def generate_idle(parameters: list[Parameter]) -> list[Animation]:
    """Author a single looping ``idle`` animation for whichever idle-able params are present.

    Returns ``[]`` if none are present (nothing to animate)."""
    by_id = {p.id: p for p in parameters}
    lanes: list[AnimationLane] = []

    # --- Blink: open most of the time, two quick closes per loop ---------------------------------
    blink_frames = [(0, 1.0)]
    for start in (150, 300):  # two blinks
        blink_frames += [(start, 1.0), (start + 6, 0.0), (start + 12, 1.0)]
    blink_frames.append((IDLE_FRAMES, 1.0))
    for eye in ("ParamEyeLOpen", "ParamEyeROpen"):
        if eye in by_id:
            lanes.append(_lane(by_id[eye], blink_frames, InterpolateMode.linear))

    # --- Breath: smooth bob 0 -> amp -> 0 -------------------------------------------------------
    if "ParamBreath" in by_id:
        lanes.append(_lane(
            by_id["ParamBreath"],
            [(0, 0.0), (IDLE_FRAMES // 2, _BREATH_AMP), (IDLE_FRAMES, 0.0)],
            InterpolateMode.cubic,
        ))

    # --- Head drift: the thing that keeps the hair alive -------------------------------------------
    # Every hair/accessory pendulum is driven by these three. Without them the physics is connected to
    # a constant and the character wears its hair like a helmet. Three sines at different rates so the
    # pose keeps changing (and so the pendulums keep being re-excited) across the whole loop.
    # The yaw carries the accent — it is the primary hair driver, so the glance is what actually swings
    # the hair. Pitch and roll drift only; three simultaneous accents would read as a flinch.
    for pid, amp, cycles, phase, accent in (
        ("ParamAngleX", _HEAD_YAW_DEG, 1.0, 0.0, _ACCENT_YAW_DEG),
        ("ParamAngleY", _HEAD_PITCH_DEG, 2.0, math.pi / 2, 0.0),   # pitch: a small nod, twice per loop
        ("ParamAngleZ", _HEAD_ROLL_DEG, 1.0, math.pi / 3, 0.0),    # roll: trails the yaw; the head arcs
    ):
        if pid in by_id:
            lanes.append(_lane(by_id[pid], _sine(amp, cycles, phase, accent), InterpolateMode.cubic))

    # --- Body sway: the driver of the skirt/cloth zones -------------------------------------------
    # A quarter-loop behind the head, because a body follows the head rather than leading it — and the
    # lag means head and body are never both at rest, so the cloth is excited even when the hair isn't.
    # No head fallback any more: the head has its own drift now, so a rig with no body param loses
    # nothing by leaving this out.
    if "ParamBodyAngleX" in by_id:
        lanes.append(_lane(by_id["ParamBodyAngleX"],
                           _sine(_BODY_SWAY_DEG, 1.0, -math.pi / 2, _ACCENT_BODY_DEG),
                           InterpolateMode.cubic))

    # --- Arm sway: a tiny shoulder drift so articulated arms read as alive, not frozen boards -------
    # Only authored when the limbs were actually separated (ParamArm*A present). Opposite phase L/R, in
    # sync with the breath half-cycle, small amplitude — a subtle "settling" motion, not a wave.
    q = IDLE_FRAMES // 4
    for arm_id, phase in (("ParamArmLA", 1.0), ("ParamArmRA", -1.0)):
        if arm_id in by_id:
            a = _ARM_SWAY * phase
            lanes.append(_lane(
                by_id[arm_id],
                [(0, 0.0), (2 * q, a), (IDLE_FRAMES, 0.0)],
                InterpolateMode.cubic,
            ))

    if not lanes:
        return []
    return [Animation(name="idle", fps=FPS, length=IDLE_FRAMES, loop=True, lanes=lanes)]


def _sine(amp: float, cycles: float, phase: float, accent: float = 0.0) -> list[tuple[int, float]]:
    """Sample ``amp * sin(2pi * cycles * t / loop + phase)``, plus an optional accent, into keyframes.

    ``cycles`` is a whole number of cycles per loop, so the first and last keyframes carry the same
    value and the clip loops seamlessly — a discontinuity at the loop point would read as a twitch, and
    would kick every pendulum hanging off this parameter once every six seconds.

    ``accent`` adds the quick glance described above. It is sampled on a finer grid than the drift,
    because a curve that is only interesting for 0.8 seconds cannot be described by keys 0.33s apart —
    coarse sampling would round the very sharpness that does the exciting straight off it.
    """
    def value(f: int) -> float:
        return amp * math.sin(2.0 * math.pi * cycles * f / IDLE_FRAMES + phase) + _accent(accent, f)

    frames = {f: value(f) for f in range(0, IDLE_FRAMES, _DRIFT_STEP)}
    if accent:
        end = _ACCENT_START + _ACCENT_FRAMES
        frames.update({f: value(f) for f in range(_ACCENT_START, end + 1, _ACCENT_STEP)})
    out = sorted(frames.items())
    out.append((IDLE_FRAMES, value(0)))          # close the loop exactly on the start value
    return out


def _accent(amp: float, frame: int) -> float:
    """A single smooth out-and-back pulse — ``sin^2`` over the accent window, zero everywhere else.

    ``sin^2`` starts and ends at zero *with zero slope*, so the glance blends into the drift instead of
    cornering into it. A corner would be a step in velocity, which is a kick, which is a visible jolt in
    the hair rather than a swing.
    """
    if not amp or not (_ACCENT_START <= frame <= _ACCENT_START + _ACCENT_FRAMES):
        return 0.0
    u = (frame - _ACCENT_START) / _ACCENT_FRAMES
    return amp * math.sin(math.pi * u) ** 2


def _lane(param: Parameter, frames: list[tuple[int, float]], interp: InterpolateMode) -> AnimationLane:
    """Build a lane, clamping each keyframe value to the parameter's [min, max]."""
    kfs = [AnimKeyframe(frame=f, value=_clamp(v, param.min, param.max)) for f, v in frames]
    return AnimationLane(param_id=param.id, keyframes=kfs, interpolation=interp)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v
