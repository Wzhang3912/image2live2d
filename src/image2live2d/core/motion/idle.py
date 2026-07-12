"""Stage 5b — Motion. Procedurally author a looping **idle** animation.

A rigged puppet sitting perfectly still reads as dead. This stage bakes a gentle, looping idle —
periodic blinks, a slow breathing bob, and a subtle body sway — as an ``Animation`` in the IRR so the
character is *alive the moment it loads*, before any external motion clip drives it. Lanes are only
authored for parameters that actually exist (so a bare face still gets blink + breath), and every
keyframe value is clamped to its parameter's range to keep the IRR valid.

Backend-neutral: the nijilive emitter writes these as animation lanes in the ``.inp``; a future
Route-A emitter can write the same data as ``.motion3.json``.
"""

from __future__ import annotations

from ...irr.schema import AnimKeyframe, Animation, AnimationLane, InterpolateMode, Parameter

FPS = 60.0
IDLE_FRAMES = 360  # 6-second loop

_BREATH_AMP = 0.35  # ParamBreath swings 0 -> this -> 0 (gentle; a big bob exposes part seams)
_SWAY_DEG = 0.5     # body sway amplitude (degrees) — very subtle lean so idle reads as calm/connected
_ARM_SWAY = 2.0     # idle shoulder drift (ParamArmLA/RA units) — tiny "settling", proves arms articulate


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

    # --- Body sway: slow left/right lean (prefer body, fall back to head) ------------------------
    sway_id = "ParamBodyAngleX" if "ParamBodyAngleX" in by_id else (
        "ParamAngleX" if "ParamAngleX" in by_id else None
    )
    if sway_id:
        q = IDLE_FRAMES // 4
        lanes.append(_lane(
            by_id[sway_id],
            [(0, 0.0), (q, _SWAY_DEG), (2 * q, 0.0), (3 * q, -_SWAY_DEG), (IDLE_FRAMES, 0.0)],
            InterpolateMode.cubic,
        ))

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


def _lane(param: Parameter, frames: list[tuple[int, float]], interp: InterpolateMode) -> AnimationLane:
    """Build a lane, clamping each keyframe value to the parameter's [min, max]."""
    kfs = [AnimKeyframe(frame=f, value=_clamp(v, param.min, param.max)) for f, v in frames]
    return AnimationLane(param_id=param.id, keyframes=kfs, interpolation=interp)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v
