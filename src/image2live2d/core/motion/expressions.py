"""Stage 5b — Motion. A small **expression sheet** (smile / surprise / sad / angry).

Auto-rigging saves body-rig time; the other big artist cost is *facial* rigging + expressions
(see docs/AUTORIG_PHYSICS_UNIVERSAL_PLAN.md, task P5). This authors a basic, reusable set of named
expression clips as ``Animation``s in the IRR — each a short ease from the neutral pose into a held
emotional pose, driven **entirely by the standard face parameters** (mouth form/open, eye open, brows).
Because they key only standard ids, both backends inherit them for free (the Live2D emitter writes each
as a ``.motion3.json``, nijilive as an animation lane set) and any stock ARKit/motion clip that drives
the same params composes with them.

Like the idle, every lane is present-gated (only authored for parameters the character actually has) and
clamped to each parameter's range, so a bare portrait still gets whatever expressions its params allow
and a pose value never escapes a parameter's bounds. An expression whose parameters are all absent is
skipped; if none apply the sheet is empty.
"""

from __future__ import annotations

from ...irr.schema import AnimKeyframe, Animation, AnimationLane, InterpolateMode, Parameter

FPS = 60.0
_RAMP_FRAMES = 8    # ease from neutral into the pose
_HOLD_FRAMES = 24   # clip length — the pose is reached at _RAMP_FRAMES and held to here

# Each expression is a target pose: {standard param id -> value at the held pose}. Values are the
# *intended* pose; a lane is only authored for params present on the character, and every value is
# clamped to the parameter's own range, so these read as "as expressive as this rig allows".
# Brow convention (ParamBrow*Y): +1 = raised, -1 = lowered/furrowed. Eye open: 1 = wide, 0 = shut.
_EXPRESSIONS: dict[str, dict[str, float]] = {
    # corners up, eyes softened to a happy squint, brows a touch up
    "smile": {"ParamMouthForm": 1.0, "ParamEyeLOpen": 0.6, "ParamEyeROpen": 0.6,
              "ParamBrowLY": 0.3, "ParamBrowRY": 0.3},
    # mouth agape, eyes wide, brows shot up
    "surprise": {"ParamMouthOpenY": 0.7, "ParamEyeLOpen": 1.0, "ParamEyeROpen": 1.0,
                 "ParamBrowLY": 1.0, "ParamBrowRY": 1.0},
    # corners down, brows raised (worried inner-up read), eyes lowered
    "sad": {"ParamMouthForm": -1.0, "ParamEyeLOpen": 0.7, "ParamEyeROpen": 0.7,
            "ParamBrowLY": 0.3, "ParamBrowRY": 0.3},
    # slight frown, brows furrowed down, eyes narrowed — the brow direction is what reads as anger vs sad
    "angry": {"ParamMouthForm": -0.6, "ParamEyeLOpen": 0.9, "ParamEyeROpen": 0.9,
              "ParamBrowLY": -1.0, "ParamBrowRY": -1.0},
}

EXPRESSION_NAMES: tuple[str, ...] = tuple(_EXPRESSIONS)


def generate_expressions(parameters: list[Parameter]) -> list[Animation]:
    """One short (non-looping) ``Animation`` per expression whose parameters the character has.

    Each clip eases from the neutral (default) pose to the target over ``_RAMP_FRAMES`` and holds it,
    so a runtime can trigger it and blend back out. Skips any expression with no present parameters;
    returns ``[]`` if none apply."""
    by_id = {p.id: p for p in parameters}
    anims: list[Animation] = []
    for name, pose in _EXPRESSIONS.items():
        lanes = [_pose_lane(by_id[pid], value) for pid, value in pose.items() if pid in by_id]
        if not lanes:
            continue
        anims.append(Animation(name=name, fps=FPS, length=_HOLD_FRAMES, loop=False, lanes=lanes))
    return anims


def _pose_lane(param: Parameter, value: float) -> AnimationLane:
    """A lane easing ``param`` from its neutral default to ``value`` (clamped to range) and holding."""
    target = _clamp(value, param.min, param.max)
    frames = [(0, param.default), (_RAMP_FRAMES, target), (_HOLD_FRAMES, target)]
    kfs = [AnimKeyframe(frame=f, value=v) for f, v in frames]
    return AnimationLane(param_id=param.id, keyframes=kfs, interpolation=InterpolateMode.cubic)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v
