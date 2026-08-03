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
# Eyes stay on the blink axis OPEN for every held expression except surprise (which widens them):
# driving ParamEyeOpen toward closed for smile/sad/angry reads as a *blink*, not emotion, because
# ParamEyeOpen is the blink axis — lowering it just squashes the open eye toward the sleepy lash line,
# and since the clip eases in (and a showcase loops it) the eyes visibly open→narrow→open, i.e. a blink.
# Smile is now the exception the old note anticipated: it layers a genuine happy squint via the eye-FORM
# axis ParamEyeSmile (the eye closes onto an upward "^" arc, not the flat blink line), so its eyes read
# as delighted rather than shut. Sad/angry still come from MOUTH + BROWS only — there is no sad-/angry-eye
# form to layer, and the blink axis would misread there for the reason above.
_EXPRESSIONS: dict[str, dict[str, float]] = {
    # corners up, brows a touch up, a genuine happy squint (eye-form, not the blink axis)
    "smile": {"ParamMouthForm": 1.0, "ParamBrowLY": 0.4, "ParamBrowRY": 0.4,
              "ParamEyeLSmile": 0.7, "ParamEyeRSmile": 0.7},
    # mouth agape, eyes wide, brows shot up
    "surprise": {"ParamMouthOpenY": 0.7, "ParamEyeLOpen": 1.0, "ParamEyeROpen": 1.0,
                 "ParamBrowLY": 1.0, "ParamBrowRY": 1.0},
    # corners down, brows raised AND tilted inner-up (the worried "/  \" that reads as sad)
    "sad": {"ParamMouthForm": -1.0, "ParamBrowLY": 0.5, "ParamBrowRY": 0.5,
            "ParamBrowLForm": -1.0, "ParamBrowRForm": -1.0},
    # frown + brows lowered AND furrowed inner-down (the "\  /" that reads as anger, not just low brows)
    "angry": {"ParamMouthForm": -0.7, "ParamBrowLY": -1.0, "ParamBrowRY": -1.0,
              "ParamBrowLForm": 1.0, "ParamBrowRForm": 1.0},
    # bashful: a full cheek blush, a soft smile, eyes cast down and slightly away — the blush carries it
    "shy": {"ParamCheek": 1.0, "ParamMouthForm": 0.35, "ParamEyeBallY": -0.4, "ParamEyeBallX": 0.3},
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
