"""IRR ``Animation`` -> Cubism ``.motion3.json`` (open JSON).

Each animation lane becomes one Cubism ``Curve`` targeting a parameter; keyframes become curve
*segments* (frames converted to seconds via the animation fps). This file is **moc-independent** — it
keys off standard parameter ids, so a generated ``.motion3.json`` drives *any* Live2D model with
those params, even before a ``.moc3`` exists. That makes it the early, immediately-useful win of
Route A.

Segment encoding follows the Cubism format: a curve's ``Segments`` array opens with the initial
point ``[t0, v0]`` then appends one segment per following keyframe — Linear ``[0, t, v]``, Stepped
``[2, t, v]``, or Bezier ``[1, c1t, c1v, c2t, c2v, t, v]``. Our ``Cubic`` lanes are emitted as
flat-tangent ease Beziers (a faithful smooth approximation); ``Nearest`` maps to Stepped.
"""

from __future__ import annotations

from ...irr.schema import Animation, InterpolateMode

MOTION_VERSION = 3

# IRR interpolation -> (segment type id, points-per-segment)
_SEG_LINEAR = 0
_SEG_BEZIER = 1
_SEG_STEPPED = 2


def _seconds(frame: int, fps: float) -> float:
    return round(frame / fps, 6)


def _curve_segments(lane, fps: float) -> tuple[list[float], int, int]:
    """Return (segments, n_segments, n_points) for one lane.

    ``n_points`` counts every (t, v) point including the initial one (Cubism's TotalPointCount)."""
    kfs = sorted(lane.keyframes, key=lambda k: k.frame)
    t0, v0 = _seconds(kfs[0].frame, fps), kfs[0].value
    segments: list[float] = [t0, v0]
    n_points = 1  # the initial point
    n_segments = 0

    prev_t, prev_v = t0, v0
    for kf in kfs[1:]:
        t, v = _seconds(kf.frame, fps), kf.value
        if lane.interpolation in (InterpolateMode.cubic, InterpolateMode.bezier):
            dt = (t - prev_t) / 3.0
            segments += [_SEG_BEZIER, prev_t + dt, prev_v, t - dt, v, t, v]
            n_points += 3
        elif lane.interpolation in (InterpolateMode.stepped, InterpolateMode.nearest):
            segments += [_SEG_STEPPED, t, v]
            n_points += 1
        else:  # Linear
            segments += [_SEG_LINEAR, t, v]
            n_points += 1
        n_segments += 1
        prev_t, prev_v = t, v

    return segments, n_segments, n_points


def motion3(anim: Animation) -> dict:
    """Build the ``.motion3.json`` document for one IRR ``Animation``."""
    curves: list[dict] = []
    total_segments = 0
    total_points = 0
    for lane in anim.lanes:
        segments, n_seg, n_pts = _curve_segments(lane, anim.fps)
        curves.append({"Target": "Parameter", "Id": lane.param_id, "Segments": segments})
        total_segments += n_seg
        total_points += n_pts

    return {
        "Version": MOTION_VERSION,
        "Meta": {
            "Duration": round(anim.length / anim.fps, 6),
            "Fps": anim.fps,
            "Loop": anim.loop,
            "AreBeziersRestricted": True,
            "CurveCount": len(curves),
            "TotalSegmentCount": total_segments,
            "TotalPointCount": total_points,
            "UserDataCount": 0,
            "TotalUserDataSize": 0,
        },
        "Curves": curves,
    }
