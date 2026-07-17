"""The head turn must be a TURN, not a squash.

The emitter builds its own warp-deformer grid for ParamAngleX/Y — it overrides whatever
``rig.author._head_turn`` bakes for head parts, so asserting on the author here would prove nothing
about the shipped .moc3. That grid used to apply a pure scale about the neck base
(``x' = x·cos(yaw)``, explicitly "NO translation"). A scale about the centre line pulls both eyes
*toward* it, so at a full ±30° yaw the two eyes drifted APART (+0.0082 / -0.0066, measured on the
emitted .moc3 through the native Cubism core) and the face only got narrower: max vertex travel 2.8%
of model width, against 17.1% for roll at the same angle.

The missing piece is depth — a point at depth ``z`` rotating about the head axis moves
``x' = x·cos(a) + z·sin(a)``. These pin the ``z·sin(a)`` term: the whole difference between a face
that turns and a face that squashes.
"""

from __future__ import annotations

import math

import pytest


def _head_warp(tmp_path):
    """Emit the sample rig, capturing the head warp-deformer grid the emitter builds."""
    pytest.importorskip("PIL")
    from image2live2d.backends.live2d import moc3_emit as M
    from image2live2d.core import decompose
    from image2live2d.pipeline import rig_from_stack
    from image2live2d.samples import make_sample_layers

    seen = []
    real = M.EmitWarp

    def spy(*a, **kw):
        w = real(*a, **kw)
        seen.append(w)
        return w

    M.EmitWarp = spy
    try:
        rig = rig_from_stack(decompose.from_layer_dir(make_sample_layers(tmp_path / "src")), name="t")
        M.rig_to_moc3(rig)
    finally:
        M.EmitWarp = real
    if not seen:
        pytest.skip("sample rig emits no head warp")
    return rig, seen[0]


def _grid_at(rig, w, **want):
    """The grid keyform for an exact (yaw, pitch, roll) combination.

    Mirrors the emitter's own index encoding (cartesian product over the turn params, first param
    fastest). Selecting by "whichever grid moved most" instead would silently pick the ROLL keyform —
    roll shares this grid and genuinely rotates, so it masks a yaw that does nothing.
    """
    order = [rig.parameters[i] for i in w.param_indices]
    idx, mul = 0, 1
    for p in order:
        ks = sorted(kf.value for kf in p.keyforms)
        target = want.get(p.id, 0.0)
        ki = min(range(len(ks)), key=lambda k: abs(ks[k] - target))
        idx += ki * mul
        mul *= len(ks)
    return [tuple(pt) for pt in w.keyforms[idx]]


def _yaw_pair(tmp_path):
    """(rest grid, full-yaw grid) with pitch and roll held at zero — yaw isolated."""
    rig, w = _head_warp(tmp_path)
    ids = {rig.parameters[i].id for i in w.param_indices}
    if "ParamAngleX" not in ids:
        pytest.skip("no ParamAngleX on this rig")
    yaw_max = max(kf.value for kf in next(p for p in rig.parameters if p.id == "ParamAngleX").keyforms)
    return _grid_at(rig, w), _grid_at(rig, w, ParamAngleX=yaw_max)


def test_head_core_travels_laterally_at_the_yaw_extreme(tmp_path):
    """Under the old pure-scale grid the head's core mapped to itself (dx == 0 exactly): the head
    could only get narrower, never turn. The face rides in front of the turn axis, so at the yaw
    extreme — with roll and pitch at zero — the grid's core must actually travel."""
    rest, ex = _yaw_pair(tmp_path)
    span = max(p[0] for p in rest) - min(p[0] for p in rest)
    cx = sum(p[0] for p in rest) / len(rest)
    cy = sum(p[1] for p in rest) / len(rest)
    ci = min(range(len(rest)), key=lambda i: math.hypot(rest[i][0] - cx, rest[i][1] - cy))
    travel = abs(ex[ci][0] - rest[ci][0])
    assert travel > 0.05 * span, (
        f"head core travels {travel:.4f} across a {span:.3f}-wide grid — that is a squash, not a "
        "turn (the z*sin(a) sweep term is missing)"
    )


def test_both_sides_of_the_face_sweep_the_same_way(tmp_path):
    """The decisive symptom. A scale about the centre line moves the left and right halves of the
    face in OPPOSITE directions; a real turn sweeps them together."""
    rest, ex = _yaw_pair(tmp_path)
    xs = sorted({round(p[0], 6) for p in rest})
    ys = sorted({round(p[1], 6) for p in rest})
    row = ys[len(ys) // 2]                                    # the face row, through the ball centre
    shifts = []
    for want in (xs[len(xs) // 4], xs[3 * len(xs) // 4]):     # either side of the centre line
        i = min(range(len(rest)), key=lambda k: math.hypot(rest[k][0] - want, rest[k][1] - row))
        shifts.append(ex[i][0] - rest[i][0])
    assert shifts[0] * shifts[1] > 0, (
        f"the two sides of the face move in OPPOSITE directions ({shifts[0]:+.4f} / "
        f"{shifts[1]:+.4f}) — a face getting narrower, not a head turning"
    )


def test_the_neck_junction_stays_put(tmp_path):
    """A turn must not drag the neck: translating the head bodily is what made an earlier attempt
    rubber-stretch it. The sphere handles this for free — the neck junction sits at the ball's bottom
    pole, where depth is ~0, so it gets no sweep."""
    rest, ex = _yaw_pair(tmp_path)
    span = max(p[0] for p in rest) - min(p[0] for p in rest)
    ys = [p[1] for p in rest]
    lo, hi = min(ys), max(ys)
    core = max(abs(e[0] - r[0]) for e, r in zip(ex, rest))
    bottom = max(abs(e[0] - r[0]) for e, r in zip(ex, rest) if abs(r[1] - lo) < 1e-9)
    top = max(abs(e[0] - r[0]) for e, r in zip(ex, rest) if abs(r[1] - hi) < 1e-9)
    assert min(bottom, top) < 0.5 * core, (
        f"the head's neck-side row travels {min(bottom, top):.4f} vs a {core:.4f} core sweep — the "
        "turn is dragging the neck instead of pivoting in it"
    )
    assert core > 0.02 * span
