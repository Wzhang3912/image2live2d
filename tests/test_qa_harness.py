"""QA pass-rate harness (Phase 2 exit gate)."""

from __future__ import annotations

from pathlib import Path

from image2live2d.core.assemble import assemble_rig
from image2live2d.core.mesh import grid_mesh
from image2live2d.core.physics import generate_physics
from image2live2d.core.qa import batch, evaluate
from image2live2d.core.rig import author_rig, select_template
from image2live2d.core.types import Layer, LayerStack
from image2live2d.irr.schema import SemanticRole as R


def _build(parts):
    layers, meshes = [], []
    for i, (pid, role, rect) in enumerate(parts):
        layers.append(Layer(id=pid, semantic_role=role, texture_path=Path(f"{pid}.png"),
                            draw_order=i * 10, width=64, height=64))
        meshes.append(grid_mesh(pid, rect, lambda u, v: 255, grid=2))
    stack = LayerStack(layers=layers, canvas_width=64, canvas_height=64)
    auth = author_rig(stack, meshes, select_template(stack))
    phys = generate_physics(stack, auth.parameters)
    return assemble_rig(name="t", source=None, stack=stack, meshes=meshes, deformers=auth.deformers,
                        parameters=auth.parameters, physics=phys, archetype="portrait_front")


_GOOD = [("face_base", R.face_base, (0.2, 0.1, 0.8, 0.9)),
         ("eye_l", R.eye_l, (0.30, 0.55, 0.45, 0.63)),
         ("eye_r", R.eye_r, (0.55, 0.55, 0.70, 0.63)),
         ("mouth", R.mouth, (0.42, 0.30, 0.58, 0.38))]


def test_good_rig_passes():
    report = evaluate(_build(_GOOD), "good")
    assert report.passed
    assert not report.reasons


def test_missing_face_roles_fails_on_lint():
    # only a face base -> lint warns missing eyes/mouth + movement params
    report = evaluate(_build([("face_base", R.face_base, (0.2, 0.1, 0.8, 0.9))]), "bare")
    assert not report.passed
    assert any(r.startswith("lint:") for r in report.reasons)


def test_right_eye_via_eye_white_only_passes_lint():
    """A right eye present only as eye_white_r (no eyelash eye_r) must satisfy the gate — the white is
    riggable (blink collapses it). Regression: See-through emits one combined eyelash (-> eye_l only)
    while splitting the whites L/R, which used to fail every such character on missing_role:eye_r."""
    parts = [("face_base", R.face_base, (0.2, 0.1, 0.8, 0.9)),
             ("eye_l", R.eye_l, (0.30, 0.55, 0.45, 0.63)),
             ("eye_white_r", R.eye_white_r, (0.55, 0.55, 0.70, 0.63)),  # right side: white only
             ("mouth", R.mouth, (0.42, 0.30, 0.58, 0.38))]
    report = evaluate(_build(parts), "white_only")
    assert report.passed and not report.reasons


def test_batch_pass_rate_and_format():
    good = _build(_GOOD)
    bare = _build([("face_base", R.face_base, (0.2, 0.1, 0.8, 0.9))])
    report = batch({"good": good, "bare": bare})
    assert report.total == 2
    assert report.passed == 1
    assert report.pass_rate == 0.5
    text = report.format()
    assert "pass-rate: 1/2" in text
    assert "PASS" in text and "FAIL" in text


def test_batch_accepts_pairs():
    report = batch([("a", _build(_GOOD)), ("b", _build(_GOOD))])
    assert report.pass_rate == 1.0


def test_landmark_warnings_fold_into_gate():
    rig = _build(_GOOD)
    ok = evaluate(rig, "x")
    assert ok.passed
    bad = evaluate(rig, "x", landmark_warnings=["pupil_outside_eye:eye_l"])
    assert not bad.passed
    assert "landmark:pupil_outside_eye:eye_l" in bad.reasons
