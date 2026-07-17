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


# --- input plausibility gate (character vs profile/scene) ------------------------------------------
from image2live2d.core.qa.harness import plausibility_issues  # noqa: E402


def _codes(rig):
    return {i.code for i in plausibility_issues(rig)}


def test_bilateral_character_is_plausible():
    """A normal front-facing face (both eyes) raises no input warning."""
    assert _codes(_build(_GOOD)) == set()


def test_one_sided_face_is_flagged():
    """A face whose features are ALL on one side (nothing on the other) is a profile view or a
    mis-decomposition — the front-facing rig can't drive both sides. This is the shape a decomposed
    scene (a girl-at-a-bar-table) came through as."""
    parts = [("face_base", R.face_base, (0.2, 0.1, 0.8, 0.9)),
             ("eye_l", R.eye_l, (0.30, 0.55, 0.45, 0.63)),
             ("eye_white_l", R.eye_white_l, (0.30, 0.55, 0.45, 0.63)),
             ("eyebrow_l", R.eyebrow_l, (0.30, 0.64, 0.45, 0.68))]     # 3 left features, 0 right
    assert "one_sided_face" in _codes(_build(parts))


def test_combined_eyelash_with_split_whites_is_still_bilateral():
    """Not every one-sided PAIR means a one-sided face: See-through emits a single combined eyelash
    (one eye_l) while splitting the whites L/R, so the face has features on both sides and must NOT be
    flagged — judged by side, not pair-by-pair."""
    parts = [("face_base", R.face_base, (0.2, 0.1, 0.8, 0.9)),
             ("eye_l", R.eye_l, (0.30, 0.55, 0.45, 0.63)),             # combined eyelash -> left only
             ("eye_white_l", R.eye_white_l, (0.30, 0.55, 0.45, 0.63)),
             ("eye_white_r", R.eye_white_r, (0.55, 0.55, 0.70, 0.63)),  # right side present via white
             ("mouth", R.mouth, (0.42, 0.30, 0.58, 0.38))]
    assert "one_sided_face" not in _codes(_build(parts))


def test_no_face_is_flagged():
    """Nothing face-like among the parts -> the head-turn/blink/gaze rig has nothing to attach to."""
    parts = [("hair_front", R.hair_front, (0.2, 0.6, 0.8, 0.95)),
             ("clothing", R.clothing, (0.3, 0.1, 0.7, 0.55))]
    assert "no_face" in _codes(_build(parts))


def test_cluttered_input_is_flagged():
    """Parts overlapping far more than a single figure (many full-canvas layers stacked) reads as a
    scene / multiple subjects, not one character."""
    full = (0.0, 0.0, 1.0, 1.0)
    parts = [("face_base", R.face_base, full),
             ("eye_l", R.eye_l, full), ("eye_r", R.eye_r, full),
             ("clothing", R.clothing, full), ("accessory", R.accessory, full),
             ("hair_front", R.hair_front, full)]                        # 6 full-canvas parts -> fill ~6x
    codes = _codes(_build(parts))
    assert "cluttered_input" in codes
    assert "one_sided_face" not in codes                               # bilateral -> only clutter fires


def test_flagged_input_still_produces_a_rig():
    """Graceful degradation: the gate WARNS, it does not block. A flagged input still emits a full rig;
    QA just reports passed=False with the reason, so the caller (and the web UI) can decide."""
    parts = [("face_base", R.face_base, (0.2, 0.1, 0.8, 0.9)),
             ("eye_l", R.eye_l, (0.30, 0.55, 0.45, 0.63)),
             ("eye_white_l", R.eye_white_l, (0.30, 0.55, 0.45, 0.63)),
             ("eyebrow_l", R.eyebrow_l, (0.30, 0.64, 0.45, 0.68))]
    rig = _build(parts)
    assert len(rig.parts) == 4                                          # rig still built, not blocked
    report = evaluate(rig, "one_sided")
    assert not report.passed
    assert any(r.startswith("input:") for r in report.reasons)
