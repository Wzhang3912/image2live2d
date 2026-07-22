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


def test_elaborate_bilateral_character_is_not_flagged():
    """An ornate character (many heavily-overlapping clothing/hair layers) must NOT be flagged. A
    fill-ratio "cluttered" signal was tried and removed: running seven diverse characters showed
    elaborate legitimate ones (a floor-length gown, huge drill-curls) overlap *more* than the one scene
    it was meant to catch, so the ratio can't discriminate. Bilateral + has-a-face is enough here."""
    full = (0.0, 0.0, 1.0, 1.0)
    parts = [("face_base", R.face_base, full),
             ("eye_l", R.eye_l, full), ("eye_r", R.eye_r, full),
             ("clothing", R.clothing, full), ("accessory", R.accessory, full),
             ("hair_front", R.hair_front, full), ("hair_back", R.hair_back, full)]  # heavy overlap
    assert _codes(_build(parts)) == set()                             # not flagged as a scene


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


# --- mouth-region plausibility (backlog T9 trigger) -----------------------------------------------
# The envelope is measured on the 8 real decomposed characters: mouth width 0.133-0.237 of the face,
# centre 0.808-0.881 of the way down it, horizontally centred to within 0.009 of face width. The bounds
# in `harness` are set several times wider than that spread, so these tests use a REAL mouth as the
# baseline and then break it grossly — the check must not police style, only catch a wrong region.
_FACE = ("face_base", R.face_base, (0.2, 0.1, 0.8, 0.9))
#           face: x 0.2-0.8 (w 0.6), y 0.1-0.9 (h 0.8, y UP so 0.9 is the top of the head)
_REAL_MOUTH = ("mouth", R.mouth, (0.44, 0.21, 0.56, 0.25))
#           width 0.12/0.6 = 0.20 of the face; centre y 0.23 -> (0.9-0.23)/0.8 = 0.84 down it; centred


def test_a_real_mouth_geometry_is_not_flagged():
    """The measured average of the 8 characters must sit comfortably inside the envelope."""
    assert _codes(_build([_FACE, _REAL_MOUTH])) == set()


def test_a_mouth_on_the_forehead_is_flagged():
    high = ("mouth", R.mouth, (0.44, 0.72, 0.56, 0.76))     # up among the eyes, ~0.20 down the face
    assert "misplaced_mouth" in _codes(_build([_FACE, high]))


def test_a_mouth_spanning_the_whole_face_is_flagged():
    wide = ("mouth", R.mouth, (0.21, 0.21, 0.79, 0.25))     # 0.97 of the face's width
    assert "implausible_mouth_width" in _codes(_build([_FACE, wide]))


def test_a_speck_sized_mouth_region_is_flagged():
    tiny = ("mouth", R.mouth, (0.49, 0.21, 0.505, 0.25))    # 0.025 of the face's width
    assert "implausible_mouth_width" in _codes(_build([_FACE, tiny]))


def test_a_mouth_off_to_one_side_is_flagged():
    off = ("mouth", R.mouth, (0.70, 0.21, 0.82, 0.25))      # centre +0.43 of face width off-centre
    assert "off_centre_mouth" in _codes(_build([_FACE, off]))


def test_a_stylised_but_valid_mouth_is_not_flagged():
    """Guard against the check policing style rather than correctness: a wide grin at nearly twice the
    widest measured character, and a small mouth at a third of the narrowest, must both pass."""
    grin = ("mouth", R.mouth, (0.37, 0.21, 0.63, 0.25))     # 0.43 of the face's width
    small = ("mouth", R.mouth, (0.475, 0.22, 0.525, 0.24))  # 0.08 of the face's width
    assert _codes(_build([_FACE, grin])) == set()
    assert _codes(_build([_FACE, small])) == set()


def test_the_check_is_silent_without_a_mouth_or_without_a_face():
    """No mouth layer = no region to judge; no face = `no_face` already covers it. Neither may raise a
    mouth issue, or every faceless/mouthless rig gets a spurious second warning."""
    mouth_codes = _codes(_build([_FACE, ("eye_l", R.eye_l, (0.30, 0.55, 0.45, 0.63))]))
    assert not any("mouth" in c for c in mouth_codes)
    faceless = _codes(_build([("hair_front", R.hair_front, (0.2, 0.6, 0.8, 0.95)), _REAL_MOUTH]))
    assert not any("mouth" in c for c in faceless)


# --- face_base coverage: the reference frame the mouth check measures against ----------------------
# `wikipetan_stand` decomposed with a face_base covering only the forehead and crown (y 0.798-0.935)
# while the nose (0.764-0.777) and mouth (0.723-0.764) sat entirely below it. Eyes above nose above
# mouth: the mouth is anatomically correct, but measured against that truncated frame it reads as 1.40
# of the way down the face and tripped `misplaced_mouth`. That misattribution was the recorded trigger
# for backlog T9 (SAM-mouth-from-source) — a check blaming the mouth for the face's defect would have
# justified a 130MB GPU dependency to re-cut a mouth that was already right.
_FOREHEAD_ONLY = ("face_base", R.face_base, (0.2, 0.62, 0.8, 0.9))   # top third of the face only


def test_a_face_base_that_misses_the_face_is_flagged():
    codes = _codes(_build([_FOREHEAD_ONLY,
                           ("nose", R.nose, (0.47, 0.35, 0.53, 0.42)),
                           _REAL_MOUTH]))
    assert "face_base_incomplete" in codes


def test_an_anatomically_correct_mouth_is_not_blamed_for_a_truncated_face():
    """The regression itself: with the frame broken, the mouth check must stay silent and let
    `face_base_incomplete` report the real defect. Otherwise the pipeline confidently names the wrong
    part, and the fix it argues for would not have fixed anything."""
    codes = _codes(_build([_FOREHEAD_ONLY,
                           ("eye_l", R.eye_l, (0.30, 0.70, 0.45, 0.78)),
                           ("nose", R.nose, (0.47, 0.35, 0.53, 0.42)),
                           _REAL_MOUTH]))
    assert "face_base_incomplete" in codes
    assert not any("mouth" in c for c in codes)


def test_a_face_base_covering_the_whole_face_is_not_flagged():
    """Twelve of thirteen real characters put every feature 1.00 inside face_base (lowest 0.98), so a
    normal face must never trip this."""
    codes = _codes(_build([_FACE,
                           ("eye_l", R.eye_l, (0.30, 0.55, 0.45, 0.63)),
                           ("eye_r", R.eye_r, (0.55, 0.55, 0.70, 0.63)),
                           ("nose", R.nose, (0.47, 0.35, 0.53, 0.42)),
                           _REAL_MOUTH]))
    assert "face_base_incomplete" not in codes


def test_the_coverage_check_is_silent_without_a_face():
    """`no_face` already covers a faceless rig; a second warning would be noise."""
    codes = _codes(_build([("hair_front", R.hair_front, (0.2, 0.6, 0.8, 0.95)), _REAL_MOUTH]))
    assert "face_base_incomplete" not in codes
