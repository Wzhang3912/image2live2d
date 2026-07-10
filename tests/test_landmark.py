"""Phase 3 #13/#15/#18 — silhouette landmark extractor, ML seams, QA, overlay."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from image2live2d.core import decompose, landmark
from image2live2d.core.landmark import (
    EyeLandmarks,
    Landmarks,
    MouthLandmarks,
    Oval,
    analyze_silhouette,
    landmarks_from_silhouettes,
)
from image2live2d.irr.schema import SemanticRole as R


# --------------------------------------------------------------------------------------------------
# Pure core: analyze_silhouette
# --------------------------------------------------------------------------------------------------
def _disc(cx: float, cy: float, rx: float, ry: float):
    """Alpha sampler (u right, v down) for an ellipse in model space (y up)."""
    def alpha_at(u: float, v: float) -> int:
        mx, my = u, 1.0 - v  # rect is the unit square, y up
        return 255 if ((mx - cx) / rx) ** 2 + ((my - cy) / ry) ** 2 <= 1.0 else 0
    return alpha_at


def test_analyze_silhouette_centroid_and_bbox():
    sil = analyze_silhouette((0.0, 0.0, 1.0, 1.0), _disc(0.5, 0.5, 0.3, 0.2), samples=200)
    assert sil is not None
    assert sil.centroid[0] == pytest.approx(0.5, abs=0.01)
    assert sil.centroid[1] == pytest.approx(0.5, abs=0.01)
    x0, y0, x1, y1 = sil.bbox
    assert x0 == pytest.approx(0.2, abs=0.02) and x1 == pytest.approx(0.8, abs=0.02)
    assert y0 == pytest.approx(0.3, abs=0.02) and y1 == pytest.approx(0.7, abs=0.02)
    # coverage ~ area of ellipse / unit square
    assert sil.coverage == pytest.approx(math.pi * 0.3 * 0.2, abs=0.02)


def test_analyze_silhouette_extremes_y_up():
    sil = analyze_silhouette((0.0, 0.0, 1.0, 1.0), _disc(0.5, 0.6, 0.2, 0.25), samples=200)
    assert sil.topmost[1] > sil.bottommost[1]  # y up: top has larger y
    assert sil.leftmost[0] < sil.rightmost[0]
    assert sil.topmost[1] == pytest.approx(0.85, abs=0.02)


def test_analyze_silhouette_empty_returns_none():
    assert analyze_silhouette((0.0, 0.0, 1.0, 1.0), lambda u, v: 0, samples=16) is None


def test_analyze_silhouette_principal_axis_of_horizontal_bar():
    # a wide, short bar -> principal axis ~horizontal (angle near 0)
    sil = analyze_silhouette((0.0, 0.0, 1.0, 1.0), _disc(0.5, 0.5, 0.4, 0.05), samples=200)
    assert abs(math.sin(sil.angle)) < 0.2


# --------------------------------------------------------------------------------------------------
# Pure core: landmarks_from_silhouettes
# --------------------------------------------------------------------------------------------------
def test_landmarks_assembly_from_silhouettes():
    sils = {
        R.face_base: analyze_silhouette((0, 0, 1, 1), _disc(0.5, 0.55, 0.35, 0.4), samples=120),
        R.eye_l: analyze_silhouette((0, 0, 1, 1), _disc(0.4, 0.6, 0.06, 0.03), samples=120),
        R.pupil_l: analyze_silhouette((0, 0, 1, 1), _disc(0.4, 0.6, 0.02, 0.02), samples=120),
        R.mouth: analyze_silhouette((0, 0, 1, 1), _disc(0.5, 0.4, 0.08, 0.03), samples=120),
        R.arm_l: analyze_silhouette((0, 0, 1, 1), _disc(0.25, 0.4, 0.05, 0.15), samples=120),
    }
    lm = landmarks_from_silhouettes(sils)
    assert lm.face_oval and lm.face_oval.radius_x == pytest.approx(0.35, abs=0.03)
    assert lm.eye_l and lm.eye_l.pupil is not None
    assert lm.eye_l.lid_top[1] > lm.eye_l.lid_bottom[1]
    assert lm.mouth and lm.mouth.left_corner[0] < lm.mouth.right_corner[0]
    # arm joint = top-center of the arm silhouette
    assert "arm_l" in lm.joints
    jx, jy = lm.joints["arm_l"]
    assert jx == pytest.approx(0.25, abs=0.03)
    assert jy == pytest.approx(0.55, abs=0.03)  # bbox top of a disc centered 0.4 r 0.15


def test_single_pupil_attaches_to_its_eye_no_outside_warning():
    """A single pupil (See-through often emits one, mapped to pupil_l) that physically sits in the
    RIGHT eye must attach to eye_r, leave eye_l pupil-less, and raise no pupil_outside_eye warning.
    Regression for the edge-case finding where raw _l/_r labelling flagged every character."""
    sils = {
        R.eye_white_l: analyze_silhouette((0, 0, 1, 1), _disc(0.40, 0.60, 0.04, 0.03), samples=120),
        R.eye_white_r: analyze_silhouette((0, 0, 1, 1), _disc(0.60, 0.60, 0.04, 0.03), samples=120),
        R.pupil_l: analyze_silhouette((0, 0, 1, 1), _disc(0.60, 0.60, 0.015, 0.015), samples=120),
    }
    lm = landmarks_from_silhouettes(sils)
    assert lm.eye_r and lm.eye_r.pupil is not None      # the lone pupil went to the eye it's in
    assert lm.eye_l and lm.eye_l.pupil is None           # the other eye is left pupil-less
    assert landmark.landmark_warnings(lm) == []          # no false pupil_outside_eye


def test_eye_white_fallback_when_no_lid_part():
    sils = {R.eye_white_r: analyze_silhouette((0, 0, 1, 1), _disc(0.6, 0.6, 0.05, 0.03), samples=120)}
    lm = landmarks_from_silhouettes(sils)
    assert lm.eye_r is not None and lm.eye_r.pupil is None


# --------------------------------------------------------------------------------------------------
# Pillow wrapper end-to-end on the built-in sample
# --------------------------------------------------------------------------------------------------
def test_extract_landmarks_from_sample(tmp_path):
    pytest.importorskip("PIL")
    from image2live2d.samples import make_sample_layers

    layer_dir = make_sample_layers(tmp_path / "s")
    stack = decompose.from_layer_dir(layer_dir)
    lm = landmark.extract_landmarks(stack)

    assert not lm.is_empty()
    assert lm.face_oval is not None
    assert lm.eye_l and lm.eye_r and lm.mouth
    # pupils localized inside their eyes
    for eye in (lm.eye_l, lm.eye_r):
        if eye.pupil:
            xlo, xhi = sorted((eye.inner[0], eye.outer[0]))
            assert xlo <= eye.pupil[0] <= xhi
    assert landmark.landmark_warnings(lm) == []


def test_extract_landmarks_resolves_small_parts_on_large_canvas(tmp_path):
    """A tiny facial part on a big canvas must still yield a non-degenerate landmark.

    Regression: the extractor sampled a fixed 64x64 grid over the whole [0,1] square, so a ~26px eye
    on a 1024px canvas caught 0-2 probes and collapsed to a point (width/height 0 -> dead pupil-look
    and mouth-open). Sampling within each part's alpha bbox fixes it. The OLD code returned width 0
    for these parts; the new code must report real extents."""
    pytest.importorskip("PIL")
    from PIL import Image, ImageDraw

    C = 1024
    d = tmp_path / "small"
    d.mkdir()

    def put(order, role, box):  # box in pixels on the big canvas
        im = Image.new("RGBA", (C, C), (0, 0, 0, 0))
        ImageDraw.Draw(im).ellipse(box, fill=(30, 30, 40, 255))
        im.save(d / f"{order:02d}_{role}.png")

    put(0, "face_base", (380, 360, 644, 660))     # ~26% of canvas
    put(1, "eye_white_l", (452, 470, 478, 486))   # ~26x16 px eye
    put(2, "eye_white_r", (546, 470, 572, 486))
    put(3, "mouth", (495, 560, 529, 576))         # ~34x16 px mouth

    stack = decompose.from_layer_dir(d)
    lm = landmark.extract_landmarks(stack)
    assert lm.eye_l and lm.eye_l.width > 0 and lm.eye_l.height > 0
    assert lm.eye_r and lm.eye_r.width > 0
    assert lm.mouth and lm.mouth.width > 0 and lm.mouth.height > 0


def test_render_overlay_writes_png(tmp_path):
    pytest.importorskip("PIL")
    from image2live2d.samples import make_sample_layers

    layer_dir = make_sample_layers(tmp_path / "s")
    stack = decompose.from_layer_dir(layer_dir)
    lm = landmark.extract_landmarks(stack)
    out = landmark.render_overlay(stack, lm, tmp_path / "overlay.png")
    assert out.exists() and out.stat().st_size > 0


# --------------------------------------------------------------------------------------------------
# QA warnings
# --------------------------------------------------------------------------------------------------
def test_landmark_warnings_flag_bad_geometry():
    lm = Landmarks(
        face_oval=Oval((0.5, 0.5), 0.0, 0.3),  # degenerate radius_x
        eye_l=EyeLandmarks(
            center=(0.4, 0.6), lid_top=(0.4, 0.63), lid_bottom=(0.4, 0.57),
            inner=(0.36, 0.6), outer=(0.44, 0.6), pupil=(0.9, 0.6),  # pupil far outside
        ),
        mouth=MouthLandmarks(
            center=(0.5, 0.4), left_corner=(0.6, 0.4), right_corner=(0.4, 0.4),  # swapped
            top=(0.5, 0.38), bottom=(0.5, 0.42),  # inverted (top below bottom)
        ),
    )
    codes = set(landmark.landmark_warnings(lm))
    assert "degenerate_face_oval" in codes
    assert "pupil_outside_eye:eye_l" in codes
    assert "mouth_corners_swapped" in codes
    assert "mouth_inverted" in codes


# --------------------------------------------------------------------------------------------------
# Gated ML seams
# --------------------------------------------------------------------------------------------------
def test_ml_seams_are_gated():
    with pytest.raises(NotImplementedError):
        landmark.detect_face_landmarks_ml(Path("x.png"))
    with pytest.raises(NotImplementedError):
        landmark.detect_pose_ml(Path("x.png"))
