"""Stage 4a — Landmarks. Per-character geometry derived from part silhouettes.

This is Phase 3's quality jump. See-through already separates each part into its own mask, so we can
derive precise landmarks — face oval, eye lid axis, mouth corners, pupil centroid, limb joints —
from each part's **alpha silhouette**, with *zero new ML*. The landmark-corrected solver
(``core.rig.author_rig``) then fits deformations to *this* character instead of a generic bounding
box (head turn pivots on the real face oval; blink collapses along the true lid line; mouth keys off
real corners; pupils stay in the eye).

Two ML seams are provided as optional upgrades for stylized art the silhouette path can't resolve:
``detect_face_landmarks_ml`` and ``detect_pose_ml`` (gated; raise ``NotImplementedError`` until a
GPU model is wired). Both return the same dataclasses, so the solver stays backend-agnostic.

Design mirrors the rest of the pipeline:
* A **pure core** (``analyze_silhouette``, ``landmarks_from_silhouettes``) takes an alpha *sampler*
  and is fully testable without Pillow or any ML extras.
* A thin **Pillow wrapper** (``extract_landmarks``) reads each layer's PNG alpha.

Coordinate space: model space, **y up**, normalized to the canvas ([0, 1]) — identical to mesh
vertices, so landmarks compose directly with keyform offsets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..types import LayerStack
from ...irr.schema import SemanticRole, Vec2

# (u, v) in [0, 1], v down -> alpha 0..255 (same convention as core.mesh).
AlphaSampler = Callable[[float, float], int]

DEFAULT_SAMPLES = 64  # NxN probe grid over a layer when measuring its silhouette
DEFAULT_ALPHA_THRESHOLD = 8  # below this a texel counts as transparent

# Roles whose silhouette we measure. Everything else (hair, clothing, accessories) is irrelevant to
# landmark fitting and skipped for speed.
_LANDMARK_ROLES = {
    SemanticRole.face_base,
    SemanticRole.eye_l, SemanticRole.eye_r,
    SemanticRole.eye_white_l, SemanticRole.eye_white_r,
    SemanticRole.pupil_l, SemanticRole.pupil_r,
    SemanticRole.mouth,
    SemanticRole.eyebrow_l, SemanticRole.eyebrow_r,
    SemanticRole.arm_l, SemanticRole.arm_r,
    SemanticRole.leg_l, SemanticRole.leg_r,
}


# --------------------------------------------------------------------------------------------------
# Silhouette analysis (pure core)
# --------------------------------------------------------------------------------------------------
@dataclass
class Silhouette:
    """Measured geometry of one part's alpha mask, in model space (y up).

    ``angle`` is the principal-axis orientation (radians, 0 = horizontal) from the covariance of the
    covered texels — useful for limb direction. ``coverage`` is the covered fraction of the probe
    grid (a sanity signal: ~0 means an empty/missing layer).
    """

    centroid: Vec2
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1), y up
    topmost: Vec2
    bottommost: Vec2
    leftmost: Vec2
    rightmost: Vec2
    angle: float
    coverage: float


def analyze_silhouette(
    rect: tuple[float, float, float, float],
    alpha_at: AlphaSampler,
    *,
    samples: int = DEFAULT_SAMPLES,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
) -> Silhouette | None:
    """Measure the alpha silhouette inside a model-space ``rect`` via an alpha sampler.

    ``rect`` is ``(x0, y0, x1, y1)`` in model space (y up); ``alpha_at(u, v)`` samples the layer with
    ``u`` left->right and ``v`` top->bottom over [0, 1]. Returns ``None`` if no texel clears
    ``alpha_threshold`` (a fully transparent / missing layer).
    """
    if samples < 2:
        raise ValueError(f"samples must be >= 2, got {samples}")
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0

    n = 0
    sx = sy = sxx = syy = sxy = 0.0
    bx0 = by0 = math.inf
    bx1 = by1 = -math.inf
    topmost = bottommost = leftmost = rightmost = (0.0, 0.0)

    for j in range(samples):
        v = (j + 0.5) / samples
        my = y1 - v * h  # v down -> y up
        for i in range(samples):
            u = (i + 0.5) / samples
            if alpha_at(u, v) < alpha_threshold:
                continue
            mx = x0 + u * w
            n += 1
            sx += mx
            sy += my
            sxx += mx * mx
            syy += my * my
            sxy += mx * my
            if mx < bx0:
                bx0, leftmost = mx, (mx, my)
            if mx > bx1:
                bx1, rightmost = mx, (mx, my)
            if my < by0:
                by0, bottommost = my, (mx, my)
            if my > by1:
                by1, topmost = my, (mx, my)

    if n == 0:
        return None

    cx, cy = sx / n, sy / n
    # Covariance of covered texels -> principal-axis angle.
    cov_xx = sxx / n - cx * cx
    cov_yy = syy / n - cy * cy
    cov_xy = sxy / n - cx * cy
    angle = 0.5 * math.atan2(2.0 * cov_xy, cov_xx - cov_yy)
    return Silhouette(
        centroid=(cx, cy),
        bbox=(bx0, by0, bx1, by1),
        topmost=topmost,
        bottommost=bottommost,
        leftmost=leftmost,
        rightmost=rightmost,
        angle=angle,
        coverage=n / (samples * samples),
    )


# --------------------------------------------------------------------------------------------------
# Landmark schema
# --------------------------------------------------------------------------------------------------
@dataclass
class Oval:
    """An ellipse fitted to a part (the face). Center + axis radii in model space."""

    center: Vec2
    radius_x: float
    radius_y: float


@dataclass
class EyeLandmarks:
    """One eye. ``inner``/``outer`` are the horizontal corners; ``lid_top``/``lid_bottom`` the
    vertical extent of the lid; ``pupil`` the pupil centroid if a pupil part exists."""

    center: Vec2
    lid_top: Vec2
    lid_bottom: Vec2
    inner: Vec2
    outer: Vec2
    pupil: Vec2 | None = None

    @property
    def width(self) -> float:
        return abs(self.outer[0] - self.inner[0])

    @property
    def height(self) -> float:
        return abs(self.lid_top[1] - self.lid_bottom[1])


@dataclass
class MouthLandmarks:
    center: Vec2
    left_corner: Vec2
    right_corner: Vec2
    top: Vec2
    bottom: Vec2

    @property
    def width(self) -> float:
        return abs(self.right_corner[0] - self.left_corner[0])

    @property
    def height(self) -> float:
        return abs(self.top[1] - self.bottom[1])


@dataclass
class BrowLandmarks:
    center: Vec2
    inner: Vec2
    outer: Vec2


@dataclass
class Landmarks:
    """The full per-character landmark set. Any field may be ``None`` when its part is absent; the
    solver feature-gates on each independently and falls back to bbox heuristics."""

    face_oval: Oval | None = None
    eye_l: EyeLandmarks | None = None
    eye_r: EyeLandmarks | None = None
    mouth: MouthLandmarks | None = None
    brow_l: BrowLandmarks | None = None
    brow_r: BrowLandmarks | None = None
    # Limb attachment points keyed by SemanticRole value ("arm_l", "leg_r", ...).
    joints: dict[str, Vec2] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not any(
            (self.face_oval, self.eye_l, self.eye_r, self.mouth, self.brow_l, self.brow_r,
             self.joints)
        )


# --------------------------------------------------------------------------------------------------
# Assembly: silhouettes -> landmarks (pure core)
# --------------------------------------------------------------------------------------------------
def _eye(eye: Silhouette, pupil: Silhouette | None) -> EyeLandmarks:
    return EyeLandmarks(
        center=eye.centroid,
        lid_top=eye.topmost,
        lid_bottom=eye.bottommost,
        inner=eye.leftmost,
        outer=eye.rightmost,
        pupil=pupil.centroid if pupil else None,
    )


def landmarks_from_silhouettes(sils: dict[SemanticRole, Silhouette]) -> Landmarks:
    """Derive the landmark set from per-role silhouettes. Pure: no IO, no Pillow."""
    R = SemanticRole
    lm = Landmarks()

    face = sils.get(R.face_base)
    if face:
        x0, y0, x1, y1 = face.bbox
        lm.face_oval = Oval(face.centroid, (x1 - x0) / 2.0, (y1 - y0) / 2.0)

    # Eyes: use the eye-white (the actual eye opening) for lid/center geometry when available, since
    # the pupil sits inside it; fall back to the eyeline/lash part. A pupil is attached to an eye only
    # if its centroid actually falls within that eye's horizontal span — See-through often emits a
    # single pupil layer (mapped to pupil_l) that physically sits in the *other* eye, which by raw
    # _l/_r labelling would land it outside its eye (the pupil_outside_eye warning). Position-matching
    # puts it in the eye it belongs to and leaves the genuinely pupil-less eye with none.
    pupils = [s for s in (sils.get(R.pupil_l), sils.get(R.pupil_r)) if s]

    def _pupil_in(eye: Silhouette) -> Silhouette | None:
        # Attach a pupil only if its centroid is within the eye's exact x AND y span — the same bounds
        # landmark_warnings checks — so an assigned pupil is never then flagged as outside. A pupil
        # beyond it (wrong eye, or a vertically-thin/degenerate See-through eye-white) goes to the
        # other eye or nowhere, leaving that eye pupil-less (the eyeball param falls back to bbox).
        xlo, xhi = sorted((eye.leftmost[0], eye.rightmost[0]))
        ylo, yhi = sorted((eye.topmost[1], eye.bottommost[1]))
        inside = [p for p in pupils
                  if xlo <= p.centroid[0] <= xhi and ylo <= p.centroid[1] <= yhi]
        return min(inside, key=lambda p: abs(p.centroid[0] - eye.centroid[0]), default=None)

    left = sils.get(R.eye_white_l) or sils.get(R.eye_l)
    if left:
        lm.eye_l = _eye(left, _pupil_in(left))
    right = sils.get(R.eye_white_r) or sils.get(R.eye_r)
    if right:
        lm.eye_r = _eye(right, _pupil_in(right))

    mouth = sils.get(R.mouth)
    if mouth:
        lm.mouth = MouthLandmarks(
            center=mouth.centroid,
            left_corner=mouth.leftmost,
            right_corner=mouth.rightmost,
            top=mouth.topmost,
            bottom=mouth.bottommost,
        )

    for role, attr in ((R.eyebrow_l, "brow_l"), (R.eyebrow_r, "brow_r")):
        brow = sils.get(role)
        if brow:
            setattr(lm, attr, BrowLandmarks(brow.centroid, brow.leftmost, brow.rightmost))

    # Limb joints: a hanging limb attaches at the top-center of its silhouette (shoulder / hip). For
    # elbow/knee bend we also record the mid joint (elbow / knee) and the end (wrist / ankle) along the
    # limb, so the rig can swing the whole limb about the shoulder/hip AND bend the lower segment about
    # the elbow/knee. Keys: "arm_l" (shoulder), "arm_l_mid" (elbow), "arm_l_end" (wrist); legs likewise.
    for role in (R.arm_l, R.arm_r, R.leg_l, R.leg_r):
        sil = sils.get(role)
        if sil:
            cx = sil.centroid[0]
            _, bot_y, _, top_y = sil.bbox                    # y-up: bbox = (x0, y0, x1, y1)
            lm.joints[role.value] = (cx, top_y)              # shoulder / hip (whole-limb swing pivot)
            lm.joints[f"{role.value}_mid"] = (cx, top_y - 0.5 * (top_y - bot_y))  # elbow / knee
            lm.joints[f"{role.value}_end"] = (cx, bot_y)     # wrist / ankle

    return lm


# --------------------------------------------------------------------------------------------------
# Pillow wrapper: LayerStack -> Landmarks
# --------------------------------------------------------------------------------------------------
def extract_landmarks(
    stack: LayerStack,
    *,
    samples: int = DEFAULT_SAMPLES,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
) -> Landmarks:
    """Measure silhouette landmarks for every landmark-relevant layer in ``stack`` (needs Pillow).

    Each layer is full-canvas, but its silhouette is measured over the part's **own alpha bbox**, not
    the whole [0, 1] square. A fixed ``samples`` x ``samples`` grid spread over the full 1280px canvas
    lands only 0-2 probes inside a ~26px eye, collapsing the silhouette to a point (degenerate eye/
    mouth landmarks -> dead pupil-look / mouth-open). Restricting the grid to the part's bbox (as
    ``build_mesh`` does) gives dense, accurate coverage; results map back to whole-canvas model
    coordinates so they still align with mesh vertices. Non-landmark roles are skipped.
    """
    from PIL import Image  # local import: keep the core contract importable without Pillow

    from ..mesh import alpha_bbox  # same alpha-bbox tightening the mesh builder uses

    sils: dict[SemanticRole, Silhouette] = {}
    for layer in stack.layers:
        if layer.semantic_role not in _LANDMARK_ROLES:
            continue
        with Image.open(layer.texture_path) as img:
            rgba = img.convert("RGBA")
            w, h = rgba.size
            alpha_px = rgba.getchannel("A").load()

        box = alpha_bbox(lambda px, py, _ap=alpha_px: _ap[px, py], w, h, alpha_threshold)
        if box is None:
            continue
        px0, py0, px1, py1 = box
        u0, v0, u1, v1 = px0 / w, py0 / h, (px1 + 1) / w, (py1 + 1) / h  # uv rect (v down)
        rect = (u0, 1.0 - v1, u1, 1.0 - v0)                              # model rect (y up)

        def alpha_at(u: float, v: float, _ap=alpha_px, _w=w, _h=h,
                     _u0=u0, _v0=v0, _du=u1 - u0, _dv=v1 - v0) -> int:
            px = min(_w - 1, max(0, int((_u0 + u * _du) * _w)))
            py = min(_h - 1, max(0, int((_v0 + v * _dv) * _h)))
            return _ap[px, py]

        sil = analyze_silhouette(rect, alpha_at, samples=samples, alpha_threshold=alpha_threshold)
        if sil is not None:
            sils[layer.semantic_role] = sil  # last layer of a role wins (roles rarely duplicate)

    return landmarks_from_silhouettes(sils)


# --------------------------------------------------------------------------------------------------
# ML seams (gated upgrades — Phase 3 #15 / #17)
# --------------------------------------------------------------------------------------------------
# A detector takes a prepared image path and returns the same dataclasses the silhouette path does,
# so it is a drop-in replacement / refinement. Wire a real model behind these when GPU is available.
FaceLandmarkDetector = Callable[[Path], Landmarks]
PoseDetector = Callable[[Path], "dict[str, Vec2]"]


def detect_face_landmarks_ml(image_path: str | Path, *, model: object | None = None) -> Landmarks:
    """ML anime-face landmark detection (gated upgrade for stylized art).

    Optional/external like See-through (needs a GPU model). The headless default is the silhouette
    path (``extract_landmarks``); only reach for this when masks are too ambiguous to localize
    features (e.g. heavy stylization, occluded eyes).
    """
    raise NotImplementedError(
        "ML face-landmark detection is a gated upgrade; use extract_landmarks (silhouette path) "
        "for the headless default"
    )


def detect_pose_ml(image_path: str | Path, *, model: object | None = None) -> dict[str, Vec2]:
    """ML body-pose detection (DWPose-style, gated upgrade).

    Returns a joint map (``{"arm_l": (x, y), ...}``) in model space. Optional/external (needs a GPU
    model); limb joints fall back to silhouette extraction in ``extract_landmarks``.
    """
    raise NotImplementedError(
        "ML pose detection (DWPose-style) is a gated upgrade; limb joints fall back to silhouette "
        "extraction in extract_landmarks"
    )


# --------------------------------------------------------------------------------------------------
# Landmark QA (Phase 3 #18) — sanity checks, render-free
# --------------------------------------------------------------------------------------------------
def landmark_warnings(lm: Landmarks) -> list[str]:
    """Flag implausible landmarks (degenerate oval, pupil outside its eye, swapped/inverted mouth).

    Returns short codes (empty == clean) so the CLI and QA harness can surface them uniformly.
    """
    out: list[str] = []
    if lm.face_oval and (lm.face_oval.radius_x <= 0.0 or lm.face_oval.radius_y <= 0.0):
        out.append("degenerate_face_oval")

    for name, eye in (("eye_l", lm.eye_l), ("eye_r", lm.eye_r)):
        if eye and eye.pupil:
            px, py = eye.pupil
            xlo, xhi = sorted((eye.inner[0], eye.outer[0]))
            ylo, yhi = sorted((eye.lid_top[1], eye.lid_bottom[1]))
            if not (xlo <= px <= xhi and ylo <= py <= yhi):
                out.append(f"pupil_outside_eye:{name}")

    if lm.mouth:
        if lm.mouth.left_corner[0] > lm.mouth.right_corner[0]:
            out.append("mouth_corners_swapped")
        if lm.mouth.top[1] < lm.mouth.bottom[1]:
            out.append("mouth_inverted")
    return out


# --------------------------------------------------------------------------------------------------
# Debug overlay (Phase 3 #18) — composite the character and draw the landmarks
# --------------------------------------------------------------------------------------------------
def _to_px(pt: Vec2, w: int, h: int) -> tuple[float, float]:
    """Model space (y up, [0,1]) -> image pixels (y down)."""
    return (pt[0] * w, (1.0 - pt[1]) * h)


def render_overlay(stack: LayerStack, landmarks: Landmarks, out_path: str | Path) -> Path:
    """Composite the character's layers and draw the extracted landmarks over them (needs Pillow).

    A green oval for the face, cyan crosses for eye corners/lids, magenta dots for pupils, yellow for
    mouth corners, orange for brows, red squares for limb joints. The fast visual check that the
    silhouette extractor located the right features before trusting the solver.
    """
    from PIL import Image, ImageDraw

    w = stack.canvas_width or 512
    h = stack.canvas_height or 512
    base = Image.new("RGBA", (w, h), (30, 30, 36, 255))
    for layer in stack.layers:  # already sorted by draw_order
        with Image.open(layer.texture_path) as img:
            lyr = img.convert("RGBA")
        if lyr.size != (w, h):
            canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            canvas.paste(lyr, (0, 0), lyr)
            lyr = canvas
        base = Image.alpha_composite(base, lyr)

    draw = ImageDraw.Draw(base)
    r = max(3, w // 160)

    def cross(pt: Vec2, color: tuple[int, int, int]) -> None:
        x, y = _to_px(pt, w, h)
        draw.line([(x - r, y), (x + r, y)], fill=color, width=2)
        draw.line([(x, y - r), (x, y + r)], fill=color, width=2)

    def dot(pt: Vec2, color: tuple[int, int, int]) -> None:
        x, y = _to_px(pt, w, h)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

    def box(pt: Vec2, color: tuple[int, int, int]) -> None:
        x, y = _to_px(pt, w, h)
        draw.rectangle([x - r, y - r, x + r, y + r], outline=color, width=2)

    if landmarks.face_oval:
        cx, cy = _to_px(landmarks.face_oval.center, w, h)
        rx = landmarks.face_oval.radius_x * w
        ry = landmarks.face_oval.radius_y * h
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=(80, 255, 120), width=2)
        dot(landmarks.face_oval.center, (80, 255, 120))

    for eye in (landmarks.eye_l, landmarks.eye_r):
        if not eye:
            continue
        for pt in (eye.lid_top, eye.lid_bottom, eye.inner, eye.outer):
            cross(pt, (0, 220, 255))
        if eye.pupil:
            dot(eye.pupil, (255, 60, 220))

    if landmarks.mouth:
        for pt in (landmarks.mouth.left_corner, landmarks.mouth.right_corner,
                   landmarks.mouth.top, landmarks.mouth.bottom):
            cross(pt, (255, 220, 0))

    for brow in (landmarks.brow_l, landmarks.brow_r):
        if brow:
            for pt in (brow.inner, brow.outer):
                cross(pt, (255, 150, 40))

    for joint in landmarks.joints.values():
        box(joint, (255, 60, 60))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(out)
    return out
