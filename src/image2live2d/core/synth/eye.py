"""Synthesise a closed-eye lash line.

A decomposed eye is three parts: ``eye_white`` (the filled sclera), ``pupil`` (the iris), and ``eye_l``
/ ``eye_r`` — the dark **lash-line lineart** (measured ~40x30 px, only ~30% filled: strokes, not a
region). There is no *closed*-eye art. So the only way to shut the eye was to squash the open parts
toward the lid line (``author._blink``), which leaves a compressed sliver of iris and white — never the
clean lash line a real closed eye is. You cannot close an eye that has no closed pose.

Hand-drawn rigs get a closed-eye layer for free — the artist draws the lid-down lash line and swaps it in
on ``ParamEyeOpen``. An auto-rigger has to make one. This module paints a shallow lid arc across the eye,
in the character's own lash colour, inserts it as a part just above the eye group, and lets the rig
crossfade it in as the eye closes (opacity 0 -> 1) while the open parts fade out. It needs the per-keyform
opacity the moc3 emitter now honours (see backends.live2d.moc3_emit).

Deliberately conservative, exactly like the mouth cavity it mirrors:

* The lash line's **colour and thickness come from the eye's own lineart**, never invented, so it sits
  inside the character's palette.
* It is **hidden at rest** (opacity 0 while the eye is open), so an open-eyed rig renders identically to
  one with no closed-eye part — this can never make an open eye look worse.
* If there is no eye lineart layer, or Pillow is unavailable, synthesis is skipped and the rig is
  unchanged.
"""

from __future__ import annotations

from pathlib import Path

from ..types import Layer, LayerStack
from ...irr.schema import SemanticRole

# The lid arc spans the eye's WIDTH and sits a little below its vertical centre — a closed lid rests low
# in the socket, not across the middle. Its sag (how far the corners drop below the crown) and stroke
# weight are shallow fractions of the eye's size, so the line reads as a relaxed blink, not an expression.
_ARC_SAG_FRAC = 0.18          # corner drop below the arc crown, as a fraction of eye height
_ARC_CENTER_FRAC = 0.55       # arc crown y, as a fraction down the eye box (0 = top, 1 = bottom)
_STROKE_H_FRAC = 0.22         # stroke thickness as a fraction of eye height...
_STROKE_MIN_PX = 2.0          # ...but never thinner than this (a hairline eye must still show a line)
_ARC_W_INSET = 0.06           # pull the ends in from the very corners so the line doesn't touch the rim

# Paint magnified and shrink back: the eye is ~40 px across, so a stroke lands on a couple of pixels and
# stair-steps without supersampling (same reason as core.synth.mouth).
_SUPERSAMPLE = 4

# left/right eye -> the lineart role it is painted from and the closed role it becomes.
_SIDES = (
    (SemanticRole.eye_l, SemanticRole.eye_closed_l),
    (SemanticRole.eye_r, SemanticRole.eye_closed_r),
)


def _alpha_bbox_solid(img) -> tuple[int, int, int, int] | None:
    """Scatter-robust ``(x0, y0, x1, y1)`` (x1/y1 exclusive) of the layer's solid mass, or ``None`` if
    empty. Uses the mesh builder's solid-mass bbox so a faint decomposer halo can't inflate the box
    (same guard as core.synth.mouth)."""
    from ..mesh.build import alpha_bbox, DEFAULT_ALPHA_THRESHOLD

    px = img.getchannel("A").load()
    w, h = img.size
    box = alpha_bbox(lambda x, y: px[x, y], w, h, DEFAULT_ALPHA_THRESHOLD)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return x0, y0, x1 + 1, y1 + 1


def _lash_color(img) -> tuple[int, int, int]:
    """The eye lineart's own lash colour: the mean of its darkest quartile of solid pixels (the lashes
    themselves, not the soft antialiased edge). Falls back to a near-black if the layer gave nothing."""
    import numpy as np

    a = np.asarray(img)
    rgb, alpha = a[:, :, :3].astype(float), a[:, :, 3]
    solid = alpha > 128
    if not solid.any():
        return (30, 20, 25)
    px = rgb[solid]
    lum = px.mean(axis=1)
    dark = px[lum <= np.percentile(lum, 25)].mean(axis=0)
    return (int(dark[0]), int(dark[1]), int(dark[2]))


def _paint_arc(size, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> "object":
    """A shallow lid arc across ``box``: crown at the centre, corners sagging ``_ARC_SAG_FRAC`` below, a
    rounded stroke ``_STROKE_H_FRAC`` of the eye height thick. Drawn as a thick polyline with round joints
    so the ends cap cleanly."""
    from PIL import Image, ImageDraw

    s = _SUPERSAMPLE
    x0, y0, x1, y1 = (v * s for v in box)
    w, h = x1 - x0, y1 - y0
    inset = w * _ARC_W_INSET
    ax0, ax1 = x0 + inset, x1 - inset
    crown = y0 + h * _ARC_CENTER_FRAC
    sag = h * _ARC_SAG_FRAC
    stroke = max(h * _STROKE_H_FRAC, _STROKE_MIN_PX * s)

    big = Image.new("RGBA", (size[0] * s, size[1] * s), (0, 0, 0, 0))
    pen = ImageDraw.Draw(big)
    half = max((ax1 - ax0) / 2.0, 1e-6)
    cx = (ax0 + ax1) / 2.0
    # crown at centre (t=0), corners drop by sag (t=±1): a parabola reads as a relaxed lid.
    pts = []
    n = 24
    for i in range(n + 1):
        px_ = ax0 + (ax1 - ax0) * i / n
        t = (px_ - cx) / half
        pts.append((px_, crown + sag * t * t))
    pen.line(pts, fill=(*color, 255), width=int(round(stroke)), joint="curve")
    # round the two end caps (line() leaves them square)
    r = stroke / 2.0
    for ex, ey in (pts[0], pts[-1]):
        pen.ellipse([ex - r, ey - r, ex + r, ey + r], fill=(*color, 255))
    # BOX (area-average) shrink — LANCZOS would ring a stray-alpha halo outside the lids it hides behind.
    return big.resize(size, Image.BOX)


def _synth_one(stack: LayerStack, src_role: SemanticRole, closed_role: SemanticRole) -> Layer | None:
    from PIL import Image

    eyes = stack.by_role(src_role)
    if not eyes or stack.by_role(closed_role):
        return None
    eye = eyes[0]
    src = Path(eye.texture_path)
    if not src.is_file():
        return None
    img = Image.open(src).convert("RGBA")
    box = _alpha_bbox_solid(img)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    if (x1 - x0) < 2 or (y1 - y0) < 2:
        return None

    arc = _paint_arc(img.size, box, _lash_color(img))
    out = src.with_name(f"{eye.draw_order}_{closed_role.value}.png")
    arc.save(out)

    layer = Layer(
        id=out.stem,
        semantic_role=closed_role,
        texture_path=out,
        draw_order=eye.draw_order + 1,   # just *above* the eye group — it shows on top when the eye shuts
        width=img.width,
        height=img.height,
        bbox=eye.bbox,
    )
    # insert right after the eye lineart so draw order stays coherent
    stack.layers.insert(stack.layers.index(eye) + 1, layer)
    return layer


def synthesize_closed_eyes(stack: LayerStack) -> list[Layer]:
    """Paint a closed-eye lash line for each eye that has lineart but no closed pose, splice them into
    ``stack``, and return the new layers (possibly empty). Mutates ``stack``.

    The images are written beside the eye textures (a decomposition work product, like every other
    synthesised layer). If Pillow is missing the whole step is a no-op.
    """
    try:
        import PIL  # noqa: F401
    except ImportError:                                   # pragma: no cover - Pillow gated
        return []
    made = []
    for src_role, closed_role in _SIDES:
        layer = _synth_one(stack, src_role, closed_role)
        if layer is not None:
            made.append(layer)
    return made
