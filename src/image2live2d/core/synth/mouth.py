"""Synthesise the inside of the mouth.

A decomposed mouth layer is only the *lip line* — on a real character it came out 21x6 px, 118 opaque
pixels: a thin closed-smile stroke. There is nothing behind it. So ``ParamMouthOpenY`` deformed the
lips correctly (the P5 lens: lower lip drops, upper lip rises, corners stay anchored) and yet the mouth
never *opened*, because parting a stroke over bare skin just reveals more skin. You cannot open a mouth
that has no interior.

Hand-drawn rigs get an inner-mouth layer for free — the artist paints the cavity (and often teeth and a
tongue) behind the lips. An auto-rigger has to make one. This module paints a cavity from the lip
line's own geometry and colour, inserts it as a part just under the lips, and lets the rig drive it:
collapsed to nothing while the mouth is shut (so a closed mouth is exactly the lip line it always was),
opening into a lens as ``ParamMouthOpenY`` rises.

Deliberately conservative:

* The cavity is **derived from the lip line**, never invented — its width, position and hue all come
  from the mouth layer's own pixels, so it cannot clash with the character's palette.
* It is **hidden at rest**. A rig whose mouth stays shut renders identically to one with no cavity at
  all, so this can never make a closed mouth look worse.
* If there is no mouth layer, or Pillow is unavailable, synthesis is skipped and the rig is unchanged.
"""

from __future__ import annotations

import colorsys
from pathlib import Path

from ..types import Layer, LayerStack
from ...irr.schema import SemanticRole

# The cavity is sized from the mouth's WIDTH, never from the lip line's height. The height of a closed
# lip line is its stroke weight (21x6 px on a real character); scaling off it made the cavity taller
# than it was wide — a brown blob hanging past the chin. Width is the mouth's true extent, so an opening
# is a fraction of it, and the cavity comes out wider than tall, like a mouth.
_CAVITY_H_FRAC = 0.55        # cavity height as a fraction of mouth width — matches the lip lens opening
# Inset from the lip line's width — the corners of a mouth close before its centre does.
_CAVITY_W_INSET = 0.10
# The lips part about their own line, so the cavity has to straddle it: mostly below (the jaw drops),
# but far enough above to back the upper lip's smaller rise.
_CAVITY_RISE_FRAC = 0.25

# The three tones of an open anime mouth, all struck from the lip line's own hue so the interior can
# never fall outside the character's palette: a dark hollow, a bright band of upper teeth just under the
# lip, and a warmer tongue filling the floor. (saturation, value) per tone.
_TONE_CAVITY = (0.66, 0.28)   # the hollow: much darker than the lips — most of an open mouth is shadow,
#                               and the teeth only read as teeth against something dark
_TONE_TEETH = (0.05, 0.98)    # near-white, only faintly tinted — enamel, not paint
_TONE_TONGUE = (0.42, 0.72)   # lighter and pinker than the hollow it sits in

_TEETH_H_FRAC = 0.26          # upper teeth as a fraction of cavity height, hanging from its roof
_TONGUE_W_FRAC = 0.58         # the tongue is a bump on the floor of the mouth, not its contents:
_TONGUE_H_FRAC = 0.42         # narrower than the mouth and confined to the lower part of it

# The mouth is a couple of dozen pixels across on a 1280 canvas, so teeth land on 2-3 of them. Paint the
# cavity magnified and shrink it back down, or the detail is a stair-stepped smear rather than a mouth.
_SUPERSAMPLE = 4

# Only a *closed* mouth needs an interior painted for it. A mouth the artist already drew open carries
# its own — real teeth, a real tongue, in the character's actual style — and painting behind that would
# be inventing over art that already exists.
#
# Aspect ratio ALONE was tried and is not enough: a closed smile with lip shading (an upper and lower
# lip, a soft curve) has a tall solid box too — measured aspect 0.41-0.69 on four test characters that
# are all plainly closed — so the aspect gate denied them a cavity and their mouths could not open (found
# by the capability report: 5 of 8 characters). What actually marks a drawn-open mouth is a genuine dark
# **interior cavity** — pixels much darker than the character's skin, a real oral shadow, not a light
# lip. So a mouth is "already open" only when it is BOTH tall AND has that dark interior: a closed smile
# (light interior, darkfrac ~0) gets its cavity; a true open mouth (dark cavity + teeth) is left alone.
_CLOSED_MAX_ASPECT = 0.40     # height/width below this the box is a stroke — always a closed mouth
_DARK_INTERIOR_FRAC = 0.15    # a tall box is "open" only if at least this fraction of its solid pixels
_DARK_REL_SKIN = 0.60         # are darker than _DARK_REL_SKIN x the skin luminance (a real cavity)


def _palette(img) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    """``(cavity, teeth, tongue)`` — all three struck from the lip line's own darkest pixels, so the
    interior sits inside the character's palette rather than a hard-coded red."""
    import numpy as np

    a = np.asarray(img)
    rgb, alpha = a[:, :, :3].astype(float), a[:, :, 3]
    solid = alpha > 128
    hue = 0.98                                        # a neutral mouth-red if the lips gave us nothing
    if solid.any():
        px = rgb[solid]
        # the darkest quartile: the lip *line*, not the soft antialiased skin-side edge
        lum = px.mean(axis=1)
        dark = px[lum <= np.percentile(lum, 25)]
        r, g, b = dark.mean(axis=0) / 255.0
        hue, _, _ = colorsys.rgb_to_hsv(r, g, b)

    def tone(sv: tuple[float, float]) -> tuple[int, int, int]:
        r, g, b = colorsys.hsv_to_rgb(hue, sv[0], sv[1])
        return (int(r * 255), int(g * 255), int(b * 255))

    return tone(_TONE_CAVITY), tone(_TONE_TEETH), tone(_TONE_TONGUE)


def _paint_interior(size, box, palette) -> "object":
    """The inside of a mouth: a dark hollow, upper teeth under the lip, a tongue on the floor.

    Teeth and tongue are drawn as plain shapes and then **clipped by the cavity's own outline**, so the
    ellipse does the work of curving them — the teeth come out as a crescent following the roof of the
    mouth, and the tongue as a dome sitting in its floor, without either being modelled as a curve.
    """
    from PIL import Image, ImageDraw

    cavity, teeth, tongue = palette
    s = _SUPERSAMPLE
    cx0, cy0, cx1, cy1 = (v * s for v in box)
    w, h = cx1 - cx0, cy1 - cy0

    big = Image.new("RGBA", (size[0] * s, size[1] * s), (0, 0, 0, 0))
    ink = ImageDraw.Draw(big)

    # the hollow, and the outline that clips everything inside it
    ink.ellipse([cx0, cy0, cx1, cy1], fill=(*cavity, 255))
    mask = Image.new("L", big.size, 0)
    ImageDraw.Draw(mask).ellipse([cx0, cy0, cx1, cy1], fill=255)

    inner = Image.new("RGBA", big.size, (0, 0, 0, 0))
    pen = ImageDraw.Draw(inner)
    # upper teeth: a straight band across the roof — the ellipse clips it into a crescent
    pen.rectangle([cx0, cy0, cx1, cy0 + h * _TEETH_H_FRAC], fill=(*teeth, 255))
    # the tongue: a dome on the floor, running off the bottom edge so the clip rounds it off
    tw, th = w * _TONGUE_W_FRAC, h * _TONGUE_H_FRAC
    tx = cx0 + (w - tw) / 2.0
    pen.ellipse([tx, cy1 - th, tx + tw, cy1 + th], fill=(*tongue, 255))

    big.paste(inner, (0, 0), Image.composite(inner.split()[3], Image.new("L", big.size, 0), mask))
    # BOX, not LANCZOS: a supersampled shrink wants a plain area-average. LANCZOS rings, and its
    # negative lobes smear a halo of stray alpha *outside* the cavity — past the lips it hides behind.
    return big.resize(size, Image.BOX)


def _alpha_bbox_solid(img) -> tuple[int, int, int, int] | None:
    """Scatter-robust replacement for ``Image.getbbox()`` on a decomposer layer.

    Returns ``(x0, y0, x1, y1)`` with ``x1``/``y1`` **exclusive** (getbbox convention), or ``None`` if
    the layer is empty. Delegates to the mesh builder's solid-mass bbox so the faint full-canvas halo
    can't inflate the box (see PR #47)."""
    from ..mesh.build import alpha_bbox, DEFAULT_ALPHA_THRESHOLD

    px = img.getchannel("A").load()
    w, h = img.size
    box = alpha_bbox(lambda x, y: px[x, y], w, h, DEFAULT_ALPHA_THRESHOLD)
    if box is None:
        return None
    x0, y0, x1, y1 = box                                   # alpha_bbox is inclusive
    return x0, y0, x1 + 1, y1 + 1                           # -> exclusive, matching getbbox


def _skin_luminance(stack: LayerStack) -> float | None:
    """Median luminance of the character's face skin, from the ``face_base`` layer's solid pixels.

    The reference a cavity is measured against — a real oral shadow is much darker than skin. Returns
    ``None`` when there is no face to compare to (then the caller falls back to aspect alone)."""
    try:
        from PIL import Image
    except ImportError:                                   # pragma: no cover - Pillow gated
        return None
    import numpy as np

    faces = stack.by_role(SemanticRole.face_base)
    if not faces or not Path(faces[0].texture_path).is_file():
        return None
    arr = np.asarray(Image.open(faces[0].texture_path).convert("RGBA"))
    solid = arr[..., 3] >= 64
    if not solid.any():
        return None
    lum = arr[..., :3].astype(float) @ (0.299, 0.587, 0.114)
    return float(np.median(lum[solid]))


def _has_dark_interior(img, box: tuple[int, int, int, int], skin_lum: float | None) -> bool:
    """Does the mouth box hold a genuine dark cavity — solid pixels much darker than skin?

    True only when a real fraction of the lip box's solid pixels fall below ``_DARK_REL_SKIN`` x the skin
    luminance (a drawn-open mouth's oral shadow). A closed smile is lip/skin toned, so its darkfrac is
    ~0. With no skin reference we cannot tell a dark lip from a cavity, so we assume the old aspect-only
    verdict held (return ``True`` — the box was already tall, so treat it as open, unchanged behaviour)."""
    if skin_lum is None:
        return True
    import numpy as np

    x0, y0, x1, y1 = box
    reg = np.asarray(img)[y0:y1, x0:x1]
    solid = reg[..., 3] >= 64
    if not solid.any():
        return False
    lum = reg[..., :3].astype(float) @ (0.299, 0.587, 0.114)
    darkfrac = float((lum[solid] < _DARK_REL_SKIN * skin_lum).mean())
    return darkfrac >= _DARK_INTERIOR_FRAC


def synthesize_mouth_cavity(stack: LayerStack) -> Layer | None:
    """Paint an inner mouth behind the lips and splice it into ``stack``. Returns the new layer, or
    ``None`` if there is nothing to do.

    The image is written beside the mouth texture (a decomposition work product, same as every other
    layer). Mutates ``stack``.
    """
    try:
        from PIL import Image
    except ImportError:                                   # pragma: no cover - Pillow gated
        return None

    mouths = stack.by_role(SemanticRole.mouth)
    if not mouths or stack.by_role(SemanticRole.mouth_cavity):
        return None
    lips = mouths[0]

    src = Path(lips.texture_path)
    if not src.is_file():
        return None
    img = Image.open(src).convert("RGBA")
    # A See-through mouth layer carries a near-transparent halo dusted across the whole canvas (measured
    # alpha 8-63), so PIL's raw getbbox() — anything > 0 — returned the ENTIRE 1280x1280 canvas. A
    # full-canvas box reads as aspect 1.0, tripping the "already drawn open" skip below, so the cavity was
    # never painted and the mouth could not open (seen on the silverdress test character). Use the same
    # solid-mass-weighted bbox the mesh builder uses (PR #47) so the faint sprinkle can't inflate the box.
    box = _alpha_bbox_solid(img)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    lip_w, lip_h = x1 - x0, y1 - y0
    if lip_w < 2 or lip_h < 1:
        return None
    if lip_h > _CLOSED_MAX_ASPECT * lip_w and _has_dark_interior(img, box, _skin_luminance(stack)):
        return None    # tall AND a real dark cavity -> already drawn open; leave the artist's own in

    cav_h = lip_w * _CAVITY_H_FRAC                  # from the WIDTH — the mouth's only honest dimension
    inset = lip_w * _CAVITY_W_INSET
    cx0, cx1 = x0 + inset, x1 - inset
    # Straddle the lip line: mostly below it (the jaw is what drops), a little above to back the upper
    # lip's smaller rise.
    lip_mid = y0 + lip_h * 0.5
    cy0 = lip_mid - cav_h * _CAVITY_RISE_FRAC
    cy1 = cy0 + cav_h

    cavity = _paint_interior(img.size, (cx0, cy0, cx1, cy1), _palette(img))

    out = src.with_name(f"{lips.draw_order}_{SemanticRole.mouth_cavity.value}.png")
    cavity.save(out)

    layer = Layer(
        id=out.stem,
        semantic_role=SemanticRole.mouth_cavity,
        texture_path=out,
        draw_order=lips.draw_order,      # just *under* the lips (stable sort keeps the lips on top)
        width=img.width,
        height=img.height,
        bbox=lips.bbox,
    )
    stack.layers.insert(stack.layers.index(lips), layer)
    return layer
