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
# The interior is darker and less saturated than the lips themselves.
_CAVITY_VALUE = 0.42
_CAVITY_SAT = 0.55


def _lip_colour(img) -> tuple[int, int, int]:
    """A plausible interior hue, taken from the lip line's own darkest pixels so the cavity always sits
    inside the character's palette rather than a hard-coded red."""
    import numpy as np

    a = np.asarray(img)
    rgb, alpha = a[:, :, :3].astype(float), a[:, :, 3]
    solid = alpha > 128
    if not solid.any():
        return (90, 40, 45)
    px = rgb[solid]
    # the darkest quartile: the lip *line*, not the soft antialiased skin-side edge
    lum = px.mean(axis=1)
    dark = px[lum <= np.percentile(lum, 25)]
    r, g, b = (dark.mean(axis=0) / 255.0)
    h, _, _ = colorsys.rgb_to_hsv(r, g, b)
    r, g, b = colorsys.hsv_to_rgb(h, _CAVITY_SAT, _CAVITY_VALUE)
    return (int(r * 255), int(g * 255), int(b * 255))


def synthesize_mouth_cavity(stack: LayerStack) -> Layer | None:
    """Paint an inner mouth behind the lips and splice it into ``stack``. Returns the new layer, or
    ``None`` if there is nothing to do.

    The image is written beside the mouth texture (a decomposition work product, same as every other
    layer). Mutates ``stack``.
    """
    try:
        from PIL import Image, ImageDraw
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
    box = img.getbbox()                                    # tight alpha bounds of the lip line
    if box is None:
        return None
    x0, y0, x1, y1 = box
    lip_w, lip_h = x1 - x0, y1 - y0
    if lip_w < 2 or lip_h < 1:
        return None

    cav_h = lip_w * _CAVITY_H_FRAC                  # from the WIDTH — the mouth's only honest dimension
    inset = lip_w * _CAVITY_W_INSET
    cx0, cx1 = x0 + inset, x1 - inset
    # Straddle the lip line: mostly below it (the jaw is what drops), a little above to back the upper
    # lip's smaller rise.
    lip_mid = y0 + lip_h * 0.5
    cy0 = lip_mid - cav_h * _CAVITY_RISE_FRAC
    cy1 = cy0 + cav_h

    cavity = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(cavity).ellipse([cx0, cy0, cx1, cy1], fill=(*_lip_colour(img), 255))

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
