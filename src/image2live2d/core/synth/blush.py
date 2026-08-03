"""Synthesise a cheek blush overlay (drives ``ParamCheek``).

See-through never emits a blush layer, so — like the mouth cavity and the closed-eye lash line — an
auto-rigger has to paint one. This paints two soft radial ovals on the cheeks (below and outward of each
eye), tinted from the character's OWN skin colour pushed toward a warm pink, onto one full-canvas
transparent layer.

Deliberately the safest synth we do:

* It is **hidden at rest** — ``author._cheek`` authors ``ParamCheek`` so the blush is opacity 0 at
  ``ParamCheek=0`` and only fades in when the parameter is driven. A character with a synthesised blush
  therefore renders **byte-identical** to one without until someone asks for it, so an imperfect blush can
  never regress the resting model (unlike the cavity / closed-eye, which must read in the default pose).
* Locating is anchored to the **eye** boxes (cheeks sit just below and outside the eyes), and synthesis is
  skipped unless BOTH eyes are present, so we never guess a cheek from a partial face.
* The colour is the character's own skin tone blended toward pink, never a foreign hue, so it stays in
  palette. If Pillow/NumPy is unavailable or an eye texture is unreadable, synthesis is skipped.
"""

from __future__ import annotations

from pathlib import Path

from ..types import Layer, LayerStack
from ...irr.schema import SemanticRole

# Cheek placement, all relative to each eye's own box: shifted outward (away from the face midline) and
# dropped below the eye, with oval radii scaled to the eye. Tuned so the ovals sit on the cheekbones, well
# clear of the eye and the nose bridge.
_CHEEK_OUT_FRAC = 0.30    # centre shifted outward from the eye centre, as a fraction of eye WIDTH
_CHEEK_DROP_FRAC = 0.95   # ...and below the eye bottom, as a fraction of eye HEIGHT
_CHEEK_RX_FRAC = 0.62     # oval x-radius as a fraction of eye width
_CHEEK_RY_FRAC = 0.85     # oval y-radius as a fraction of eye height
_PEAK_ALPHA = 185         # peak opacity of the gradient at full drive (the layer is still opacity-gated
#                           to 0 at rest by ParamCheek, so this is what shows at ParamCheek=1)
_PINK = (246, 120, 132)   # the warm pink the skin tone is blended toward
_PINK_MIX = 0.70          # how far from the skin colour toward _PINK (0 = skin, 1 = pure pink)
_FALLOFF = 3.0            # gaussian tightness: alpha ~ exp(-falloff * (r/radius)^2)


def _solid_bbox(img) -> tuple[int, int, int, int] | None:
    """Scatter-robust ``(x0, y0, x1, y1)`` (exclusive) of the layer's solid mass — the same guard the
    mesh builder and the other synths use, so a faint decomposer halo can't inflate the box."""
    from ..mesh.build import DEFAULT_ALPHA_THRESHOLD, alpha_bbox

    px = img.getchannel("A").load()
    w, h = img.size
    box = alpha_bbox(lambda x, y: px[x, y], w, h, DEFAULT_ALPHA_THRESHOLD)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return x0, y0, x1 + 1, y1 + 1


def _blush_color(stack: LayerStack) -> tuple[int, int, int]:
    """The character's own skin colour (median solid pixel of ``face_base``) blended toward ``_PINK``.
    Falls back to pure ``_PINK`` if there is no readable face."""
    import numpy as np
    from PIL import Image

    faces = stack.by_role(SemanticRole.face_base)
    if faces:
        src = Path(faces[0].texture_path)
        if src.is_file():
            a = np.asarray(Image.open(src).convert("RGBA"))
            rgb, alpha = a[:, :, :3].astype(float), a[:, :, 3]
            solid = alpha > 128
            if solid.any():
                skin = np.median(rgb[solid], axis=0)
                mixed = skin * (1.0 - _PINK_MIX) + np.array(_PINK, dtype=float) * _PINK_MIX
                return (int(mixed[0]), int(mixed[1]), int(mixed[2]))
    return _PINK


def synthesize_blush(stack: LayerStack) -> Layer | None:
    """Paint a two-cheek blush overlay and splice it into ``stack`` (above the skin), or ``None`` if there
    is nothing to do. Needs both eyes; mutates ``stack``. The image is written beside the face texture, a
    decomposition work product like every other synthesised layer."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError:                                   # pragma: no cover - Pillow/NumPy gated
        return None

    if stack.by_role(SemanticRole.blush):
        return None                                        # already have one (real or synthesised)
    el, er = stack.by_role(SemanticRole.eye_l), stack.by_role(SemanticRole.eye_r)
    faces = stack.by_role(SemanticRole.face_base)
    if not el or not er or not faces:
        return None                                        # need both eyes + a face to place/anchor on

    boxes: list[tuple[int, int, int, int]] = []
    size: tuple[int, int] | None = None
    ref_src: Path | None = None
    for eyes in (el, er):
        src = Path(eyes[0].texture_path)
        if not src.is_file():
            return None
        img = Image.open(src).convert("RGBA")
        size, ref_src = img.size, src
        b = _solid_bbox(img)
        if b is None:
            return None
        boxes.append(b)

    W, H = size
    centres = [((x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0) for x0, y0, x1, y1 in boxes]
    face_cx = sum(c[0] for c in centres) / 2.0
    r, g, b = _blush_color(stack)

    canvas = np.zeros((H, W, 4), dtype=np.float32)
    for cx, cy, ew, eh in centres:
        out_dir = 1.0 if cx >= face_cx else -1.0
        bx = cx + out_dir * _CHEEK_OUT_FRAC * ew
        by = (cy + eh / 2.0) + _CHEEK_DROP_FRAC * eh       # below the eye bottom (image y is DOWN)
        rx = max(_CHEEK_RX_FRAC * ew, 1.0)
        ry = max(_CHEEK_RY_FRAC * eh, 1.0)
        # paint the gaussian only over the oval's local box, clipped to the canvas
        px0, px1 = max(0, int(bx - 2 * rx)), min(W, int(bx + 2 * rx) + 1)
        py0, py1 = max(0, int(by - 2 * ry)), min(H, int(by + 2 * ry) + 1)
        if px0 >= px1 or py0 >= py1:
            continue
        ys, xs = np.mgrid[py0:py1, px0:px1]
        d2 = ((xs - bx) / rx) ** 2 + ((ys - by) / ry) ** 2
        a = _PEAK_ALPHA * np.exp(-_FALLOFF * d2)
        a[d2 > 4.0] = 0.0                                  # hard cut past 2 radii (already ~0)
        sub = canvas[py0:py1, px0:px1]
        sub[..., 0] = r
        sub[..., 1] = g
        sub[..., 2] = b
        sub[..., 3] = np.maximum(sub[..., 3], a)           # keep the stronger where the ovals meet

    out = ref_src.with_name(f"{faces[0].draw_order}_{SemanticRole.blush.value}.png")
    Image.fromarray(canvas.clip(0, 255).astype("uint8"), "RGBA").save(out)

    face = faces[0]
    layer = Layer(
        id=out.stem,
        semantic_role=SemanticRole.blush,
        texture_path=out,
        draw_order=face.draw_order + 1,      # just above the skin; normalize_face_zorder finalises order
        width=W,
        height=H,
        bbox=face.bbox,
    )
    stack.layers.insert(stack.layers.index(face) + 1, layer)
    return layer
