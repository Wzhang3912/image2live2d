"""Generate a throwaway, fully-synthetic sample character as decomposition-style layers.

This stands in for See-through so you have something to push through the spine *today*: it draws a
simple anime-ish face as a set of full-canvas RGBA PNGs named with the
``{draw_order}_{semantic_role}.png`` convention that ``decompose.from_layer_dir`` consumes. The art
is deliberately crude — its only job is to exercise blink / mouth / head-turn / eyeball / brow in a
real viewer.

Requires Pillow (the ``preprocess`` / ``decompose`` extra).
"""

from __future__ import annotations

from pathlib import Path

# Palette (RGBA).
_SKIN = (255, 224, 196, 255)
_HAIR = (96, 64, 128, 255)
_WHITE = (255, 255, 255, 255)
_LASH = (48, 36, 48, 255)
_IRIS = (72, 48, 160, 255)
_BROW = (120, 84, 150, 255)
_NOSE = (224, 184, 162, 220)
_MOUTH = (206, 92, 110, 255)


def make_sample_layers(out_dir: str | Path, *, size: int = 512) -> Path:
    """Draw the sample layers into ``out_dir`` and return it. Coordinates are authored on a 512px
    canvas and scaled by ``size / 512``."""
    from PIL import Image, ImageDraw

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s = size / 512.0

    def b(*coords: float) -> tuple[float, ...]:
        return tuple(c * s for c in coords)

    def face_base(d: "ImageDraw.ImageDraw") -> None:
        d.ellipse(b(150, 120, 362, 430), fill=_SKIN)

    def hair_front(d: "ImageDraw.ImageDraw") -> None:
        # top-half ellipse = forehead fringe, ends above the eyebrows
        d.pieslice(b(126, 84, 386, 320), 180, 360, fill=_HAIR)

    def eyebrow_l(d: "ImageDraw.ImageDraw") -> None:
        d.ellipse(b(186, 206, 244, 220), fill=_BROW)

    def eyebrow_r(d: "ImageDraw.ImageDraw") -> None:
        d.ellipse(b(268, 206, 326, 220), fill=_BROW)

    def eye_white_l(d: "ImageDraw.ImageDraw") -> None:
        d.ellipse(b(180, 235, 242, 278), fill=_WHITE)

    def eye_white_r(d: "ImageDraw.ImageDraw") -> None:
        d.ellipse(b(270, 235, 332, 278), fill=_WHITE)

    def eye_l(d: "ImageDraw.ImageDraw") -> None:
        d.ellipse(b(180, 235, 242, 278), outline=_LASH, width=max(2, int(5 * s)))

    def eye_r(d: "ImageDraw.ImageDraw") -> None:
        d.ellipse(b(270, 235, 332, 278), outline=_LASH, width=max(2, int(5 * s)))

    def pupil_l(d: "ImageDraw.ImageDraw") -> None:
        d.ellipse(b(200, 245, 224, 272), fill=_IRIS)

    def pupil_r(d: "ImageDraw.ImageDraw") -> None:
        d.ellipse(b(288, 245, 312, 272), fill=_IRIS)

    def nose(d: "ImageDraw.ImageDraw") -> None:
        d.polygon([b(252, 286), b(262, 300), b(248, 300)], fill=_NOSE)

    def mouth(d: "ImageDraw.ImageDraw") -> None:
        d.chord(b(226, 318, 286, 348), 0, 180, fill=_MOUTH)

    # (draw_order, role, fn) — back to front
    spec = [
        (0, "face_base", face_base),
        (40, "eyebrow_l", eyebrow_l),
        (41, "eyebrow_r", eyebrow_r),
        (50, "eye_white_l", eye_white_l),
        (51, "eye_white_r", eye_white_r),
        (52, "eye_l", eye_l),
        (53, "eye_r", eye_r),
        (55, "nose", nose),
        (60, "pupil_l", pupil_l),
        (61, "pupil_r", pupil_r),
        (70, "mouth", mouth),
        (90, "hair_front", hair_front),
    ]

    for order, role, fn in spec:
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        fn(ImageDraw.Draw(img))
        img.save(out / f"{order:02d}_{role}.png")

    return out


_SKIN_LIMB = (255, 216, 186, 255)
_CLOTH = (96, 132, 196, 255)
_SKIRT = (72, 104, 168, 255)


def make_sample_fullbody(out_dir: str | Path, *, size: int = 512) -> Path:
    """Draw a crude chibi full-body character (head + hair_back/side/front + torso + arms + legs) so
    the body params and hair physics are exercisable in nijigenerate. Not pretty — functional."""
    from PIL import Image, ImageDraw

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s = size / 512.0

    def b(*c: float) -> tuple[float, ...]:
        return tuple(v * s for v in c)

    def ell(*box, fill):
        return lambda d: d.ellipse(b(*box), fill=fill)

    def rect(*box, fill):
        return lambda d: d.rounded_rectangle(b(*box), radius=12 * s, fill=fill)

    # (draw_order, role, fn) — back to front
    spec = [
        (0, "hair_back", ell(150, 40, 362, 270, fill=_HAIR)),
        (5, "leg_l", rect(212, 395, 250, 500, fill=_SKIN_LIMB)),
        (6, "leg_r", rect(262, 395, 300, 500, fill=_SKIN_LIMB)),
        (8, "arm_l", rect(168, 255, 205, 380, fill=_SKIN_LIMB)),
        (9, "arm_r", rect(307, 255, 344, 380, fill=_SKIN_LIMB)),
        (12, "torso", rect(200, 248, 312, 405, fill=_CLOTH)),
        # a trapezoid skirt over the upper legs (clothing role -> body-driven hem physics)
        (13, "clothing", lambda d: d.polygon(b(206, 360, 306, 360, 330, 440, 182, 440), fill=_SKIRT)),
        (20, "hair_side", lambda d: (d.ellipse(b(150, 90, 196, 250), fill=_HAIR),
                                     d.ellipse(b(316, 90, 362, 250), fill=_HAIR))),
        (24, "face_base", ell(170, 70, 342, 252, fill=_SKIN)),
        (40, "eyebrow_l", ell(196, 150, 238, 162, fill=_BROW)),
        (41, "eyebrow_r", ell(274, 150, 316, 162, fill=_BROW)),
        (50, "eye_white_l", ell(198, 168, 240, 200, fill=_WHITE)),
        (51, "eye_white_r", ell(272, 168, 314, 200, fill=_WHITE)),
        (52, "eye_l", lambda d: d.ellipse(b(198, 168, 240, 200), outline=_LASH, width=max(2, int(5 * s)))),
        (53, "eye_r", lambda d: d.ellipse(b(272, 168, 314, 200), outline=_LASH, width=max(2, int(5 * s)))),
        (60, "pupil_l", ell(212, 174, 230, 196, fill=_IRIS)),
        (61, "pupil_r", ell(286, 174, 304, 196, fill=_IRIS)),
        (55, "nose", lambda d: d.polygon([b(252, 205), b(262, 218), b(248, 218)], fill=_NOSE)),
        (70, "mouth", lambda d: d.chord(b(232, 222, 280, 244), 0, 180, fill=_MOUTH)),
        (90, "hair_front", lambda d: d.pieslice(b(160, 50, 352, 230), 180, 360, fill=_HAIR)),
    ]

    for order, role, fn in spec:
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        fn(ImageDraw.Draw(img))
        img.save(out / f"{order:02d}_{role}.png")

    return out
