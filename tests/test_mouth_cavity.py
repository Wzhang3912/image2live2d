"""Synthesised inner mouth (core.synth.mouth + the ParamMouthOpenY rig).

A decomposed mouth layer is only the lip *line* — on a real character, 21x6 px of closed-smile stroke
with nothing behind it. ParamMouthOpenY parted the lips correctly and the mouth still never opened,
because parting a stroke over bare skin only reveals more skin. These pin that we paint an interior,
that it is invisible until the mouth opens, and that a rig without a mouth is left alone.
"""

from __future__ import annotations

import pytest

from image2live2d.core import decompose, mesh
from image2live2d.core.rig import author_rig, select_template
from image2live2d.core.synth import synthesize_mouth_cavity
from image2live2d.irr.schema import SemanticRole as R

pytest.importorskip("PIL")


def _layers(tmp_path, *, with_mouth=True):
    """A minimal face: a skin block and (optionally) a thin lip line — the shape a decomposer returns."""
    from PIL import Image, ImageDraw

    d = tmp_path / "layers"
    d.mkdir()
    face = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    ImageDraw.Draw(face).rectangle([30, 20, 98, 110], fill=(250, 220, 205, 255))
    face.save(d / "00_face_base.png")
    if with_mouth:
        lips = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
        ImageDraw.Draw(lips).rectangle([56, 84, 76, 87], fill=(150, 70, 80, 255))  # a 20x3 stroke
        lips.save(d / "20_mouth.png")
    return d


def test_a_cavity_is_painted_behind_the_lips(tmp_path):
    stack = decompose.from_layer_dir(_layers(tmp_path))
    layer = synthesize_mouth_cavity(stack)

    assert layer is not None
    assert layer.semantic_role is R.mouth_cavity
    assert layer.texture_path.is_file()
    # it sits *under* the lips, so the lip line still reads on top of the hole it covers
    ids = [ly.id for ly in stack.layers]
    assert ids.index(layer.id) < ids.index("20_mouth")


def test_the_cavity_is_taller_than_the_lip_line_it_hides_behind(tmp_path):
    """A closed-smile stroke is a few pixels tall; the mouth behind it opens far wider."""
    from PIL import Image

    stack = decompose.from_layer_dir(_layers(tmp_path))
    layer = synthesize_mouth_cavity(stack)
    lips = Image.open(tmp_path / "layers" / "20_mouth.png").getbbox()
    cav = Image.open(layer.texture_path).getbbox()

    lip_h = lips[3] - lips[1]
    cav_h = cav[3] - cav[1]
    assert cav_h > 2 * lip_h
    assert cav[0] >= lips[0] and cav[2] <= lips[2]      # inset: a mouth's corners close first


def test_the_cavity_is_invisible_until_the_mouth_opens(tmp_path):
    """At rest the mouth is shut, so the painted interior must collapse to nothing — a closed mouth has
    to render as the bare lip line it always was."""
    stack = decompose.from_layer_dir(_layers(tmp_path))
    synthesize_mouth_cavity(stack)
    meshes = mesh.build_meshes(stack)
    params = author_rig(stack, meshes, select_template(stack)).parameters

    p = next(p for p in params if p.id == "ParamMouthOpenY")
    shut = next(k for k in p.keyforms if k.value == 0.0)
    opened = next(k for k in p.keyforms if k.value == 1.0)
    cav = next(m for m in meshes if m.part_id.endswith("mouth_cavity"))

    shut_h = _height(cav, shut.mesh_offsets[cav.part_id])
    open_h = _height(cav, opened.mesh_offsets[cav.part_id])
    assert shut_h == pytest.approx(0.0, abs=1e-6)      # collapsed -> zero area -> not drawn
    assert open_h > 0.0                                # full size once the mouth is open


def test_no_mouth_means_no_cavity(tmp_path):
    stack = decompose.from_layer_dir(_layers(tmp_path, with_mouth=False))
    assert synthesize_mouth_cavity(stack) is None
    assert not stack.by_role(R.mouth_cavity)


def test_synthesis_is_idempotent(tmp_path):
    stack = decompose.from_layer_dir(_layers(tmp_path))
    assert synthesize_mouth_cavity(stack) is not None
    assert synthesize_mouth_cavity(stack) is None      # already has one; don't stack cavities
    assert len(stack.by_role(R.mouth_cavity)) == 1


def _height(m, offsets) -> float:
    ys = [y + offsets[i][1] for i, (_, y) in enumerate(m.vertices)]
    return max(ys) - min(ys)


def _interior_tones(layer):
    """The distinct tones down the centre of the painted cavity, brightest first."""
    import numpy as np
    from PIL import Image

    img = Image.open(layer.texture_path).convert("RGBA")
    a = np.asarray(img.crop(img.getbbox()))
    col = a[:, a.shape[1] // 2]                       # straight down the middle of the mouth
    return [tuple(int(v) for v in px[:3]) for px in col if px[3] > 200]


def test_the_interior_has_teeth_and_a_tongue_not_just_a_hole(tmp_path):
    """A flat lens reads as a hole punched in the face. An open anime mouth is three things: a dark
    hollow, a band of upper teeth under the lip, and a tongue on the floor."""
    stack = decompose.from_layer_dir(_layers(tmp_path))
    layer = synthesize_mouth_cavity(stack)
    tones = _interior_tones(layer)
    assert len(tones) > 5

    lum = [sum(t) / 3 for t in tones]
    teeth, hollow = max(lum), min(lum)
    assert teeth > 200                                # enamel: near-white
    assert hollow < 110                               # the mouth is mostly shadow
    # the teeth hang from the roof and the tongue sits on the floor, so brightness is not monotonic:
    # bright (teeth) -> dark (hollow) -> mid (tongue)
    top_third = lum[: len(lum) // 3]
    middle = lum[len(lum) // 3: 2 * len(lum) // 3]
    bottom = lum[2 * len(lum) // 3:]
    assert max(top_third) > max(middle)               # teeth are above the hollow
    assert max(bottom) > min(middle)                  # the tongue is lighter than the hollow it sits in


def test_the_interior_is_struck_from_the_character_s_own_lip_hue(tmp_path):
    """Nothing is hard-coded red: recolour the lips and the whole interior follows, so a cavity can
    never clash with the character's palette."""
    import colorsys

    from PIL import Image, ImageDraw

    d = tmp_path / "layers"
    d.mkdir()
    face = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    ImageDraw.Draw(face).rectangle([30, 20, 98, 110], fill=(250, 220, 205, 255))
    face.save(d / "00_face_base.png")
    lips = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    ImageDraw.Draw(lips).rectangle([56, 84, 76, 87], fill=(70, 90, 160, 255))   # blue lips
    lips.save(d / "20_mouth.png")

    layer = synthesize_mouth_cavity(decompose.from_layer_dir(d))
    hues = [colorsys.rgb_to_hsv(*(c / 255 for c in t))[0] for t in _interior_tones(layer)]
    lip_hue = colorsys.rgb_to_hsv(70 / 255, 90 / 255, 160 / 255)[0]
    assert all(abs(h - lip_hue) < 0.06 for h in hues)  # the whole interior tracks the lips


def test_a_mouth_already_drawn_open_is_left_alone(tmp_path):
    """If the artist drew the mouth open, the art already carries its own teeth and tongue, in the
    character's real style. Painting an interior behind that would be inventing over existing art."""
    from PIL import Image, ImageDraw

    d = tmp_path / "layers"
    d.mkdir()
    face = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    ImageDraw.Draw(face).rectangle([30, 20, 98, 110], fill=(250, 220, 205, 255))
    face.save(d / "00_face_base.png")
    # a mouth that is a *shape*, not a stroke — 20 wide x 16 tall
    open_mouth = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    ImageDraw.Draw(open_mouth).ellipse([56, 80, 76, 96], fill=(120, 50, 55, 255))
    open_mouth.save(d / "20_mouth.png")

    stack = decompose.from_layer_dir(d)
    assert synthesize_mouth_cavity(stack) is None
    assert not stack.by_role(R.mouth_cavity)


def test_a_tall_but_light_closed_smile_still_gets_a_cavity(tmp_path):
    """A closed smile with lip shading has a tall solid box (aspect > the closed threshold) yet is
    plainly shut — no dark oral cavity, just lip/skin tones. Aspect alone wrongly called it 'open' and
    denied it an interior (5 of 8 test characters could not open their mouths). It's only open if the box
    is tall AND holds a genuine dark interior, so a light tall smile must still be synthesised."""
    from PIL import Image, ImageDraw

    d = tmp_path / "layers"
    d.mkdir()
    face = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    ImageDraw.Draw(face).rectangle([30, 20, 98, 110], fill=(250, 220, 205, 255))
    face.save(d / "00_face_base.png")
    # a tall mouth SHAPE (20x16, aspect 0.8 — well over the closed threshold) but LIGHT: a soft lip tone
    # barely darker than skin, no dark cavity. This is the closed-smile-with-shading the aspect gate hit.
    smile = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    ImageDraw.Draw(smile).ellipse([56, 80, 76, 96], fill=(235, 200, 195, 255))
    smile.save(d / "20_mouth.png")

    stack = decompose.from_layer_dir(d)
    layer = synthesize_mouth_cavity(stack)
    assert layer is not None                                # tall but light -> closed -> gets a cavity
    assert stack.by_role(R.mouth_cavity)


def test_faint_full_canvas_scatter_does_not_suppress_the_cavity(tmp_path):
    """A See-through mouth layer carries a near-transparent halo (alpha 8-63) dusted across the whole
    canvas. PIL's raw getbbox() — anything > 0 — then returned the ENTIRE canvas, so a thin closed lip
    line read as a full-canvas square (aspect 1.0), tripped the 'already drawn open' skip, and the
    cavity was never painted (the mouth could not open — seen on the silverdress character). The
    solid-mass bbox must ignore the sprinkle so a closed mouth still gets its interior."""
    from PIL import Image, ImageDraw

    d = tmp_path / "layers"
    d.mkdir()
    face = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    ImageDraw.Draw(face).rectangle([30, 20, 98, 110], fill=(250, 220, 205, 255))
    face.save(d / "00_face_base.png")

    lips = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    ImageDraw.Draw(lips).rectangle([56, 84, 76, 87], fill=(150, 70, 80, 255))  # a 20x3 closed stroke
    # dust a faint halo (alpha 40) into the far corners — the decomposer scatter that broke getbbox
    px = lips.load()
    for x, y in ((2, 2), (125, 2), (2, 125), (125, 125), (64, 3), (3, 64)):
        px[x, y] = (150, 70, 80, 40)
    assert lips.getbbox() == (2, 2, 126, 126)              # raw getbbox is fooled: near-full canvas
    lips.save(d / "20_mouth.png")

    stack = decompose.from_layer_dir(d)
    layer = synthesize_mouth_cavity(stack)
    assert layer is not None                                # the scatter must NOT suppress the cavity
    assert stack.by_role(R.mouth_cavity)


def test_mouth_cavity_joins_the_head_group_so_it_turns_with_the_head():
    """The synthesised cavity sits behind the lips and must turn with the head exactly as they do — the
    head turn is delivered by head-GROUP membership (the Live2D warp grid / nijilive group rotation),
    not per-part offsets. A stale local ``_HEAD_ROLES`` in the emitter once omitted ``mouth_cavity``, so
    the cavity was left out of the head group and drifted from the mouth on a turn (followed it only
    ~2% on yaw). Pin both the single source of truth and the end-to-end membership."""
    from image2live2d.backends.nijilive import puppet
    from image2live2d.core.structure.graph import HEAD_ROLES

    # the emitter's head-role set must not drift from the canonical one (it once did, silently)
    assert puppet._HEAD_ROLES == HEAD_ROLES
    assert R.mouth_cavity in puppet._HEAD_ROLES

    # end to end: a rig whose mouth is a bare stroke gets a synthesised cavity, and it lands in the head
    # group alongside the lips it hides behind
    from image2live2d.core import decompose
    from image2live2d.pipeline import rig_from_stack
    from image2live2d.samples import make_sample_layers

    import pathlib
    tmp = pathlib.Path(__import__("tempfile").mkdtemp())
    rig = rig_from_stack(decompose.from_layer_dir(make_sample_layers(tmp / "src")), name="c")
    cavity = [p for p in rig.parts if p.semantic_role is R.mouth_cavity]
    if not cavity:
        pytest.skip("sample produced no synthesised cavity")
    drawn = [(p, rig.mesh_for(p.id)) for p in rig.parts]
    head_ids = puppet.head_group_ids(drawn)
    mouth = next(p for p in rig.parts if p.semantic_role is R.mouth)
    assert mouth.id in head_ids                            # the lips turn with the head
    assert cavity[0].id in head_ids                        # ...and so must the cavity behind them
