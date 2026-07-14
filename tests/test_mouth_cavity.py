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
