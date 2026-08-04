"""Synthesised cheek blush (core.synth.blush + the ParamCheek opacity fade).

See-through never emits a blush layer, so we paint one: two soft ovals on the cheeks, tinted from the
character's own skin toward pink. These pin that it is painted below/outside the eyes, that its colour is
skin-derived (not a foreign hue), that it is INVISIBLE at rest (opacity 0 until ParamCheek is driven), and
that a face without eyes is left alone.
"""

from __future__ import annotations

import pytest

from image2live2d.core import decompose, mesh
from image2live2d.core.rig import author_rig, select_template
from image2live2d.core.synth import synthesize_blush
from image2live2d.irr.schema import SemanticRole as R

pytest.importorskip("PIL")


def _layers(tmp_path, *, with_eyes=True):
    """A minimal face: a skin block, and (optionally) a white + lash-line stroke per eye."""
    from PIL import Image, ImageDraw

    d = tmp_path / "layers"
    d.mkdir()
    face = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    ImageDraw.Draw(face).rectangle([24, 16, 104, 118], fill=(250, 220, 205, 255))
    face.save(d / "00_face_base.png")
    if with_eyes:
        for name, cx in (("eye_l", 48), ("eye_r", 80)):
            white = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
            ImageDraw.Draw(white).ellipse([cx - 9, 44, cx + 9, 58], fill=(255, 255, 255, 255))
            white.save(d / f"50_{name.replace('eye', 'eye_white')}.png")
            lash = Image.new("RGBA", (128, 128), (0, 0, 0, 0))       # lash lineart around the whole eye
            ImageDraw.Draw(lash).ellipse([cx - 9, 42, cx + 9, 58], outline=(30, 18, 20, 255), width=2)
            lash.save(d / f"52_{name}.png")
    return d


def _solid(img):
    import numpy as np
    a = np.asarray(img)
    return a, a[..., 3] > 64


def test_a_blush_is_painted_below_the_eyes(tmp_path):
    from PIL import Image
    import numpy as np

    stack = decompose.from_layer_dir(_layers(tmp_path))
    layer = synthesize_blush(stack)
    assert layer is not None and layer.semantic_role is R.blush
    assert layer.texture_path.is_file()
    # inserted just above the skin
    ids = [ly.id for ly in stack.layers]
    assert ids.index(layer.id) > ids.index("00_face_base")
    # the painted mass sits BELOW the eyes (image y is down; eyes span y~42-58) and in two lobes L/R
    a, solid = _solid(Image.open(layer.texture_path).convert("RGBA"))
    ys, xs = np.where(solid)
    assert ys.min() > 58                                   # entirely below the eye boxes
    assert xs.min() < 64 < xs.max()                        # a lobe on each side of the midline


def test_blush_colour_is_skin_tinted_pink_not_a_foreign_hue(tmp_path):
    from PIL import Image

    stack = decompose.from_layer_dir(_layers(tmp_path))
    layer = synthesize_blush(stack)
    a, solid = _solid(Image.open(layer.texture_path).convert("RGBA"))
    mean = a[..., :3][solid].mean(axis=0)
    assert mean[0] > mean[1] and mean[0] > mean[2]         # red-leaning: a blush, not grey/blue
    assert mean[1] < 220 and mean[2] < 220                 # pinker than the (250,220,205) bare skin


def test_blush_is_hidden_at_rest_and_fades_in_on_paramcheek(tmp_path):
    stack = decompose.from_layer_dir(_layers(tmp_path))
    synthesize_blush(stack)
    meshes = mesh.build_meshes(stack)
    params = author_rig(stack, meshes, select_template(stack)).parameters
    p = next(p for p in params if p.id == "ParamCheek")
    rest = next(k for k in p.keyforms if k.value == 0.0)
    on = next(k for k in p.keyforms if k.value == 1.0)
    blush_id = stack.by_role(R.blush)[0].id
    assert rest.opacity_overrides[blush_id] == 0.0         # invisible at rest -> resting face unchanged
    assert on.opacity_overrides[blush_id] == 1.0           # fades fully in when driven
    # opacity-only: the blush does not deform
    assert all(dx == dy == 0.0 for dx, dy in on.mesh_offsets[blush_id])


def test_no_eyes_means_no_blush(tmp_path):
    stack = decompose.from_layer_dir(_layers(tmp_path, with_eyes=False))
    assert synthesize_blush(stack) is None
    assert not stack.by_role(R.blush)


def test_blush_synthesis_is_idempotent(tmp_path):
    stack = decompose.from_layer_dir(_layers(tmp_path))
    assert synthesize_blush(stack) is not None
    assert synthesize_blush(stack) is None                 # already has one; don't stack blushes
    assert len(stack.by_role(R.blush)) == 1
