"""Synthesised closed-eye lash line (core.synth.eye + the ParamEye*Open crossfade).

A decomposed eye is white + pupil + a dark lash-line lineart; there is no *closed* pose, so the only way
to shut it was to squash the open parts, leaving a compressed sliver of iris. These pin that we paint a
lash line from the eye's own colour, that it is invisible while the eye is open, that opening the eye
crossfades it out while the open parts fade in, and that a rig without eyes is left alone.
"""

from __future__ import annotations

import pytest

from image2live2d.core import decompose, mesh
from image2live2d.core.rig import author_rig, select_template
from image2live2d.core.synth import synthesize_closed_eyes
from image2live2d.irr.schema import SemanticRole as R

pytest.importorskip("PIL")


def _layers(tmp_path, *, with_eyes=True):
    """A minimal face: a skin block, and (optionally) a dark lash-line stroke + white + pupil per eye —
    the shapes a decomposer returns for an open eye."""
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
            lash = Image.new("RGBA", (128, 128), (0, 0, 0, 0))          # dark lash-line lineart
            ImageDraw.Draw(lash).arc([cx - 9, 42, cx + 9, 58], 200, 340, fill=(30, 18, 20, 255), width=2)
            lash.save(d / f"52_{name}.png")
    return d


def test_a_lash_line_is_painted_per_eye(tmp_path):
    stack = decompose.from_layer_dir(_layers(tmp_path))
    made = synthesize_closed_eyes(stack)

    roles = {ly.semantic_role for ly in made}
    assert roles == {R.eye_closed_l, R.eye_closed_r}
    for ly in made:
        assert ly.texture_path.is_file()
    # each lash line sits just *above* its eye lineart, so it shows on top when the eye shuts
    ids = [ly.id for ly in stack.layers]
    assert ids.index("52_eye_closed_l") > ids.index("52_eye_l")


def test_lash_colour_comes_from_the_eye_not_a_hardcoded_black(tmp_path):
    """The lash line is struck from the eye lineart's own dark pixels, so it matches the character."""
    from PIL import Image
    import numpy as np

    stack = decompose.from_layer_dir(_layers(tmp_path))
    made = synthesize_closed_eyes(stack)
    arc = np.asarray(Image.open(made[0].texture_path).convert("RGBA"))
    solid = arc[..., 3] > 128
    mean = arc[..., :3][solid].mean(axis=0)
    # our stub lash is (30,18,20): a dark, faintly warm tone — not pure black, not the white sclera
    assert mean[0] > mean[2] and mean.mean() < 80


def test_opening_the_eye_crossfades_lash_out_and_eye_in(tmp_path):
    """ParamEyeLOpen 1 (open) -> lash hidden, open parts shown; 0 (closed) -> the reverse. The fade is
    carried by opacity_overrides (see backends.live2d.moc3_emit)."""
    stack = decompose.from_layer_dir(_layers(tmp_path))
    synthesize_closed_eyes(stack)
    meshes = mesh.build_meshes(stack)
    params = author_rig(stack, meshes, select_template(stack)).parameters

    p = next(p for p in params if p.id == "ParamEyeLOpen")
    shut = next(k for k in p.keyforms if k.value == 0.0)
    opened = next(k for k in p.keyforms if k.value == 1.0)

    assert shut.opacity_overrides["52_eye_closed_l"] == 1.0   # lash visible when shut
    assert opened.opacity_overrides["52_eye_closed_l"] == 0.0  # ...gone when open
    assert shut.opacity_overrides["52_eye_l"] == 0.0          # open lineart gone when shut
    assert opened.opacity_overrides["52_eye_l"] == 1.0        # ...back when open


def test_no_eyes_means_no_lash_line(tmp_path):
    stack = decompose.from_layer_dir(_layers(tmp_path, with_eyes=False))
    assert synthesize_closed_eyes(stack) == []
    assert not stack.by_role(R.eye_closed_l) and not stack.by_role(R.eye_closed_r)


def test_synthesis_is_idempotent(tmp_path):
    stack = decompose.from_layer_dir(_layers(tmp_path))
    assert len(synthesize_closed_eyes(stack)) == 2
    assert synthesize_closed_eyes(stack) == []                # already has them; don't stack lash lines
    assert len(stack.by_role(R.eye_closed_l)) == 1
