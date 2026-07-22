"""Layer speck denoise (RIVAL_HARVEST_BACKLOG T8).

See-through flings small opaque blobs to the far corners of every layer. The bbox defence in
``mesh.build.alpha_bbox`` hides them from the finished mesh, but the L/R splitters read the layer's
alpha *structure* — "exactly two blobs" — so one stray speck column made a two-eyed layer look
three-blobbed and the character shipped with a single eye (10 of 32 combined facial layers across the
8 real characters). These pin the rule that separates scatter from a layer's own small-but-real art.
"""

from __future__ import annotations

import pytest

from image2live2d.core.decompose.denoise import components, drop_specks, speck_labels

# denoise imports numpy/Pillow lazily, so the module above loads even on a light CI install.
np = pytest.importorskip("numpy")
pytest.importorskip("PIL")


def _blank(size=256):
    from PIL import Image

    return Image.new("RGBA", (size, size), (0, 0, 0, 0))


def _mask(*boxes, size=64):
    """A bool mask with a filled rectangle per ``(x0, y0, x1, y1)`` (inclusive)."""
    m = np.zeros((size, size), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        m[y0:y1 + 1, x0:x1 + 1] = True
    return m


def test_components_labels_separated_blobs():
    labels, areas = components(_mask((2, 2, 5, 5), (40, 40, 43, 43)))
    assert len(areas) == 2
    assert sorted(areas.values()) == [16, 16]
    assert labels[3, 3] != labels[41, 41] and labels[0, 0] == 0


def test_components_are_eight_connected():
    """A thin anti-aliased diagonal is one stroke. Under 4-connectivity it shatters into pixels, and
    every fragment would then read as a speck beside the layer's main blob."""
    m = np.zeros((16, 16), dtype=bool)
    for i in range(10):
        m[i, i] = True                      # a pure diagonal: 4-connectivity sees 10 components
    _, areas = components(m)
    assert len(areas) == 1 and next(iter(areas.values())) == 10


def test_far_speck_is_dropped_even_when_it_is_not_tiny():
    """The measured killer: a ``lavendergown`` eyebrow's scatter is 15% of a brow — far too big for a
    size threshold to catch, but it sits 11 hull-spans away from the brows."""
    labels, areas = components(_mask((20, 30, 29, 33), (34, 30, 43, 33), (60, 2, 62, 4)))
    dropped = speck_labels(labels, areas)
    assert len(dropped) == 1
    assert labels[3, 61] in dropped                     # the corner blob, and only it


def test_small_secondary_art_on_top_of_the_content_is_kept():
    """Every real eyelash layer carries a mirrored pair of lower lashes at 8-12% of the main lash.
    They overlap the size band of the scatter, so only their position saves them."""
    labels, areas = components(
        _mask((20, 30, 29, 33), (34, 30, 43, 33), (21, 36, 24, 36), (39, 36, 42, 36)))
    assert speck_labels(labels, areas) == []


def test_core_components_are_never_dropped_however_far_apart():
    # Two twintails at opposite edges are content, not noise — both are within 4x of each other.
    labels, areas = components(_mask((1, 1, 10, 40), (53, 1, 62, 40)))
    assert speck_labels(labels, areas) == []


def test_tiny_speck_inside_the_content_is_still_dropped():
    labels, areas = components(_mask((10, 10, 40, 40), (45, 20, 45, 20)))
    assert len(speck_labels(labels, areas)) == 1


def test_drop_specks_leaves_a_clean_layer_untouched():
    from PIL import ImageDraw

    img = _blank()
    ImageDraw.Draw(img).ellipse([40, 110, 100, 150], fill=(20, 20, 30, 255))
    assert drop_specks(img) is img          # single component -> returned unchanged, no copy


def test_a_corner_speck_costs_the_character_its_second_eye_until_denoised():
    """The end-to-end regression, at the geometry it was actually measured at: ``lavendergown``'s
    eyelash layer on a 1280 canvas — two lashes at x 569-617 / 649-696, plus a 9px-wide speck column
    out at x 1265. ``_split_lr`` refuses that (three column runs, not two), so only ``eye_l`` is
    authored and the character ships one-eyed.

    The speck is 18% of a lash, so no size threshold reaches it; it is dropped for being 4.5 hull-spans
    away. The proportions matter — on a small canvas the same corner is under one span out — which is
    why this test is built full-size rather than scaled down."""
    from PIL import Image, ImageDraw

    from image2live2d.core.decompose.sources import _split_lr

    img = Image.new("RGBA", (1280, 1280), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([569, 600, 617, 604], fill=(20, 20, 30, 255))   # left lash
    d.rectangle([649, 600, 696, 604], fill=(20, 20, 30, 255))   # right lash
    assert _split_lr(img) is not None, "two clean lashes must split"

    d.rectangle([1265, 100, 1273, 104], fill=(20, 20, 30, 255))  # the scatter speck
    assert _split_lr(img) is None, "the speck must break the split (this is the bug)"
    assert _split_lr(drop_specks(img)) is not None, "denoising must restore the split"
