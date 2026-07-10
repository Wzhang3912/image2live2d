"""Phase 5 calibration — split See-through's combined (non-L/R) facial layers into _l / _r."""

from __future__ import annotations

import pytest

from image2live2d.irr.schema import SemanticRole as R


def _two_eye_image(size=256):
    """An image with two horizontally-separated alpha blobs (a combined 'both eyes' layer)."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([40, 110, 100, 150], fill=(20, 20, 30, 255))    # left eye
    d.ellipse([156, 110, 216, 150], fill=(20, 20, 30, 255))   # right eye
    return img


def test_split_lr_separates_two_blobs():
    pytest.importorskip("PIL")
    from image2live2d.core.decompose import _split_lr

    left, right = _split_lr(_two_eye_image())
    # left image keeps only the left blob (alpha present on the left, gone on the right) and vice versa
    assert left.getchannel("A").getbbox()[2] <= 128       # left blob's right edge is in the left half
    assert right.getchannel("A").getbbox()[0] >= 128      # right blob's left edge is in the right half


def test_split_lr_separates_thin_lashes_on_tall_canvas():
    """Regression: a thin combined eyelash on a hi-res (tall) canvas must still split L/R.

    The original _split_lr subsampled every ``h//200``-th row "for speed"; on a 1280px-tall
    See-through layer that's every 6th row, which fragmented thin eyelashes into >2 spurious column
    runs so they never split — char2/char3 lost their right eye (no eye_r). Full-resolution column
    detection fixes it. This builds the failure case: two ~4px-tall lash strokes on a 1280 canvas."""
    pytest.importorskip("PIL")
    from PIL import Image, ImageDraw
    from image2live2d.core.decompose import _split_lr

    img = Image.new("RGBA", (1280, 1280), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([592, 600, 625, 604], fill=(20, 20, 30, 255))   # left lash (4px tall)
    d.rectangle([657, 600, 690, 604], fill=(20, 20, 30, 255))   # right lash, clean gap between
    split = _split_lr(img)
    assert split is not None                                    # not fragmented away by subsampling
    left, right = split
    assert left.getchannel("A").getbbox()[2] <= 641            # left stroke stays left of the gap
    assert right.getchannel("A").getbbox()[0] >= 641           # right stroke stays right of the gap


def test_split_lr_returns_none_for_single_blob():
    pytest.importorskip("PIL")
    from PIL import Image, ImageDraw
    from image2live2d.core.decompose import _split_lr

    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse([100, 100, 156, 156], fill=(0, 0, 0, 255))  # one blob
    assert _split_lr(img) is None


def test_raws_to_stack_splits_combined_eye(tmp_path):
    pytest.importorskip("PIL")
    from image2live2d.core.decompose import RawLayer, raws_to_stack

    raws = [
        RawLayer(name="face", image=_solid(256), offset=(0, 0)),
        RawLayer(name="eyelash", image=_two_eye_image(256), offset=(0, 0)),  # combined eyes
    ]
    stack = raws_to_stack(raws, (256, 256), tmp_path)
    roles = {layer.semantic_role for layer in stack.layers}
    assert R.eye_l in roles and R.eye_r in roles          # combined -> both sides
    assert R.face_base in roles


def _solid(size):
    from PIL import Image
    return Image.new("RGBA", (size, size), (200, 180, 160, 255))
