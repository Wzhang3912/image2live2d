"""Phase 5 — Tier-2 seams: ingest.load_image + preprocess.prepare."""

from __future__ import annotations

import pytest

from image2live2d.core import ingest, preprocess
from image2live2d.core.types import ImageAsset


def _sample_png(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))  # transparent margins
    ImageDraw.Draw(img).ellipse((60, 50, 140, 150), fill=(200, 120, 90, 255))
    p = tmp_path / "src.png"
    img.save(p)
    return p


def test_load_image_from_file(tmp_path):
    p = _sample_png(tmp_path)
    asset = ingest.load_image(p)
    assert isinstance(asset, ImageAsset)
    assert asset.width == 200 and asset.height == 200
    assert asset.path == p


def test_load_image_from_bytes(tmp_path):
    p = _sample_png(tmp_path)
    asset = ingest.load_image(p.read_bytes(), work_dir=tmp_path / "w")
    assert asset.width == 200 and asset.height == 200
    assert asset.path.is_file()


def test_load_image_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest.load_image(tmp_path / "nope.png")


def test_generate_image_is_gated():
    with pytest.raises(NotImplementedError):
        ingest.generate_image("a catgirl", out_dir=None)


def test_prepare_crops_to_alpha_without_bg_removal(tmp_path):
    p = _sample_png(tmp_path)
    asset = ingest.load_image(p)
    prepared = preprocess.prepare(asset, work_dir=tmp_path / "prep", remove_bg=False)
    # cropped to the ellipse bbox -> smaller than the 200x200 source
    assert prepared.width < 200 and prepared.height < 200
    assert prepared.path.is_file()
    assert prepared.alpha_path is not None and prepared.alpha_path.is_file()
