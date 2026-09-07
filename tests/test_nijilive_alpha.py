"""Issue #95: INP textures must match nijilive's premultiplied-alpha renderer."""

from io import BytesIO

import pytest

from image2live2d.backends.live2d import Live2DEmitter
from image2live2d.backends.nijilive import NijiliveEmitter
from image2live2d.backends.nijilive.inp import InpFile
from image2live2d.irr.example import build_example_rig

Image = pytest.importorskip("PIL.Image")


def _emit_texture(tmp_path, source):
    rig = build_example_rig()
    texture = rig.textures[0]
    texture.width, texture.height = source.size
    path = tmp_path / texture.path
    path.parent.mkdir(parents=True)
    source.save(path, format="PNG")
    original = path.read_bytes()
    emitter = NijiliveEmitter(asset_root=tmp_path)
    out = emitter.emit(rig, tmp_path / "niji")
    first = out.read_bytes()
    # Re-exporting the same rig must not multiply alpha a second time or mutate the art.
    assert emitter.emit(rig, out.parent).read_bytes() == first
    assert path.read_bytes() == original
    bundle = Live2DEmitter(asset_root=tmp_path).build(rig, tmp_path / "live2d")
    assert (bundle.model3_path.parent / "textures/000_tex_face.png").read_bytes() == original
    with Image.open(BytesIO(InpFile.read(out).textures[0].data)) as embedded:
        return embedded.convert("RGBA")


@pytest.mark.parametrize("mode", ["RGBA", "LA", "P", "RGB"])
def test_embedded_texture_pixels_follow_alpha_convention(tmp_path, mode):
    if mode == "RGBA":
        source = Image.new(mode, (4, 1))
        source.putdata([(255, 255, 255, 30), (200, 100, 50, 128),
                        (255, 80, 40, 0), (17, 99, 231, 255)])
    elif mode == "LA":
        source = Image.new(mode, (4, 1))
        source.putdata([(255, 30), (100, 128), (80, 0), (17, 255)])
    elif mode == "P":
        source = Image.new(mode, (4, 1))
        source.putpalette([255, 255, 255, 200, 100, 50, 255, 80, 40, 17, 99, 231])
        source.putdata([0, 1, 2, 3])
        source.info["transparency"] = bytes([30, 128, 0, 255])
    else:
        source = Image.new(mode, (4, 1), (17, 99, 231))

    embedded = _emit_texture(tmp_path, source)
    assert embedded.size == source.size
    rgba = source.convert("RGBA")
    for x in range(source.width):
        actual, original = embedded.getpixel((x, 0)), rgba.getpixel((x, 0))
        *rgb, alpha = original
        assert actual[3] == alpha
        assert actual[:3] == tuple(round(channel * alpha / 255) for channel in rgb)


@pytest.mark.parametrize("background", [(0, 0, 0), (40, 70, 110), (255, 255, 255)])
def test_translucent_edge_blends_like_source_art(tmp_path, background):
    source = Image.new("RGBA", (3, 1))
    source.putdata([(255, 255, 255, 30), (200, 100, 50, 128), (255, 255, 255, 0)])
    embedded = _emit_texture(tmp_path, source)
    for x in range(source.width):
        actual, original = embedded.getpixel((x, 0)), source.getpixel((x, 0))
        alpha = actual[3] / 255
        # nijilive Normal: GL_ONE, GL_ONE_MINUS_SRC_ALPHA. Compare before framebuffer
        # clamping so overbright white-on-white pixels cannot hide the mismatch.
        rendered = [actual[c] + background[c] * (1 - alpha) for c in range(3)]
        expected = [original[c] * alpha + background[c] * (1 - alpha) for c in range(3)]
        assert rendered == pytest.approx(expected, abs=0.5)
