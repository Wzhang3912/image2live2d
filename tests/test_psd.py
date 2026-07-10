"""PSD adapter tests: pure name->role mapping (no deps) + raws_to_stack end-to-end (Pillow-gated)."""

from __future__ import annotations

import importlib.util

import pytest

from image2live2d.core.decompose import RawLayer, raws_to_stack, role_from_layer_name
from image2live2d.irr.schema import SemanticRole as R

_HAS_PIL = importlib.util.find_spec("PIL") is not None
_HAS_PSD = importlib.util.find_spec("psd_tools") is not None


@pytest.mark.parametrize(
    "name,expected",
    [
        ("face", R.face_base), ("Skin", R.face_base), ("head_base", R.face_base),
        ("Hair Front", R.hair_front), ("hair_back", R.hair_back), ("back hair", R.hair_back),
        ("Side Hair", R.hair_side), ("bangs", R.hair_front), ("ahoge", R.hair_front),
        ("eyebrow_left", R.eyebrow_l), ("Right Brow", R.eyebrow_r),
        ("left eye", R.eye_l), ("eye_r", R.eye_r), ("Eyelash L", R.eye_l),
        ("eye white left", R.eye_white_l), ("sclera_r", R.eye_white_r),
        ("pupil_left", R.pupil_l), ("iris right", R.pupil_r),
        ("nose", R.nose), ("mouth", R.mouth), ("Lips", R.mouth),
        ("left ear", R.ear_l), ("blush", R.blush), ("cheek", R.blush),
        ("torso", R.torso), ("body", R.torso),
        ("left arm", R.arm_l), ("right hand", R.hand_r), ("left leg", R.leg_l),
        ("dress", R.clothing), ("ribbon", R.clothing), ("hat", R.accessory),
        ("background", R.background), ("bg", R.background),
        ("eye_white_l", R.eye_white_l),  # exact value wins
        ("mysterywidget", R.other),
    ],
)
def test_role_from_layer_name(name, expected):
    assert role_from_layer_name(name) is expected


def test_ambiguous_side_defaults_left():
    assert role_from_layer_name("eye") is R.eye_l


def test_custom_default():
    assert role_from_layer_name("???", default=R.accessory) is R.accessory


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not installed")
def test_raws_to_stack_builds_pipeline_ready_stack(tmp_path):
    from PIL import Image
    from image2live2d.pipeline import rig_from_stack

    canvas = (64, 64)

    def blob(box):
        img = Image.new("RGBA", canvas, (0, 0, 0, 0))
        for x in range(box[0], box[2]):
            for y in range(box[1], box[3]):
                img.putpixel((x, y), (200, 120, 120, 255))
        # crop to extent + offset, like a PSD layer
        crop = img.crop(box)
        return crop, (box[0], box[1])

    # bottom -> top order
    specs = [("face skin", (10, 10, 54, 54)), ("left eye", (20, 24, 30, 32)),
             ("right eye", (40, 24, 50, 32)), ("mouth", (28, 40, 44, 48))]
    raws = [RawLayer(name=n, image=img, offset=off) for n, (img, off) in
            ((n, blob(b)) for n, b in specs)]

    stack = raws_to_stack(raws, canvas, tmp_path / "layers")
    assert [l.semantic_role for l in stack.layers] == [R.face_base, R.eye_l, R.eye_r, R.mouth]
    assert [l.draw_order for l in stack.layers] == [0, 1, 2, 3]
    assert stack.canvas_width == 64

    # the extracted stack drives the rest of the spine unchanged
    rig = rig_from_stack(stack, name="frompsd")
    assert {"ParamMouthOpenY", "ParamEyeLOpen", "ParamEyeROpen"} <= rig.parameter_ids()
    assert len(rig.parts) == 4


@pytest.mark.skipif(not (_HAS_PIL and _HAS_PSD), reason="needs Pillow + psd-tools")
def test_from_psd_end_to_end(tmp_path):
    from PIL import Image
    from psd_tools import PSDImage
    from psd_tools.api.layers import PixelLayer

    from image2live2d.core.decompose import from_psd
    from image2live2d.pipeline import rig_from_stack

    psd = PSDImage.new(mode="RGBA", size=(64, 64))

    def add(name, box):
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        for x in range(box[0], box[2]):
            for y in range(box[1], box[3]):
                img.putpixel((x, y), (200, 120, 120, 255))
        psd.append(PixelLayer.frompil(img, psd, name))

    add("face skin", (8, 8, 56, 56))  # appended first -> bottom -> draw_order 0
    add("left eye", (18, 22, 28, 30))
    add("right eye", (36, 22, 46, 30))
    add("mouth", (26, 40, 40, 46))
    psd_path = tmp_path / "char.psd"
    psd.save(psd_path)

    stack = from_psd(psd_path, tmp_path / "layers")
    assert stack.layers[0].semantic_role is R.face_base  # bottom layer first
    assert {R.eye_l, R.eye_r, R.mouth} <= {l.semantic_role for l in stack.layers}
    rig = rig_from_stack(stack, name="char")
    assert "ParamMouthOpenY" in rig.parameter_ids()
    assert len(rig.parts) == 4
