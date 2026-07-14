"""Sample-layer generator + CLI tests (Pillow-gated, since both draw/read PNGs)."""

from __future__ import annotations

import importlib.util
import json

import pytest

from image2live2d.backends.nijilive.inp import InpFile

_HAS_PIL = importlib.util.find_spec("PIL") is not None
pytestmark = pytest.mark.skipif(not _HAS_PIL, reason="Pillow not installed")


def test_make_sample_layers_writes_named_pngs(tmp_path):
    from image2live2d.samples import make_sample_layers

    out = make_sample_layers(tmp_path / "layers", size=256)
    names = sorted(p.name for p in out.glob("*.png"))
    assert "00_face_base.png" in names
    assert "70_mouth.png" in names
    assert "90_hair_front.png" in names
    assert len(names) == 12


def test_sample_layers_drive_the_spine(tmp_path):
    from image2live2d.core import decompose
    from image2live2d.pipeline import rig_from_stack
    from image2live2d.samples import make_sample_layers

    layers = make_sample_layers(tmp_path / "layers", size=256)
    stack = decompose.from_layer_dir(layers)
    rig = rig_from_stack(stack, name="sample")
    # all 12 parts present (+1 synthesised mouth cavity), blink + mouth + head-turn authored
    assert len(rig.parts) == 13
    assert {"ParamEyeLOpen", "ParamMouthOpenY", "ParamAngleX"} <= rig.parameter_ids()
    # meshes are non-trivial (tightened grids actually clipped to the drawn art)
    assert all(len(m.vertices) >= 3 for m in rig.meshes)


def test_fullbody_sample_drives_body_and_physics(tmp_path):
    from image2live2d.core import decompose
    from image2live2d.pipeline import rig_from_stack
    from image2live2d.samples import make_sample_fullbody

    layers = make_sample_fullbody(tmp_path / "fb", size=256)
    roles = {p.stem.split("_", 1)[1] for p in layers.glob("*.png")}
    assert {"hair_back", "torso", "leg_l", "leg_r", "arm_l"} <= roles

    rig = rig_from_stack(decompose.from_layer_dir(layers), name="fb")
    assert {"ParamBodyAngleX", "ParamBodyAngleZ", "ParamHairBack"} <= rig.parameter_ids()
    assert any(p.output_param == "ParamHairBack" for p in rig.physics)  # hair physics wired


def test_cli_sample_emits_loadable_inp(tmp_path):
    from image2live2d.__main__ import main

    out = tmp_path / "sample.inp"
    rc = main(["--sample", str(tmp_path / "layers"), "-o", str(out)])
    assert rc == 0
    assert out.exists()

    inp = InpFile.read(out)
    puppet = json.loads(inp.payload)
    assert len(inp.textures) == 13          # 12 decomposed + the synthesised mouth cavity
    assert puppet["meta"]["name"] == "sample"
    assert {"ParamMouthOpenY", "ParamEyeLOpen"} <= {p["name"] for p in puppet["param"]}
