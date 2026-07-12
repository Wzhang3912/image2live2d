"""Phase 4 (Route A) — open JSON emitters + gated .moc3 seam + bundle assembly.

The .moc3 binary is gated, so these verify the four open JSON files map the IRR correctly and that a
JSON-only bundle is well-formed (renderable once a .moc3 is supplied)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from image2live2d.backends.live2d import Live2DEmitter, write_moc3_from_template
from image2live2d.backends.live2d.cdi3 import cdi3
from image2live2d.backends.live2d.model3 import model3
from image2live2d.backends.live2d.motion3 import motion3
from image2live2d.backends.live2d.physics3 import physics3
from image2live2d.core.assemble import assemble_rig
from image2live2d.core.mesh import grid_mesh
from image2live2d.core.motion import generate_idle
from image2live2d.core.physics import generate_physics
from image2live2d.core.rig import author_rig, select_template
from image2live2d.core.types import Layer, LayerStack
from image2live2d.irr.schema import SemanticRole as R


def _rig(name="t"):
    parts = [("face_base", R.face_base, (0.2, 0.5, 0.8, 0.95)),
             ("eye_l", R.eye_l, (0.30, 0.70, 0.45, 0.78)),
             ("eye_r", R.eye_r, (0.55, 0.70, 0.70, 0.78)),
             ("mouth", R.mouth, (0.42, 0.55, 0.58, 0.63)),
             ("hair_front", R.hair_front, (0.2, 0.75, 0.8, 0.98)),
             ("torso", R.torso, (0.35, 0.20, 0.65, 0.55)),
             ("clothing", R.clothing, (0.30, 0.05, 0.70, 0.30))]
    layers, meshes = [], []
    for i, (pid, role, rect) in enumerate(parts):
        layers.append(Layer(id=pid, semantic_role=role, texture_path=Path(f"{pid}.png"),
                            draw_order=i * 10, width=64, height=64))
        meshes.append(grid_mesh(pid, rect, lambda u, v: 255, grid=2))
    stack = LayerStack(layers=layers, canvas_width=64, canvas_height=64)
    auth = author_rig(stack, meshes, select_template(stack))
    phys = generate_physics(stack, auth.parameters)
    anims = generate_idle(auth.parameters)
    return assemble_rig(name=name, source=None, stack=stack, meshes=meshes,
                        deformers=auth.deformers, parameters=auth.parameters, physics=phys,
                        animations=anims)


# --------------------------------------------------------------------------------------------------
# physics3
# --------------------------------------------------------------------------------------------------
def test_physics3_maps_rigs():
    rig = _rig()
    doc = physics3(rig)
    assert doc["Version"] == 3
    assert doc["Meta"]["PhysicsSettingCount"] == len(rig.physics)
    s = doc["PhysicsSettings"][0]
    assert s["Input"][0]["Source"]["Id"] in rig.parameter_ids()
    assert s["Output"][0]["Destination"]["Id"] in rig.parameter_ids()
    assert s["Output"][0]["VertexIndex"] == len(s["Vertices"])  # tip drives output
    # vertex count meta matches
    assert doc["Meta"]["VertexCount"] == sum(len(x["Vertices"]) for x in doc["PhysicsSettings"])


# --------------------------------------------------------------------------------------------------
# motion3
# --------------------------------------------------------------------------------------------------
def test_motion3_curves_and_counts():
    rig = _rig()
    anim = rig.animations[0]
    doc = motion3(anim)
    assert doc["Version"] == 3
    assert doc["Meta"]["CurveCount"] == len(anim.lanes) == len(doc["Curves"])
    assert doc["Meta"]["Duration"] == pytest.approx(anim.length / anim.fps)
    # each curve opens with an initial [t0, v0] point
    for curve, lane in zip(doc["Curves"], anim.lanes):
        assert curve["Target"] == "Parameter" and curve["Id"] == lane.param_id
        assert len(curve["Segments"]) >= 2
    # segment count == sum(keyframes-1)
    assert doc["Meta"]["TotalSegmentCount"] == sum(len(ln.keyframes) - 1 for ln in anim.lanes)


def test_motion3_cubic_emits_bezier_segments():
    rig = _rig()
    anim = rig.animations[0]
    doc = motion3(anim)
    breath = next(c for c in doc["Curves"] if c["Id"] == "ParamBreath")
    # ParamBreath is authored Cubic -> first segment after the initial point is a Bezier (id 1)
    assert breath["Segments"][2] == 1


# --------------------------------------------------------------------------------------------------
# cdi3
# --------------------------------------------------------------------------------------------------
def test_cdi3_names_groups_parts():
    rig = _rig()
    doc = cdi3(rig)
    assert doc["Version"] == 3
    assert len(doc["Parameters"]) == len(rig.parameters)
    assert len(doc["Parts"]) == len(rig.parts)
    by_id = {p["Id"]: p for p in doc["Parameters"]}
    assert by_id["ParamAngleX"]["Name"] == "Angle X"
    assert by_id["ParamAngleX"]["GroupId"] == "Head"
    group_ids = {g["Id"] for g in doc["ParameterGroups"]}
    assert {"Head", "Eyes", "Mouth"} <= group_ids


# --------------------------------------------------------------------------------------------------
# model3
# --------------------------------------------------------------------------------------------------
def test_model3_references_and_groups():
    rig = _rig()
    doc = model3(rig, moc="t.moc3", textures=["textures/a.png"],
                 physics="t.physics3.json", display_info="t.cdi3.json",
                 motions={"Idle": ["t.idle.motion3.json"]})
    refs = doc["FileReferences"]
    assert refs["Moc"] == "t.moc3" and refs["Physics"] == "t.physics3.json"
    assert refs["Motions"]["Idle"] == [{"File": "t.idle.motion3.json"}]
    names = {g["Name"] for g in doc["Groups"]}
    assert "EyeBlink" in names and "LipSync" in names
    eye = next(g for g in doc["Groups"] if g["Name"] == "EyeBlink")
    assert set(eye["Ids"]) == {"ParamEyeLOpen", "ParamEyeROpen"}
    assert any(h["Name"] == "Head" for h in doc["HitAreas"])


# --------------------------------------------------------------------------------------------------
# .moc3 seam (gated)
# --------------------------------------------------------------------------------------------------
def test_moc3_seam_is_gated():
    with pytest.raises(NotImplementedError):
        write_moc3_from_template(_rig(), None)


def test_moc3_seam_uses_injected_writer():
    rig = _rig()
    data = write_moc3_from_template(rig, "template.moc3", writer=lambda r, t: b"MOC3FAKE")
    assert data == b"MOC3FAKE"


# --------------------------------------------------------------------------------------------------
# bundle assembly (JSON-only, no moc writer)
# --------------------------------------------------------------------------------------------------
def test_emitter_writes_json_only_bundle(tmp_path):
    rig = _rig("hero")
    bundle = Live2DEmitter().build(rig, tmp_path)
    assert bundle.moc_written is False
    assert bundle.model3_path.name == "hero.model3.json"
    # the four open JSON files exist and parse
    for rel in ("hero.model3.json", "hero.physics3.json", "hero.cdi3.json", "hero.idle.motion3.json"):
        p = tmp_path / rel
        assert p.is_file()
        json.loads(p.read_text())
    # textures written
    assert list((tmp_path / "textures").glob("*.png"))
    # model3 references resolve to real files (except the gated .moc3)
    m = json.loads(bundle.model3_path.read_text())
    refs = m["FileReferences"]
    assert (tmp_path / refs["Physics"]).is_file()
    for tex in refs["Textures"]:
        assert (tmp_path / tex).is_file()


def test_emitter_writes_moc3_with_writer(tmp_path):
    rig = _rig("hero")
    em = Live2DEmitter(moc_writer=lambda r, t: b"MOC3FAKE", moc_template="tmpl.moc3")
    bundle = em.build(rig, tmp_path)
    assert bundle.moc_written is True
    assert (tmp_path / "hero.moc3").read_bytes() == b"MOC3FAKE"
