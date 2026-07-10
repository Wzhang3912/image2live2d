"""Tests for the IRR schema, validators, and example fixture."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from image2live2d.irr import Rig, lint, make_parameter, standard_parameters
from image2live2d.irr.example import build_example_rig
from image2live2d.irr.schema import Keyform, Mesh, Meta, Parameter, Part, SemanticRole, Texture


def test_example_rig_is_valid():
    rig = build_example_rig()
    assert rig.meta.name == "example"
    assert rig.part_ids() == {"face_base", "mouth"}
    assert rig.mesh_for("mouth") is not None


def test_example_rig_json_roundtrip():
    rig = build_example_rig()
    restored = Rig.model_validate_json(rig.model_dump_json())
    assert restored == rig


def test_standard_parameters_have_valid_ranges():
    params = standard_parameters()
    assert any(p.id == "ParamMouthOpenY" for p in params)
    for p in params:
        assert p.min <= p.default <= p.max


def test_make_parameter_unknown_id_raises():
    with pytest.raises(KeyError):
        make_parameter("ParamNotReal")


def test_bad_triangle_index_rejected():
    with pytest.raises(ValidationError):
        Mesh(part_id="x", vertices=[(0, 0), (1, 0), (1, 1)], uvs=[(0, 0), (1, 0), (1, 1)], triangles=[(0, 1, 9)])


def test_missing_texture_reference_rejected():
    with pytest.raises(ValidationError):
        Rig(
            meta=Meta(name="bad"),
            parts=[Part(id="p", semantic_role=SemanticRole.mouth, texture_id="nope", draw_order=0)],
        )


def test_keyform_offset_length_must_match_mesh():
    tex = Texture(id="t", path="t.png", width=8, height=8)
    part = Part(id="mouth", semantic_role=SemanticRole.mouth, texture_id="t", draw_order=0)
    m = Mesh(
        part_id="mouth",
        vertices=[(0, 0), (1, 0), (1, 1), (0, 1)],
        uvs=[(0, 0), (1, 0), (1, 1), (0, 1)],
        triangles=[(0, 1, 2), (0, 2, 3)],
    )
    bad = Parameter(
        id="ParamMouthOpenY",
        min=0,
        max=1,
        keyforms=[Keyform(value=0.0, mesh_offsets={"mouth": [(0, 0)]})],  # 1 != 4
    )
    with pytest.raises(ValidationError):
        Rig(meta=Meta(name="x"), textures=[tex], parts=[part], meshes=[m], parameters=[bad])


def test_physics_references_must_exist():
    from image2live2d.irr.schema import PhysicsRig

    with pytest.raises(ValidationError):
        Rig(
            meta=Meta(name="x"),
            physics=[PhysicsRig(id="ph", driver_param="ParamAngleX", output_param="ParamHairFront")],
        )


def test_example_rig_lint_reports_expected_gaps():
    # The minimal example lacks eyes and most movement params -> lint should warn (but not raise).
    issues = lint(build_example_rig())
    codes = {i.code for i in issues}
    assert "missing_role" in codes
