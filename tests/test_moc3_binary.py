"""Route A / Phase 4B S1: the low-level .moc3 binary codec round-trips.

Proves we can both READ and WRITE Live2D's closed .moc3 format (v3.00) — the foundation for generating
a Live2D model from the IRR. A self-contained synthetic model always runs; a real official sample
(Haru, if downloaded to out/moc3_research/haru.moc3) is round-tripped when present."""

from __future__ import annotations

from pathlib import Path

from image2live2d.backends.live2d.moc3_binary import (
    COUNT_KEYS, FIELDS, Moc3, read_moc3, write_moc3,
)


def _empty_moc() -> Moc3:
    counts = {k: 0 for k in COUNT_KEYS}
    sections: dict = {}
    for section, field, kind, _ in FIELDS:
        sections.setdefault(section, {})[field] = b"" if kind == "rt" else []
    canvas = {"pixelsPerUnit": 100.0, "originX": 0.0, "originY": 0.0,
              "width": 2.0, "height": 2.0, "flags": 0}
    return Moc3(canvas=canvas, counts=counts, sections=sections)


def test_roundtrip_empty_model():
    m = _empty_moc()
    out = write_moc3(m)
    assert out[:4] == b"MOC3" and out[4] == 1
    back = read_moc3(out)
    assert back.counts == m.counts
    assert abs(back.canvas["width"] - 2.0) < 1e-6


def test_roundtrip_minimal_parameter_and_mesh():
    m = _empty_moc()
    # one parameter
    m.counts["parameters"] = 1
    m.sections["parameters"] = {
        "runtimeSpace0": b"\0" * 8, "ids": ["ParamAngleX"],
        "maxValues": [30.0], "minValues": [-30.0], "defaultValues": [0.0],
        "isRepeat": [0], "decimalPlaces": [2],
        "parameterBindingSourcesBeginIndices": [-1], "parameterBindingSourcesCounts": [0],
    }
    # one art mesh with 3 verts / 1 tri, and its keyform positions + uvs + indices
    m.counts["artMeshes"] = 1
    m.counts["uvs"] = 6            # 3 UV pairs (count is a float count)
    m.counts["positionIndices"] = 3
    m.counts["keyformPositions"] = 6  # 3 XY pairs
    m.counts["artMeshKeyforms"] = 1
    m.sections["artMeshes"] = {
        "runtimeSpace0": b"\0" * 8, "runtimeSpace1": b"\0" * 8,
        "runtimeSpace2": b"\0" * 8, "runtimeSpace3": b"\0" * 8,
        "ids": ["ArtMesh1"], "keyformBindingSourcesIndices": [-1],
        "keyformSourcesBeginIndices": [0], "keyformSourcesCounts": [1],
        "isVisible": [1], "isEnabled": [1], "parentPartIndices": [-1],
        "parentDeformerIndices": [-1], "textureNos": [0], "drawableFlags": [0],
        "vertexCounts": [3], "uvSourcesBeginIndices": [0],
        "positionIndexSourcesBeginIndices": [0], "positionIndexSourcesCounts": [3],
        "drawableMaskSourcesBeginIndices": [0], "drawableMaskSourcesCounts": [0],
    }
    m.sections["artMeshKeyforms"] = {
        "opacities": [1.0], "drawOrders": [500.0], "keyformPositionSourcesBeginIndices": [0]}
    m.sections["keyformPositions"] = {"xys": [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]}
    m.sections["uvs"] = {"uvs": [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]}
    m.sections["positionIndices"] = {"indices": [0, 1, 2]}

    back = read_moc3(write_moc3(m))
    assert back.counts["parameters"] == 1 and back.counts["artMeshes"] == 1
    assert back.get("parameters", "ids") == ["ParamAngleX"]
    assert back.get("parameters", "minValues") == [-30.0]
    assert back.get("artMeshes", "vertexCounts") == [3]
    assert back.get("keyformPositions", "xys") == [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    assert back.get("uvs", "uvs") == [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    assert back.get("positionIndices", "indices") == [0, 1, 2]


def test_roundtrip_real_sample_if_present():
    sample = Path("out/moc3_research/haru.moc3")
    if not sample.exists():
        import pytest
        pytest.skip("no real .moc3 sample present (out/moc3_research/haru.moc3)")
    m1 = read_moc3(sample.read_bytes())
    assert m1.counts["parts"] > 0 and m1.counts["artMeshes"] > 0
    m2 = read_moc3(write_moc3(m1))
    assert m1.counts == m2.counts
    for section, field, _, _ in FIELDS:
        assert m1.sections[section].get(field) == m2.sections[section].get(field), f"{section}.{field}"
