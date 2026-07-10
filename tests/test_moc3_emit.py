"""Route A / Phase 4B S2: build a .moc3 from geometry + parameters and verify it is internally
consistent when read back through the S1 codec (structure valid; runtime-load proof is separate)."""

from __future__ import annotations

from image2live2d.backends.live2d.moc3_binary import read_moc3, write_moc3
from image2live2d.backends.live2d.moc3_emit import (
    EmitMesh, EmitParam, EmitPart, build_moc3, default_canvas, native_moc_writer, rig_to_moc3,
)
from image2live2d.irr.schema import (
    Keyform, Mesh, Meta, Parameter, Part, Rig, SemanticRole, Texture,
)


def _mini_rig():
    tri = [(0.4, 0.4), (0.6, 0.4), (0.5, 0.6)]
    return Rig(
        meta=Meta(name="t"),
        textures=[Texture(id="t0", path="t0.png", width=64, height=64)],
        parts=[Part(id="p0", semantic_role=SemanticRole.face_base, texture_id="t0", draw_order=0)],
        meshes=[Mesh(part_id="p0", vertices=tri, uvs=[(0, 0), (1, 0), (0.5, 1)], triangles=[(0, 1, 2)])],
        parameters=[Parameter(id="ParamAngleX", min=-30.0, max=30.0, default=0.0, keyforms=[
            Keyform(value=-30.0, mesh_offsets={"p0": [(-0.1, 0.0)] * 3}),
            Keyform(value=0.0, mesh_offsets={}),
            Keyform(value=30.0, mesh_offsets={"p0": [(0.1, 0.0)] * 3}),
        ])],
    )


def _sample():
    params = [EmitParam("ParamAngleX", -30.0, 30.0, 0.0, [-30.0, 0.0, 30.0])]
    parts = [EmitPart("PartBody", 500.0), EmitPart("PartTri", 400.0)]
    rest = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
    shift = lambda dx: [(x + dx, y) for x, y in rest]
    quad = EmitMesh("Quad", 0, 0, [(0, 1), (1, 1), (1, 0), (0, 0)], [(0, 1, 2), (0, 2, 3)],
                    param_indices=[0], keyforms=[shift(-0.3), rest, shift(0.3)])
    tri = EmitMesh("Tri", 1, 0, [(0, 0), (1, 0), (0.5, 1)], [(0, 1, 2)],
                   param_indices=[], keyforms=[[(0.6, 0.6), (0.9, 0.6), (0.75, 0.9)]])
    return build_moc3(default_canvas(), params, parts, [quad, tri])


def test_generated_moc3_roundtrips_and_is_consistent():
    data = write_moc3(_sample())
    assert data[:4] == b"MOC3" and data[4] == 1
    r = read_moc3(data)
    assert r.counts["parameters"] == 1 and r.counts["artMeshes"] == 2 and r.counts["parts"] == 2
    assert r.get("parameters", "ids") == ["ParamAngleX"]
    assert r.get("parameters", "minValues") == [-30.0]
    # animated quad has one keyform per parameter key (3); static triangle has exactly 1
    assert r.get("artMeshes", "keyformSourcesCounts") == [3, 1]
    assert r.get("keys", "values") == [-30.0, 0.0, 30.0]


def test_generated_indices_all_in_range():
    r = read_moc3(write_moc3(_sample()))
    n_xy = len(r.get("keyformPositions", "xys"))
    n_uv = len(r.get("uvs", "uvs"))
    n_idx = len(r.get("positionIndices", "indices"))
    n_keys = len(r.get("keys", "values"))
    for b in r.get("artMeshKeyforms", "keyformPositionSourcesBeginIndices"):
        assert 0 <= b // 2 < n_xy
    for b in r.get("artMeshes", "uvSourcesBeginIndices"):
        assert 0 <= b // 2 <= n_uv
    for b, c in zip(r.get("artMeshes", "positionIndexSourcesBeginIndices"),
                    r.get("artMeshes", "positionIndexSourcesCounts")):
        assert 0 <= b and b + c <= n_idx
    for b, c in zip(r.get("parameterBindings", "keysSourcesBeginIndices"),
                    r.get("parameterBindings", "keysSourcesCounts")):
        assert 0 <= b and b + c <= n_keys
    # every art mesh's keyform count == product of its params' key counts (1 for none)
    pbi = r.get("parameterBindingIndices", "bindingSourcesIndices")
    kb_b = r.get("keyformBindings", "parameterBindingIndexSourcesBeginIndices")
    kb_c = r.get("keyformBindings", "parameterBindingIndexSourcesCounts")
    pb_kc = r.get("parameterBindings", "keysSourcesCounts")
    for kf_count, kb in zip(r.get("artMeshes", "keyformSourcesCounts"),
                            r.get("artMeshes", "keyformBindingSourcesIndices")):
        prod = 1
        for pb in pbi[kb_b[kb]: kb_b[kb] + kb_c[kb]]:
            prod *= pb_kc[pb]
        assert kf_count == prod


def test_rig_to_moc3_full_pipeline():
    r = read_moc3(write_moc3(rig_to_moc3(_mini_rig())))
    assert r.counts["artMeshes"] == 1 and r.counts["parameters"] == 1 and r.counts["parts"] == 1
    assert r.get("parameters", "ids") == ["ParamAngleX"]
    # one param with 3 keys -> 3 keyforms for the mesh
    assert r.get("artMeshes", "keyformSourcesCounts") == [3]
    assert r.get("keys", "values") == [-30.0, 0.0, 30.0]


def test_native_moc_writer_returns_moc3_bytes():
    data = native_moc_writer(_mini_rig())
    assert data[:4] == b"MOC3" and data[4] == 1
    assert read_moc3(data).counts["artMeshes"] == 1
