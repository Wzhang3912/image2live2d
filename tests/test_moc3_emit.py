"""Route A / Phase 4B S2: build a .moc3 from geometry + parameters and verify it is internally
consistent when read back through the S1 codec (structure valid; runtime-load proof is separate)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

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
    def shift(dx):
        return [(x + dx, y) for x, y in rest]
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
    # The only part is a head part (face_base) and its only driver is a head-turn param, so the turn is
    # applied by the D_HEAD warp deformer (nijilive-style squash about the neck base) rather than baked
    # into the mesh keyforms -> the mesh keeps a single rest keyform and one warp deformer is emitted.
    assert r.get("artMeshes", "keyformSourcesCounts") == [1]
    assert r.counts["warpDeformers"] == 1 and r.get("deformers", "ids") == ["D_HEAD"]
    assert r.get("keys", "values") == [-30.0, 0.0, 30.0]


def test_native_moc_writer_returns_moc3_bytes():
    data = native_moc_writer(_mini_rig())
    assert data[:4] == b"MOC3" and data[4] == 1
    assert read_moc3(data).counts["artMeshes"] == 1


def test_build_moc3_carries_per_keyform_opacity():
    # A mesh that fades out across a parameter axis: opacities are stored per keyform, parallel to the
    # position keyforms, and read back verbatim. A mesh with no opacities stays fully opaque.
    params = [EmitParam("ParamEyeLOpen", 0.0, 1.0, 1.0, [0.0, 1.0])]
    parts = [EmitPart("PartLash", 500.0)]
    rest = [(-0.1, -0.1), (0.1, -0.1), (0.0, 0.1)]
    fading = EmitMesh("Lash", 0, 0, [(0, 0), (1, 0), (0.5, 1)], [(0, 1, 2)],
                      param_indices=[0], keyforms=[rest, rest], opacities=[1.0, 0.0])
    r = read_moc3(write_moc3(build_moc3(default_canvas(), params, parts, [fading])))
    assert r.get("artMeshKeyforms", "opacities") == [1.0, 0.0]


def _eye_fade_rig():
    tri = [(0.45, 0.45), (0.55, 0.45), (0.5, 0.55)]
    return Rig(
        meta=Meta(name="t"),
        textures=[Texture(id="t0", path="t0.png", width=64, height=64)],
        parts=[Part(id="lash", semantic_role=SemanticRole.eye_l, texture_id="t0", draw_order=5)],
        meshes=[Mesh(part_id="lash", vertices=tri, uvs=[(0, 0), (1, 0), (0.5, 1)], triangles=[(0, 1, 2)])],
        # A closed-eye lash line: no geometry motion, only an opacity fade — visible (1.0) when the eye
        # is closed (value 0), invisible (0.0) when open (value 1). No turn params -> no head deformer.
        parameters=[Parameter(id="ParamEyeLOpen", min=0.0, max=1.0, default=1.0, keyforms=[
            Keyform(value=0.0, opacity_overrides={"lash": 1.0}),
            Keyform(value=1.0, opacity_overrides={"lash": 0.0}),
        ])],
    )


def test_rig_to_moc3_bakes_opacity_only_param_into_a_grid_axis():
    # ParamEyeLOpen moves no vertices — it only keys the lash's opacity. It must still become a grid axis
    # (2 keyforms), or the fade would have nothing to vary along. Opacities come out in ascending-value
    # order: closed (value 0) opaque, open (value 1) transparent.
    r = read_moc3(write_moc3(rig_to_moc3(_eye_fade_rig())))
    assert r.get("artMeshes", "keyformSourcesCounts") == [2]
    assert r.get("artMeshKeyforms", "opacities") == [1.0, 0.0]
    assert r.get("parameters", "ids") == ["ParamEyeLOpen"]


def _lid_rig():
    # A part driven by ONE param for BOTH a vertex offset (a lid collapse) AND an opacity fade — the eye
    # lineart case. 'torso' keeps it off the head so no warp deformer complicates the grid.
    tri = [(0.45, 0.45), (0.55, 0.45), (0.5, 0.55)]
    return Rig(
        meta=Meta(name="t"),
        textures=[Texture(id="t0", path="t0.png", width=64, height=64)],
        parts=[Part(id="lid", semantic_role=SemanticRole.torso, texture_id="t0", draw_order=5)],
        meshes=[Mesh(part_id="lid", vertices=tri, uvs=[(0, 0), (1, 0), (0.5, 1)], triangles=[(0, 1, 2)])],
        parameters=[Parameter(id="ParamEyeLOpen", min=0.0, max=1.0, default=1.0, keyforms=[
            Keyform(value=0.0, mesh_offsets={"lid": [(0.0, 0.02)] * 3}, opacity_overrides={"lid": 0.0}),
            Keyform(value=1.0, mesh_offsets={}, opacity_overrides={"lid": 1.0}),
        ])],
    )


def test_rig_to_moc3_emits_opacity_when_the_driving_param_also_moves_the_part():
    # Regression: the opacity was kept only when its param was opacity-ONLY, so a lid driven by
    # ParamEyeLOpen for BOTH its collapse and its fade silently stayed fully opaque. It must fade.
    r = read_moc3(write_moc3(rig_to_moc3(_lid_rig())))
    assert r.get("artMeshes", "keyformSourcesCounts") == [2]
    assert r.get("artMeshKeyforms", "opacities") == [0.0, 1.0]   # value 0 -> faded out, value 1 -> shown


def _have_core() -> bool:
    try:
        import cubism_core
        cubism_core.find_core()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_core(), reason="Live2DCubismCore not found (set CUBISM_CORE); proprietary")
def test_opacity_keyform_drives_in_native_core():
    # Runtime-truth: load the emitted .moc3 in the real Cubism core and confirm the keyed opacity actually
    # drives the drawable — a byte-consistent keyform is not proof the runtime fades it. ParamEyeLOpen 0
    # (closed) -> lash fully opaque; 1 (open) -> fully transparent; the core interpolates in between.
    import cubism_core

    data = write_moc3(rig_to_moc3(_eye_fade_rig()))
    with tempfile.NamedTemporaryFile(suffix=".moc3", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        m = cubism_core.Model(path)
        def op(v):
            m.reset()
            m.set_param("ParamEyeLOpen", v)
            m.update()
            return m.opacity_of("lash")
        assert op(0.0) == pytest.approx(1.0, abs=1e-3)
        assert op(1.0) == pytest.approx(0.0, abs=1e-3)
        assert op(0.5) == pytest.approx(0.5, abs=1e-2)
    finally:
        Path(path).unlink(missing_ok=True)
