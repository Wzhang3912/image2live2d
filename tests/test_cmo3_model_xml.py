""".cmo3 model-graph export (Phase 1). Builds ``main.xml`` + textures from an IRR ``Rig`` and packs the
CAFF archive. These tests verify the graph is *internally* consistent — every ``xs.ref`` resolves, the
mandatory collections the Editor's reader requires are present, the well-known UUIDs are literal, and
every drawable is reachable from the root part (an unreachable drawable is what the Editor flags as
"(recovered)"). Whether the proprietary Cubism Editor actually opens the file cannot be checked here;
structural parity against the validated reference generator is the confidence ceiling."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from image2live2d.backends.live2d.cmo3 import rig_to_cmo3, unpack_caff
from image2live2d.backends.live2d.cmo3.model_xml import (
    DEFORMER_ROOT,
    FILTER_DEF_LAYER_FILTER,
    FILTER_DEF_LAYER_SELECTOR,
    PARAM_GROUP_ROOT,
    build_main_xml,
)
from image2live2d.irr.schema import (
    Keyform, Mesh, Meta, Parameter, Part, Rig, SemanticRole, Texture,
)


def _rig(n_parts=2):
    parts, meshes, textures = [], [], []
    for i in range(n_parts):
        textures.append(Texture(id=f"t{i}", path=f"t{i}.png", width=256, height=256))
        parts.append(Part(id=f"part{i}", semantic_role=SemanticRole.face_base, texture_id=f"t{i}",
                          draw_order=i, opacity=1.0 - 0.1 * i))
        meshes.append(Mesh(
            part_id=f"part{i}",
            vertices=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)],
            uvs=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)],
            triangles=[(0, 1, 2), (0, 2, 3)]))
    return Rig(
        meta=Meta(name="CmoTest"), textures=textures, parts=parts, meshes=meshes,
        parameters=[Parameter(id="ParamAngleX", min=-30, max=30, default=0,
                              keyforms=[Keyform(value=-30), Keyform(value=0), Keyform(value=30)])])


def _png(_tex):  # a valid 1x1 PNG signature is enough for build_main_xml (it only reads bytes/len)
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def _parse_shared(rig):
    xml_bytes, files = build_main_xml(rig, _png)
    xml = xml_bytes.decode("utf-8")
    root = ET.fromstring(xml[xml.index("<root"):])
    return xml, root, root.find("shared"), root.find("main"), files


def test_every_xs_ref_resolves():
    _, root, shared, _, _ = _parse_shared(_rig(3))
    ids = {o.get("xs.id") for o in shared}
    refs = [el.get("xs.ref") for el in root.iter() if el.get("xs.ref") is not None]
    dangling = sorted({r for r in refs if r not in ids})
    assert not dangling, f"dangling xs.ref: {dangling}"
    assert len(refs) > 50  # the graph is densely cross-linked


def test_mandatory_model_collections_present():
    _, _, _, main, _ = _parse_shared(_rig())
    model = main.find("CModelSource")
    for path in ("CImageCanvas", "CParameterSourceSet", "CDrawableSourceSet", "CPartSourceSet",
                 "CTextureManager", "CDeformerSourceSet", "CAffecterSourceSet",
                 "CParameterGroupSet", "CModelInfo"):
        assert model.find(path) is not None, f"missing required {path}"
    # rootPart must reference a CPartSource (not a bare CPartGuid) or the Editor NPEs
    assert model.find("CPartSource[@xs.n='rootPart']").get("xs.ref") is not None


def test_counts_match_rig():
    _, _, _, main, files = _parse_shared(_rig(4))
    model = main.find("CModelSource")
    assert model.find("CDrawableSourceSet/carray_list").get("count") == "4"
    assert model.find("CParameterSourceSet/carray_list").get("count") == "1"
    assert model.find("CPartSourceSet/carray_list").get("count") == "1"  # single root part in Phase 1
    assert len(files) == 4  # one texture PNG per part


def test_well_known_uuids_are_literal():
    xml, _, _, _, _ = _parse_shared(_rig())
    assert DEFORMER_ROOT in xml
    assert PARAM_GROUP_ROOT in xml
    assert FILTER_DEF_LAYER_SELECTOR in xml
    assert FILTER_DEF_LAYER_FILTER in xml


def test_all_drawables_reachable_from_root_part():
    # An unreachable drawable is exactly what the Editor marks "(recovered)".
    _, _, shared, _, _ = _parse_shared(_rig(3))
    part = next(o for o in shared if o.tag == "CPartSource")
    child_refs = {c.get("xs.ref") for c in part.find("carray_list[@xs.n='_childGuids']")}
    drawable_refs = set()
    for m in shared:
        if m.tag == "CArtMeshSource":
            g = m.find("ACDrawableSource/CDrawableGuid[@xs.n='guid']")
            drawable_refs.add(g.get("xs.ref"))
    assert drawable_refs and drawable_refs <= child_refs


def test_model_image_texture_mode():
    xml, _, _, main, _ = _parse_shared(_rig(2))
    # ModelImage pipeline must be enabled and every mesh must render in MODEL_IMAGE state
    assert main.find("CModelSource/CTextureManager/b[@xs.n='isTextureInputModelImageMode']").text == "true"
    assert xml.count(">MODEL_IMAGE<") == 0  # it's an attribute, not text
    assert xml.count('v="MODEL_IMAGE"') == 2


def test_single_layered_image_with_n_layers():
    # Single-PSD rule: ONE CLayeredImage holding N canvas-sized CLayers (else textures don't render).
    _, _, shared, _, _ = _parse_shared(_rig(3))
    assert sum(1 for o in shared if o.tag == "CLayeredImage") == 1
    assert sum(1 for o in shared if o.tag == "CLayer") == 3
    lg = next(o for o in shared if o.tag == "CLayerGroup")
    assert lg.find("ACLayerGroup/carray_list[@xs.n='_children']").get("count") == "3"


def test_geometry_maps_to_canvas_pixels():
    # vertex (0.2,0.2) on a 256px canvas -> pixel (51.2, 51.2); uv passes through.
    _, _, shared, _, _ = _parse_shared(_rig(1))
    mesh = next(o for o in shared if o.tag == "CArtMeshSource")
    positions = [float(v) for v in mesh.find("float-array[@xs.n='positions']").text.split()]
    uvs = [float(v) for v in mesh.find("float-array[@xs.n='uvs']").text.split()]
    assert positions[:2] == pytest.approx([0.2 * 256, 0.2 * 256])
    assert uvs[:2] == pytest.approx([0.2, 0.2])
    assert mesh.find("int-array[@xs.n='indices']").text == "0 1 2 0 2 3"


def test_param_guid_shared_between_source_and_pool():
    # The CParameterSource's guid must be a shared ref (Phase 2 keyform bindings reuse the same guid).
    _, _, shared, main, _ = _parse_shared(_rig())
    guid_ref = main.find("CModelSource/CParameterSourceSet/carray_list/CParameterSource/"
                         "CParameterGuid").get("xs.ref")
    assert guid_ref is not None
    shared_ids = {o.get("xs.id") for o in shared if o.tag == "CParameterGuid"}
    assert guid_ref in shared_ids


def test_rig_to_cmo3_packs_and_unpacks():
    data = rig_to_cmo3(_rig(2), asset_root=None)
    assert data[:4] == b"CAFF"
    entries = unpack_caff(data)
    paths = {e.path for e in entries}
    assert "main.xml" in paths
    assert sum(1 for e in entries if e.path.endswith(".png")) == 2
    main = next(e for e in entries if e.path == "main.xml")
    assert main.tag == "main_xml"
    assert b"<?version CModelSource:4?>" in main.content
    # placeholder PNGs are valid PNGs
    for e in entries:
        if e.path.endswith(".png"):
            assert e.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_empty_rig_rejected():
    rig = Rig(meta=Meta(name="empty"),
              textures=[Texture(id="t0", path="t0.png", width=64, height=64)],
              parts=[Part(id="p0", semantic_role=SemanticRole.face_base, texture_id="t0", draw_order=0)])
    # part p0 has no mesh -> no drawables
    with pytest.raises(ValueError, match="no drawable parts"):
        rig_to_cmo3(rig, asset_root=None)
