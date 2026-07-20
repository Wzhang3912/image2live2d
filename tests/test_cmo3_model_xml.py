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
    Deformer, DeformerType, Keyform, Mesh, Meta, Parameter, Part, PhysicsRig, Rig, SemanticRole,
    Texture,
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


def _deform_rig():
    """A rig whose parameters actually deform parts (Phase 2): an ``eye`` driven by one parameter (with a
    keyed opacity fade), and a ``face`` driven by two parameters (a 3x3 keyform grid)."""
    q = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
    tris = [(0, 1, 2), (0, 2, 3)]
    textures = [Texture(id="tex", path="tex.png", width=256, height=256)]
    parts = [
        Part(id="eye", semantic_role=SemanticRole.eye_l, texture_id="tex", draw_order=1),
        Part(id="face", semantic_role=SemanticRole.face_base, texture_id="tex", draw_order=0),
    ]
    meshes = [
        Mesh(part_id="eye", vertices=q, uvs=q, triangles=tris),
        Mesh(part_id="face", vertices=q, uvs=q, triangles=tris),
    ]
    z = [(0.0, 0.0)] * 4
    eye = Parameter(id="ParamEyeLOpen", min=0, max=1, default=1, keyforms=[
        Keyform(value=0, mesh_offsets={"eye": [(0, 0.1), (0, 0.1), (0, -0.1), (0, -0.1)]},
                opacity_overrides={"eye": 0.0}),
        Keyform(value=1, mesh_offsets={"eye": z}, opacity_overrides={"eye": 1.0})])
    dx = [(0.05, 0.0)] * 4
    ndx = [(-0.05, 0.0)] * 4
    angle_x = Parameter(id="ParamAngleX", min=-30, max=30, default=0, keyforms=[
        Keyform(value=-30, mesh_offsets={"face": ndx}),
        Keyform(value=0, mesh_offsets={"face": z}),
        Keyform(value=30, mesh_offsets={"face": dx})])
    dy = [(0.0, 0.05)] * 4
    ndy = [(0.0, -0.05)] * 4
    angle_y = Parameter(id="ParamAngleY", min=-30, max=30, default=0, keyforms=[
        Keyform(value=-30, mesh_offsets={"face": ndy}),
        Keyform(value=0, mesh_offsets={"face": z}),
        Keyform(value=30, mesh_offsets={"face": dy})])
    return Rig(meta=Meta(name="Deform"), textures=textures, parts=parts, meshes=meshes,
               parameters=[eye, angle_x, angle_y])


def _deformer_rig():
    """A rig with deformers (Phase 3): a static root rotation deformer, a warp deformer beneath it that a
    parameter drives, and a ``head`` part parented to the warp (``body`` stays on the root deformer)."""
    q = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
    tris = [(0, 1, 2), (0, 2, 3)]
    textures = [Texture(id="tex", path="tex.png", width=256, height=256)]
    parts = [
        Part(id="head", semantic_role=SemanticRole.face_base, texture_id="tex", draw_order=1,
             parent_deformer="warp_head"),
        Part(id="body", semantic_role=SemanticRole.torso, texture_id="tex", draw_order=0),
    ]
    meshes = [Mesh(part_id="head", vertices=q, uvs=q, triangles=tris),
              Mesh(part_id="body", vertices=q, uvs=q, triangles=tris)]
    lattice = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]  # 2x2
    deformers = [
        Deformer(id="rot_root", type=DeformerType.rotation, parent=None, pivot=(0.5, 0.5)),
        Deformer(id="warp_head", type=DeformerType.warp, parent="rot_root",
                 grid_rows=2, grid_cols=2, grid_vertices=lattice),
    ]
    angle = Parameter(id="ParamAngleX", min=-30, max=30, default=0, keyforms=[
        Keyform(value=-30, deformer_offsets={"warp_head": [(-0.05, 0.0)] * 4}),
        Keyform(value=0, deformer_offsets={"warp_head": [(0.0, 0.0)] * 4}),
        Keyform(value=30, deformer_offsets={"warp_head": [(0.05, 0.0)] * 4})])
    return Rig(meta=Meta(name="Deformers"), textures=textures, parts=parts, meshes=meshes,
               deformers=deformers, parameters=[angle])


def _physics_rig():
    """A rig with physics (Phase 4): a hair pendulum driven by yaw (translate) + roll (gravity-angle),
    whose output is a hair param. Physics forces the CModelSource:14 graph."""
    q = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
    tris = [(0, 1, 2), (0, 2, 3)]
    textures = [Texture(id="tex", path="tex.png", width=256, height=256)]
    parts = [Part(id="hair", semantic_role=SemanticRole.hair_front, texture_id="tex", draw_order=0)]
    meshes = [Mesh(part_id="hair", vertices=q, uvs=q, triangles=tris)]
    params = [
        Parameter(id="ParamAngleX", min=-30, max=30, default=0),
        Parameter(id="ParamAngleZ", min=-30, max=30, default=0),
        Parameter(id="ParamHairFront", min=-10, max=10, default=0),
    ]
    physics = [PhysicsRig(id="phys_hair", driver_param="ParamAngleX", output_param="ParamHairFront",
                          extra_drivers=["ParamAngleZ"])]
    return Rig(meta=Meta(name="Phys"), textures=textures, parts=parts, meshes=meshes,
               parameters=params, physics=physics)


def _deformer_named(shared, tag):
    for o in shared:
        if o.tag == tag:
            return o
    raise AssertionError(f"no {tag}")


def _shared_by_id(shared):
    return {o.get("xs.id"): o for o in shared}


def _mesh_named(shared, name):
    for o in shared:
        if o.tag == "CArtMeshSource":
            ln = o.find("ACDrawableSource/ACParameterControllableSource/s[@xs.n='localName']")
            if ln is not None and ln.text == name:
                return o
    raise AssertionError(f"no CArtMeshSource named {name!r}")


def _grid_of(shared, name):
    byid = _shared_by_id(shared)
    ref = _mesh_named(shared, name).find(
        "ACDrawableSource/ACParameterControllableSource/"
        "KeyformGridSource[@xs.n='keyformGridSource']").get("xs.ref")
    return byid[ref]


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


# --- Phase 2: keyform bindings (parameters actually deform the mesh) --------------------------------

def test_single_param_grid_has_one_cell_per_key():
    # ParamEyeLOpen has 2 keyforms -> 2 grid cells, 1 binding, 2 CArtMeshForms.
    _, _, shared, _, _ = _parse_shared(_deform_rig())
    grid = _grid_of(shared, "eye")
    assert grid.find("array_list[@xs.n='keyformsOnGrid']").get("count") == "2"
    assert grid.find("array_list[@xs.n='keyformBindings']").get("count") == "1"
    kis = sorted(k.text for k in grid.iter("i") if k.get("xs.n") == "keyIndex")
    assert kis == ["0", "1"]
    keyforms = _mesh_named(shared, "eye").find("carray_list[@xs.n='keyforms']")
    assert keyforms.get("count") == "2"
    assert len(keyforms.findall("CArtMeshForm")) == 2


def test_binding_references_shared_param_guid_and_real_keys():
    # The binding's parameterGuid must be the same shared guid the CParameterSource uses, and its keys
    # must be the parameter's sorted keyform values.
    _, _, shared, main, _ = _parse_shared(_deform_rig())
    byid = _shared_by_id(shared)
    grid = _grid_of(shared, "eye")
    binding_ref = grid.find("array_list[@xs.n='keyformBindings']/KeyformBindingSource").get("xs.ref")
    binding = byid[binding_ref]
    guid_ref = binding.find("CParameterGuid[@xs.n='parameterGuid']").get("xs.ref")
    assert byid[guid_ref].get("note") == "ParamEyeLOpen"  # the shared guid for that parameter
    keys = [f.text for f in binding.find("array_list[@xs.n='keys']")]
    assert [float(k) for k in keys] == [0.0, 1.0]
    # same guid object the source references (Phase 1 invariant still holds through Phase 2)
    src_guid = main.find("CModelSource/CParameterSourceSet/carray_list/CParameterSource/"
                         "CParameterGuid").get("xs.ref")
    assert byid[src_guid].get("note") in {"ParamEyeLOpen", "ParamAngleX", "ParamAngleY"}


def test_keyed_opacity_bakes_into_forms():
    # The eye fades: closed keyform opacity 0.0, open 1.0 -> the two forms carry those opacities.
    _, _, shared, _, _ = _parse_shared(_deform_rig())
    keyforms = _mesh_named(shared, "eye").find("carray_list[@xs.n='keyforms']")
    opac = sorted(float(f.find("ACDrawableForm/f[@xs.n='opacity']").text)
                  for f in keyforms.findall("CArtMeshForm"))
    assert opac == pytest.approx([0.0, 1.0])


def test_deformation_actually_moves_vertices():
    # The closed-eye form must differ from the open (rest) form — offsets are baked, not dropped.
    _, _, shared, _, _ = _parse_shared(_deform_rig())
    forms = _mesh_named(shared, "eye").find("carray_list[@xs.n='keyforms']").findall("CArtMeshForm")
    pos = [tuple(float(v) for v in f.find("float-array[@xs.n='positions']").text.split())
           for f in forms]
    assert pos[0] != pos[1]
    # cell order is keyIndex-ascending, so form 0 is the closed key (value 0): vertex-0 y is the rest
    # 0.2*256 shifted by the +0.1 offset, in canvas pixels; form 1 (open, value 1) is the undeformed rest.
    assert pos[0][1] == pytest.approx(0.2 * 256 + 0.1 * 256)
    assert pos[1][1] == pytest.approx(0.2 * 256)


def test_multi_param_grid_is_cartesian_product():
    # face is driven by ParamAngleX (3) and ParamAngleY (3) -> a 9-cell grid, 2 bindings, each access
    # key naming both bindings.
    _, _, shared, _, _ = _parse_shared(_deform_rig())
    grid = _grid_of(shared, "face")
    assert grid.find("array_list[@xs.n='keyformsOnGrid']").get("count") == "9"
    assert grid.find("array_list[@xs.n='keyformBindings']").get("count") == "2"
    for kog in grid.find("array_list[@xs.n='keyformsOnGrid']").findall("KeyformOnGrid"):
        kop = kog.find("KeyformGridAccessKey/array_list[@xs.n='_keyOnParameterList']")
        assert kop.get("count") == "2"  # one KeyOnParameter per bound parameter
        assert len(kop.findall("KeyOnParameter")) == 2
    assert len(_mesh_named(shared, "face").find("carray_list[@xs.n='keyforms']")
               .findall("CArtMeshForm")) == 9


def test_deform_rig_refs_all_resolve_and_pack():
    # Bindings/forms add many cross-links; none may dangle, and the whole thing still packs.
    xml, root, shared, _, _ = _parse_shared(_deform_rig())
    ids = {o.get("xs.id") for o in shared}
    dangling = sorted({el.get("xs.ref") for el in root.iter()
                       if el.get("xs.ref") is not None} - ids)
    assert not dangling, f"dangling xs.ref: {dangling}"
    data = rig_to_cmo3(_deform_rig(), asset_root=None)
    assert unpack_caff(data)  # round-trips through the CAFF container


# --- Phase 3: deformers ----------------------------------------------------------------------------

def test_deformers_populate_source_set():
    _, _, _, main, _ = _parse_shared(_deformer_rig())
    srcs = main.find("CModelSource/CDeformerSourceSet/carray_list[@xs.n='_sources']")
    assert srcs.get("count") == "2"
    tags = {c.tag for c in srcs}
    assert tags == {"CWarpDeformerSource", "CRotationDeformerSource"}


def test_deformer_hierarchy_targets():
    # warp targets its parent rotation deformer; the root rotation targets the well-known root deformer.
    _, _, shared, _, _ = _parse_shared(_deformer_rig())
    byid = _shared_by_id(shared)

    def target_note(tag):
        src = _deformer_named(shared, tag)
        ref = src.find("ACDeformerSource/CDeformerGuid[@xs.n='targetDeformerGuid']").get("xs.ref")
        return byid[ref].get("note")

    assert target_note("CWarpDeformerSource") == "rot_root"
    rot = _deformer_named(shared, "CRotationDeformerSource")
    root_ref = rot.find("ACDeformerSource/CDeformerGuid[@xs.n='targetDeformerGuid']").get("xs.ref")
    assert byid[root_ref].get("uuid") == DEFORMER_ROOT


def test_warp_deformer_has_parameter_grid():
    # ParamAngleX (3 keys) drives the warp -> 3 grid cells, 1 binding, 3 CWarpDeformerForms whose
    # lattice positions actually differ.
    _, _, shared, _, _ = _parse_shared(_deformer_rig())
    byid = _shared_by_id(shared)
    warp = _deformer_named(shared, "CWarpDeformerSource")
    assert warp.find("i[@xs.n='col']").text == "1"  # 2x2 lattice -> 1x1 segments
    assert warp.find("i[@xs.n='row']").text == "1"
    grid_ref = warp.find("ACDeformerSource/ACParameterControllableSource/"
                         "KeyformGridSource[@xs.n='keyformGridSource']").get("xs.ref")
    grid = byid[grid_ref]
    assert grid.find("array_list[@xs.n='keyformsOnGrid']").get("count") == "3"
    assert grid.find("array_list[@xs.n='keyformBindings']").get("count") == "1"
    forms = warp.find("carray_list[@xs.n='keyforms']").findall("CWarpDeformerForm")
    assert len(forms) == 3
    pos = [f.find("float-array[@xs.n='positions']").text for f in forms]
    assert len(set(pos)) == 3  # each key deforms the lattice differently
    # 8 floats per form (4 lattice points x XY)
    assert all(len(p.split()) == 8 for p in pos)


def test_rotation_deformer_is_static_at_pivot():
    # No angle in the IRR -> a single rest form, angle 0, origin at pivot*canvas (0.5*256 = 128).
    _, _, shared, _, _ = _parse_shared(_deformer_rig())
    rot = _deformer_named(shared, "CRotationDeformerSource")
    forms = rot.find("carray_list[@xs.n='keyforms']").findall("CRotationDeformerForm")
    assert len(forms) == 1
    f = forms[0]
    assert float(f.get("angle")) == 0.0
    assert float(f.get("originX")) == pytest.approx(0.5 * 256)
    assert float(f.get("originY")) == pytest.approx(0.5 * 256)


def test_part_targets_its_parent_deformer():
    _, _, shared, _, _ = _parse_shared(_deformer_rig())
    byid = _shared_by_id(shared)

    def mesh_target_note(name):
        ref = _mesh_named(shared, name).find(
            "ACDrawableSource/CDeformerGuid[@xs.n='targetDeformerGuid']").get("xs.ref")
        return byid[ref].get("note"), byid[ref].get("uuid")

    assert mesh_target_note("head")[0] == "warp_head"          # parented part targets its deformer
    assert mesh_target_note("body")[1] == DEFORMER_ROOT         # un-parented part targets the root


def test_deformers_reachable_from_root_part():
    # Both deformers must be listed in the root part's _childGuids alongside the drawables (an unreachable
    # node is what the Editor flags "(recovered)").
    _, _, shared, _, _ = _parse_shared(_deformer_rig())
    part = next(o for o in shared if o.tag == "CPartSource")
    cg = part.find("carray_list[@xs.n='_childGuids']")
    assert cg.get("count") == "4"  # 2 drawables + 2 deformers
    assert len(cg.findall("CDeformerGuid")) == 2
    assert len(cg.findall("CDrawableGuid")) == 2


def test_deformer_rig_refs_resolve_and_pack():
    xml, root, shared, _, _ = _parse_shared(_deformer_rig())
    ids = {o.get("xs.id") for o in shared}
    dangling = sorted({el.get("xs.ref") for el in root.iter()
                       if el.get("xs.ref") is not None} - ids)
    assert not dangling, f"dangling xs.ref: {dangling}"
    # deformer element/import versions present; CModelSource stays at 4 (v14 is Phase 4 / physics)
    assert "<?version CRotationDeformerForm:1?>" in xml
    assert "warp.CWarpDeformerSource" in xml
    assert "<?version CModelSource:4?>" in xml
    assert unpack_caff(rig_to_cmo3(_deformer_rig(), asset_root=None))


# --- Phase 4: physics + the CModelSource:14 tail ---------------------------------------------------

def test_physics_bumps_model_to_v14():
    xml, _, _, _, _ = _parse_shared(_physics_rig())
    assert "<?version CModelSource:14?>" in xml
    assert "<?version CModelSource:4?>" not in xml


def test_physics_settings_source_set():
    _, _, _, main, _ = _parse_shared(_physics_rig())
    pset = main.find("CModelSource/CPhysicsSettingsSourceSet[@xs.n='physicsSettingsSourceSet']")
    assert pset is not None
    srcs = pset.find("carray_list[@xs.n='_sourceCubismPhysics']")
    assert srcs.get("count") == "1"
    setting = srcs.find("CPhysicsSettingsSource")
    assert setting.find("s[@xs.n='name']").text == "ParamHairFront"


def test_physics_inputs_and_output_reference_shared_param_guids():
    _, _, shared, main, _ = _parse_shared(_physics_rig())
    byid = _shared_by_id(shared)
    setting = main.find("CModelSource/CPhysicsSettingsSourceSet/carray_list/CPhysicsSettingsSource")
    inputs = setting.find("carray_list[@xs.n='inputs']")
    assert inputs.get("count") == "2"  # yaw (translate) + roll (gravity-angle); pitch would be dropped
    src_notes = {byid[i.find("CParameterGuid[@xs.n='source']").get("xs.ref")].get("note")
                 for i in inputs.findall("CPhysicsInput")}
    assert src_notes == {"ParamAngleX", "ParamAngleZ"}
    types = {i.find("CPhysicsSourceType[@xs.n='type']").get("v") for i in inputs.findall("CPhysicsInput")}
    assert types == {"SRC_TO_X", "SRC_TO_G_ANGLE"}
    out = setting.find("carray_list[@xs.n='outputs']/CPhysicsOutput")
    dest = byid[out.find("CParameterGuid[@xs.n='destination']").get("xs.ref")].get("note")
    assert dest == "ParamHairFront"


def test_physics_vertices_and_tip_output():
    _, _, _, main, _ = _parse_shared(_physics_rig())
    setting = main.find("CModelSource/CPhysicsSettingsSourceSet/carray_list/CPhysicsSettingsSource")
    verts = setting.find("carray_list[@xs.n='vertices']")
    assert verts.get("count") == "2"  # fixed root + swinging tip
    out = setting.find("carray_list[@xs.n='outputs']/CPhysicsOutput")
    assert out.find("i[@xs.n='vertexIndex']").text == "1"  # 0-based tip = len(verts) - 1


def test_v14_mandatory_tail_present():
    _, _, shared, main, _ = _parse_shared(_physics_rig())
    byid = _shared_by_id(shared)
    model = main.find("CModelSource")
    # root parameter group entity: referenced by both rootParameterGroup and the group set
    rpg_ref = model.find("CParameterGroup[@xs.n='rootParameterGroup']").get("xs.ref")
    assert byid[rpg_ref].tag == "CParameterGroup"
    assert model.find("CParameterGroupSet/carray_list[@xs.n='_groups']").get("count") == "1"
    assert model.find("CModelInfo/CEffectParameterGroups[@xs.n='_effectParameterGroups']") is not None
    assert model.find("hash_map[@xs.n='modelOptions']") is not None
    for field in ("_icon64", "_icon32", "_icon16"):
        assert model.find(f"CImageIcon[@xs.n='{field}']") is not None
    assert model.find("CGameMotionSet[@xs.n='gameMotionSet']") is not None
    assert model.find("ModelViewerSetting[@xs.n='modelViewerSetting']") is not None
    assert model.find("CRandomPoseSettingManager[@xs.n='randomPoseSetting']") is not None


def test_physics_rig_refs_resolve_and_packs_icons():
    xml, root, shared, _, _ = _parse_shared(_physics_rig())
    ids = {o.get("xs.id") for o in shared}
    dangling = sorted({el.get("xs.ref") for el in root.iter()
                       if el.get("xs.ref") is not None} - ids)
    assert not dangling, f"dangling xs.ref: {dangling}"
    entries = unpack_caff(rig_to_cmo3(_physics_rig(), asset_root=None))
    paths = {e.path for e in entries}
    for sz in (16, 32, 64):
        assert f"cmo3_icon_{sz}.png" in paths  # v14 preview icons packed
    assert "main.xml" in paths


def test_no_physics_stays_v4_without_icons():
    # The v14 surface must NOT leak into physics-less rigs — they stay on the validated v4 graph.
    xml, _, _, main, _ = _parse_shared(_rig())
    assert "<?version CModelSource:4?>" in xml
    assert main.find("CModelSource/CPhysicsSettingsSourceSet") is None
    assert main.find("CModelSource/CParameterGroup[@xs.n='rootParameterGroup']") is None
    entries = unpack_caff(rig_to_cmo3(_rig(), asset_root=None))
    assert not any(e.path.startswith("cmo3_icon") for e in entries)
