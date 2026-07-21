"""Build the ``main.xml`` model graph of a ``.cmo3`` (editable Cubism Editor project) from an IRR ``Rig``.

A ``.cmo3`` bundles this ``main.xml`` plus the model's PNG textures into a CAFF archive (see
:mod:`.caff`). ``main.xml`` is a serialized Java object graph: a ``<shared>`` pool of objects (each
tagged ``xs.id="#N"``) followed by a ``<main>`` ``CModelSource`` that wires them together via
``xs.ref``. The Editor's reader is tolerant — order-independent, unknown elements ignored, missing
fields default to null/0 — but a fixed set of collections must be present or it throws, and four
built-in UUIDs must be exact literals (see :data:`DEFORMER_ROOT` etc.) or features silently break.

**Scope (this module).** An openable, *deformable* model: every drawable part becomes one
``CArtMeshSource`` with its real geometry, UVs and texture, all hung under a single root ``CPartSource``.
Parameters that key a part (via the IRR's per-vertex ``mesh_offsets`` / ``opacity_overrides``) are bound
to it through a keyform grid (Phase 2): each driving parameter is a ``KeyformBindingSource`` axis, and
the mesh carries one ``CArtMeshForm`` per grid cell (the cartesian product of the axes' keys) holding
that cell's deformed positions and opacity. A part no parameter drives keeps a single rest form.
Deformers (warps/rotations) and physics are later phases. Textures use Cubism 5's *ModelImage* pipeline:
each part is one canvas-sized ``CLayer`` inside a single ``CLayeredImage`` (a synthetic PSD), rendered
through a per-mesh filter graph. This mirrors, mesh for mesh, the reverse-engineering reference that is
confirmed to open in Cubism Editor 5.0.

Because our decomposed part textures are already full-canvas PNGs (the part painted at its true canvas
location, transparent elsewhere) and mesh UVs are full-canvas ``[0, 1]``, the mapping is direct: a
part's canvas-sized image *is* its layer, a vertex's pixel position is ``norm * canvas_size`` (both
spaces are y-down, top-left origin), and its UV passes straight through.
"""

from __future__ import annotations

import uuid as _uuid_mod
import xml.etree.ElementTree as ET
from collections.abc import Callable

from ....irr.schema import DeformerType, Rig, Texture
# Reuse the moc3 emitter's keyform math so both Live2D backends deform *identically* from the same IRR:
# which parameters drive a part, the per-vertex offset at a key, opacity keying, and the cap on how many
# params may drive one mesh (keyforms per mesh = product of their key counts). See :mod:`..moc3_emit`.
from ..moc3_emit import _affecting_params, _offset_at, _opacity_at, _opacity_params
# Reuse the physics3.json emitter's semantics so a .cmo3's editor physics matches the runtime physics we
# ship: the same input type per driver (translate vs gravity-angle vs dropped), the same pendulum
# vertices, and the same normalization ranges. See :mod:`..physics3`.
from ..physics3 import _INPUT_WEIGHT, _NORM, _OUTPUT_WEIGHT, _input_type, _vertices

# --- Well-known UUIDs the Editor compares by literal equality --------------------------------------
# The root deformer and root parameter group are referenced by these exact UUIDs; the two filter-def
# GUIDs identify the built-in CLayerSelector / CLayerFilter used by the ModelImage texture pipeline.
DEFORMER_ROOT = "71fae776-e218-4aee-873e-78e8ac0cb48a"
PARAM_GROUP_ROOT = "e9fe6eff-953b-4ce2-be7c-4a7c3913686b"
FILTER_DEF_LAYER_SELECTOR = "5e9fe1ea-0ec3-4d68-a5fa-018fc7abe301"
FILTER_DEF_LAYER_FILTER = "4083cd1f-40ba-4eda-8400-379019d55ed8"

# Processing instructions — element/format versions from Cubism Editor 5.0. CModelSource:4 is the key
# choice: v4 does not require rootParameterGroup / modelOptions / gameMotionSet, so an MVE can omit them.
_VERSION_PIS = [
    ("CArtMeshSource", "4"),
    ("CRotationDeformerForm", "1"),
    ("KeyformGridSource", "1"),
    ("CParameterGroup", "4"),
    ("SerializeFormatVersion", "2"),
    # CModelSource version is chosen per-rig in build_main_xml: 4 by default, 14 when the rig has physics
    # (only v14 parses physicsSettingsSourceSet). The v14 bump drags in mandatory rootParameterGroup /
    # modelOptions / preview-icon / gameMotionSet fields, all emitted in the v14 tail below. The value
    # here is a placeholder; build_main_xml overrides it.
    ("CModelSource", "4"),
    ("CFloatColor", "1"),
    ("CLabelColor", "0"),
    ("CModelImage", "3"),
]

# The full set of ``<?import ...?>`` class references Editor 5.0 writes for a ModelImage-mode model.
# A missing import can make the reader silently skip an element, so the whole set is reproduced.
_IMPORT_PIS = [
    "com.live2d.cubism.doc.model.ACForm",
    "com.live2d.cubism.doc.model.ACParameterControllableSource",
    "com.live2d.cubism.doc.model.CModelInfo",
    "com.live2d.cubism.doc.model.CModelSource",
    "com.live2d.cubism.doc.model.affecter.CAffecterSourceSet",
    "com.live2d.cubism.doc.model.deformer.ACDeformerForm",
    "com.live2d.cubism.doc.model.deformer.ACDeformerSource",
    "com.live2d.cubism.doc.model.deformer.CDeformerSourceSet",
    "com.live2d.cubism.doc.model.deformer.rotation.CRotationDeformerForm",
    "com.live2d.cubism.doc.model.deformer.rotation.CRotationDeformerSource",
    "com.live2d.cubism.doc.model.deformer.warp.CWarpDeformerForm",
    "com.live2d.cubism.doc.model.deformer.warp.CWarpDeformerSource",
    "com.live2d.cubism.doc.model.drawable.ACDrawableForm",
    "com.live2d.cubism.doc.model.drawable.ACDrawableSource",
    "com.live2d.cubism.doc.model.drawable.CDrawableSourceSet",
    "com.live2d.cubism.doc.model.drawable.ColorComposition",
    "com.live2d.cubism.doc.model.drawable.TextureState",
    "com.live2d.cubism.doc.model.drawable.artMesh.CArtMeshForm",
    "com.live2d.cubism.doc.model.drawable.artMesh.CArtMeshSource",
    "com.live2d.cubism.doc.model.extension.ACExtension",
    "com.live2d.cubism.doc.model.extension.editableMesh.CEditableMeshExtension",
    "com.live2d.cubism.doc.model.extension.meshGenerator.CMeshGeneratorExtension",
    "com.live2d.cubism.doc.model.extension.meshGenerator.MeshGenerateSetting",
    "com.live2d.cubism.doc.model.extension.textureInput.ACTextureInput",
    "com.live2d.cubism.doc.model.extension.textureInput.CTextureInputExtension",
    "com.live2d.cubism.doc.model.extension.textureInput.CTextureInput_ModelImage",
    "com.live2d.cubism.doc.model.extension.textureInput.inputFilter.CLayerInputData",
    "com.live2d.cubism.doc.model.extension.textureInput.inputFilter.CLayerSelectorMap",
    "com.live2d.cubism.doc.model.extension.textureInput.inputFilter.ModelImageFilterEnv",
    "com.live2d.cubism.doc.model.extension.textureInput.inputFilter.ModelImageFilterSet",
    "com.live2d.cubism.doc.model.id.CDeformerId",
    "com.live2d.cubism.doc.model.id.CDrawableId",
    "com.live2d.cubism.doc.model.id.CParameterId",
    "com.live2d.cubism.doc.model.id.CPartId",
    "com.live2d.cubism.doc.model.interpolator.InterpolationType",
    "com.live2d.cubism.doc.model.interpolator.KeyOnParameter",
    "com.live2d.cubism.doc.model.interpolator.KeyformBindingSource",
    "com.live2d.cubism.doc.model.interpolator.KeyformGridAccessKey",
    "com.live2d.cubism.doc.model.interpolator.KeyformGridSource",
    "com.live2d.cubism.doc.model.interpolator.KeyformOnGrid",
    "com.live2d.cubism.doc.model.interpolator.extendedInterpolation.ExtendedInterpolationType",
    "com.live2d.cubism.doc.model.morphTarget.KeyFormMorphTargetSet",
    "com.live2d.cubism.doc.model.morphTarget.MorphTargetBlendWeightConstraintSet",
    "com.live2d.cubism.doc.model.options.edition.EditorEdition",
    "com.live2d.cubism.doc.model.param.CParameterSource",
    "com.live2d.cubism.doc.model.param.CParameterSource$Type",
    "com.live2d.cubism.doc.model.param.CParameterSourceSet",
    "com.live2d.cubism.doc.model.param.group.CParameterGroup",
    "com.live2d.cubism.doc.model.param.group.CParameterGroupSet",
    "com.live2d.cubism.doc.model.parts.CPartForm",
    "com.live2d.cubism.doc.model.parts.CPartSource",
    "com.live2d.cubism.doc.model.parts.CPartSourceSet",
    "com.live2d.cubism.doc.model.texture.CTextureManager",
    "com.live2d.cubism.doc.model.texture.LayeredImageWrapper",
    "com.live2d.cubism.doc.model.texture.TextureImageGroup",
    "com.live2d.cubism.doc.model.texture.modelImage.CModelImage",
    "com.live2d.cubism.doc.model.texture.modelImage.CModelImageGroup",
    "com.live2d.cubism.doc.resources.ACImageLayer",
    "com.live2d.cubism.doc.resources.ACLayerEntry",
    "com.live2d.cubism.doc.resources.ACLayerGroup",
    "com.live2d.cubism.doc.resources.CLayer",
    "com.live2d.cubism.doc.resources.CLayerGroup",
    "com.live2d.cubism.doc.resources.CLayerIdentifier",
    "com.live2d.cubism.doc.resources.CLayeredImage",
    "com.live2d.cubism.doc.resources.LayerSet",
    "com.live2d.doc.CoordType",
    "com.live2d.graphics.CImageCanvas",
    "com.live2d.graphics.CImageResource",
    "com.live2d.graphics.CWritableImage",
    "com.live2d.graphics.cachedImage.CCachedImage",
    "com.live2d.graphics.cachedImage.CCachedImageManager",
    "com.live2d.graphics.cachedImage.CachedImageType",
    "com.live2d.graphics.filter.AValueConnector",
    "com.live2d.graphics.filter.FilterEnv",
    "com.live2d.graphics.filter.FilterEnv$EnvValueSet",
    "com.live2d.graphics.filter.FilterSet",
    "com.live2d.graphics.filter.FilterSet$EnvConnection",
    "com.live2d.graphics.filter.FilterValue",
    "com.live2d.graphics.filter.concreteConnector.EnvValueConnector",
    "com.live2d.graphics.filter.concreteConnector.FilterOutputValueConnector",
    "com.live2d.graphics.filter.filterInstance.FilterInstance",
    "com.live2d.graphics.filter.id.FilterInstanceId",
    "com.live2d.graphics.filter.id.FilterValueId",
    "com.live2d.graphics.psd.blend.ACBlend",
    "com.live2d.graphics.psd.blend.CBlend_Normal",
    "com.live2d.graphics3d.editableMesh.GEditableMesh2",
    "com.live2d.graphics3d.texture.Anisotropy",
    "com.live2d.graphics3d.texture.GTexture",
    "com.live2d.graphics3d.texture.GTexture$FilterMode",
    "com.live2d.graphics3d.texture.GTexture2D",
    "com.live2d.graphics3d.texture.MagFilter",
    "com.live2d.graphics3d.texture.MinFilter",
    "com.live2d.graphics3d.texture.WrapMode",
    "com.live2d.graphics3d.type.GVector2",
    "com.live2d.type.CAffine",
    "com.live2d.type.CColor",
    "com.live2d.type.CDeformerGuid",
    "com.live2d.type.CDrawableGuid",
    "com.live2d.type.CExtensionGuid",
    "com.live2d.type.CFloatColor",
    "com.live2d.type.CFormGuid",
    "com.live2d.type.CImageIcon",
    "com.live2d.type.CLayerGuid",
    "com.live2d.type.CLayeredImageGuid",
    "com.live2d.type.CModelGuid",
    "com.live2d.type.CModelImageGuid",
    "com.live2d.type.CParameterGroupGuid",
    "com.live2d.type.CParameterGuid",
    "com.live2d.type.CPartGuid",
    "com.live2d.type.CPoint",
    "com.live2d.type.CRect",
    "com.live2d.type.CSize",
    "com.live2d.type.GEditableMeshGuid",
    "com.live2d.type.GTextureGuid",
    "com.live2d.type.StaticFilterDefGuid",
    # --- physics + the CModelSource:14 tail (Phase 4). Emitted only for rigs that carry physics; import
    # order is irrelevant to the reader, but a class that is *emitted* without its import is a fatal
    # ClassNotFoundException, so every v14 class we can emit is registered here.
    "com.live2d.cubism.doc.gameData.motions.CGameMotionSet",
    "com.live2d.cubism.doc.gameData.physics.CPhysicsController$CPhysicsSourceType",
    "com.live2d.cubism.doc.gameData.physics.CPhysicsInput",
    "com.live2d.cubism.doc.gameData.physics.CPhysicsOutput",
    "com.live2d.cubism.doc.gameData.physics.CPhysicsSettingsSource",
    "com.live2d.cubism.doc.gameData.physics.CPhysicsSettingsSourceSet",
    "com.live2d.cubism.doc.gameData.physics.CPhysicsVertex",
    "com.live2d.cubism.doc.model.CEffectParameterGroups",
    "com.live2d.cubism.doc.model.drawable.artPath.Line.CArtPathBrushSetting",
    "com.live2d.cubism.doc.model.id.CParameterGroupId",
    "com.live2d.cubism.doc.model.id.CPhysicsSettingId",
    "com.live2d.cubism.doc.model.randomPose.CRandomPoseSettingManager",
    "com.live2d.cubism.doc.modeling.ui.guide.CGuidesSetting",
    "com.live2d.cubism.doc.modeling.ui.viewer.ModelViewerSetting",
    "com.live2d.type.CPhysicsDataGuid",
    "com.live2d.type.CPhysicsSettingsGuid",
]

_IDENTITY_AFFINE = dict(m00="1.0", m01="0.0", m02="0.0", m10="0.0", m11="1.0", m12="0.0")


def _new_uuid() -> str:
    return str(_uuid_mod.uuid4())


def _e(tag: str, **attrs: object) -> ET.Element:
    """A namespaced XML element. ``xs__n`` -> ``xs.n`` etc. (``.`` is illegal in a Python kwarg)."""
    el = ET.Element(tag)
    for k, v in attrs.items():
        el.set(k.replace("__", "."), str(v))
    return el


def _sub(parent: ET.Element, tag: str, **attrs: object) -> ET.Element:
    el = _e(tag, **attrs)
    parent.append(el)
    return el


def _text(parent: ET.Element, tag: str, text: str, **attrs: object) -> ET.Element:
    el = _sub(parent, tag, **attrs)
    el.text = text
    return el


class _Shared:
    """The ``<shared>`` object pool. Each ``add`` allocates the next ``#N`` id and records the object's
    positional ``xs.idx``; returns ``(element, ref_id)`` so callers can wire ``xs.ref`` links to it."""

    def __init__(self) -> None:
        self.objects: list[ET.Element] = []

    def add(self, tag: str, **attrs: object) -> tuple[ET.Element, str]:
        ref = f"#{len(self.objects)}"
        el = _e(tag, **{**attrs, "xs__id": ref, "xs__idx": str(len(self.objects))})
        self.objects.append(el)
        return el, ref


def _canvas_size(rig: Rig, drawn_texs: list[Texture]) -> tuple[int, int]:
    """The canvas (in pixels) the model lives on. Our decomposed part textures are full-canvas, so any
    drawn texture carries the canvas size; take the largest to be safe against a stray cropped layer."""
    if not drawn_texs:
        return 1024, 1024
    return max(t.width for t in drawn_texs), max(t.height for t in drawn_texs)


def _edges_from_triangles(triangles: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
    """Unique undirected edges of a triangulation, for the editable-mesh representation. The drawable
    renders from ``indices`` regardless; these let the Editor's mesh tool re-edit the geometry."""
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for a, b, c in triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u <= v else (v, u)
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


def build_main_xml(
    rig: Rig, load_png: Callable[[Texture], bytes]
) -> tuple[bytes, list[tuple[str, bytes]]]:
    """Serialize ``rig`` to ``(main_xml_bytes, [(archive_path, png_bytes), ...])``.

    ``load_png(texture)`` returns the RGBA PNG bytes for a texture (the caller decides where from — an
    asset directory, an in-memory image, ...). Only textures used by drawable parts are emitted, one
    canvas-sized ``CLayer`` per part.
    """
    tex_of = {t.id: t for t in rig.textures}
    drawn = [p for p in rig.parts_in_draw_order() if rig.mesh_for(p.id) is not None]
    if not drawn:
        raise ValueError("cannot export a .cmo3 from a rig with no drawable parts")
    drawn_texs = [tex_of[p.texture_id] for p in drawn]
    canvas_w, canvas_h = _canvas_size(rig, drawn_texs)

    sh = _Shared()

    # ---- global shared GUIDs & singletons --------------------------------------------------------
    _, ref_param_group = sh.add("CParameterGroupGuid", uuid=PARAM_GROUP_ROOT, note="root_group")
    _, ref_part_root = sh.add("CPartGuid", uuid=_new_uuid(), note="PartRoot")
    _, ref_model = sh.add("CModelGuid", uuid=_new_uuid(), note="model")

    # Parameter guids are shared: the CParameterSource and (from Phase 2) each mesh's keyform binding
    # must reference the *same* guid object, so allocate one per parameter up front.
    param_guid_ref: dict[str, str] = {
        p.id: sh.add("CParameterGuid", uuid=_new_uuid(), note=p.id)[1] for p in rig.parameters
    }
    # Canonical parameter order (matches the moc3 grid): a mesh's keyform axes are laid out ascending by
    # this index so the two backends index the grid identically.
    pidx = {p.id: k for k, p in enumerate(rig.parameters)}

    blend_normal, ref_blend = sh.add("CBlend_Normal")
    _blend_super = _sub(blend_normal, "ACBlend", xs__n="super")
    _text(_blend_super, "s", "通常", xs__n="displayName")  # "通常" (Normal)

    _, ref_li_guid = sh.add("CLayeredImageGuid", uuid=_new_uuid(), note="synthetic_psd")
    _, ref_deformer_root = sh.add("CDeformerGuid", uuid=DEFORMER_ROOT, note="ROOT")

    # Deformer guids are shared and allocated up front so a mesh can target its parent deformer during the
    # loop below and each deformer can target its own parent deformer regardless of build order.
    deformer_guid: dict[str, str] = {
        d.id: sh.add("CDeformerGuid", uuid=_new_uuid(), note=d.id)[1] for d in rig.deformers
    }

    coord_type, ref_coord = sh.add("CoordType")
    _text(coord_type, "s", "DeformerLocal", xs__n="coordName")

    _, ref_fdef_sel = sh.add("StaticFilterDefGuid", uuid=FILTER_DEF_LAYER_SELECTOR, note="CLayerSelector")
    _, ref_fdef_flt = sh.add("StaticFilterDefGuid", uuid=FILTER_DEF_LAYER_FILTER, note="CLayerFilter")

    # Shared filter-graph value ids (port identifiers) and their FilterValue definitions.
    _, ref_fvid_ilf_output = sh.add("FilterValueId", idstr="ilf_outputLayerData")
    _, ref_fvid_mi_layer = sh.add("FilterValueId", idstr="mi_input_layerInputData")
    _, ref_fvid_ilf_input = sh.add("FilterValueId", idstr="ilf_inputLayerData")
    _, ref_fvid_mi_guid = sh.add("FilterValueId", idstr="mi_currentImageGuid")
    _, ref_fvid_ilf_guid = sh.add("FilterValueId", idstr="ilf_currentImageGuid")
    _, ref_fvid_mi_out_img = sh.add("FilterValueId", idstr="mi_output_image")
    _, ref_fvid_mi_out_xfm = sh.add("FilterValueId", idstr="mi_output_transform")
    _, ref_fvid_ilf_in_layer = sh.add("FilterValueId", idstr="ilf_inputLayer")

    def _filter_value(name: str, id_ref: str | None = None, inline_id: str | None = None) -> str:
        fv, ref = sh.add("FilterValue")
        _text(fv, "s", name, xs__n="name")
        if id_ref is not None:
            _sub(fv, "FilterValueId", xs__n="id", xs__ref=id_ref)
        else:
            _sub(fv, "FilterValueId", xs__n="id", idstr=inline_id)
        _sub(fv, "null", xs__n="defaultValueInitializer")
        return ref

    ref_fv_sel = _filter_value("Select Layer", id_ref=ref_fvid_ilf_output)
    ref_fv_imp = _filter_value("Import Layer", id_ref=ref_fvid_mi_layer)
    ref_fv_imp_sel = _filter_value("Import Layer selection", id_ref=ref_fvid_ilf_input)
    ref_fv_cur_guid = _filter_value("Current GUID", id_ref=ref_fvid_mi_guid)
    ref_fv_sel_guid = _filter_value("GUID of Selected Source Image", id_ref=ref_fvid_ilf_guid)
    ref_fv_out_img = _filter_value("Output image", id_ref=ref_fvid_mi_out_img)
    ref_fv_out_img_res = _filter_value("Output Image (Resource Format)", inline_id="ilf_outputImageRes")
    ref_fv_out_xfm = _filter_value("LayerToCanvas変換", id_ref=ref_fvid_mi_out_xfm)
    ref_fv_out_xfm2 = _filter_value("LayerToCanvas変換", inline_id="ilf_outputTransform")

    # The synthetic-PSD document and its root layer group; per-part CLayers are appended below.
    layered_img, ref_li = sh.add("CLayeredImage")
    layer_group, ref_lg = sh.add("CLayerGroup")
    layer_refs: list[str] = []  # (element order matches `drawn`)

    # The image group that owns the inline CModelImages (one per mesh, appended below).
    img_group, ref_img_grp = sh.add("CModelImageGroup")

    # ---- per-part shared objects -----------------------------------------------------------------
    mesh_refs: list[str] = []
    drawable_refs: list[str] = []
    model_image_els: list[ET.Element] = []
    texture_files: list[tuple[str, bytes]] = []

    for i, part in enumerate(drawn):
        mesh = rig.mesh_for(part.id)
        tex = tex_of[part.texture_id]
        name = _safe_name(part.id, i)
        png = load_png(tex)
        arc_path = f"texture_{i:02d}.png"
        texture_files.append((arc_path, png))

        ref_drawable = sh.add("CDrawableGuid", uuid=_new_uuid(), note=name)[1]
        ref_mi_guid = sh.add("CModelImageGuid", uuid=_new_uuid(), note=f"modelimg{i}")[1]
        ref_tex_guid = sh.add("GTextureGuid", uuid=_new_uuid(), note=f"tex{i}")[1]
        ref_ext_mesh = sh.add("CExtensionGuid", uuid=_new_uuid(), note="mesh_ext")[1]
        ref_ext_tex = sh.add("CExtensionGuid", uuid=_new_uuid(), note="tex_ext")[1]
        ref_emesh = sh.add("GEditableMeshGuid", uuid=_new_uuid(), note=f"editmesh{i}")[1]
        drawable_refs.append(ref_drawable)

        # CImageResource — this part's canvas-sized PNG.
        img_res, ref_img = sh.add(
            "CImageResource", width=str(tex.width), height=str(tex.height), type="INT_ARGB",
            imageFileBuf_size=str(len(png)), previewFileBuf_size="0")
        _sub(img_res, "file", xs__n="imageFileBuf", path=arc_path)

        # CLayer inside the shared CLayeredImage.
        ref_layer = _build_layer(sh, name, i, tex.width, tex.height, ref_blend, ref_lg, ref_li, ref_img)
        layer_refs.append(ref_layer)

        # Per-mesh filter graph (selector + filter) driving the ModelImage pipeline.
        ref_fset = _build_filter_set(
            sh, ref_fdef_sel, ref_fdef_flt,
            ref_fvid_ilf_output, ref_fvid_ilf_input, ref_fvid_ilf_guid, ref_fvid_mi_layer,
            ref_fvid_mi_guid, ref_fvid_mi_out_img, ref_fvid_mi_out_xfm, ref_fvid_ilf_in_layer,
            ref_fv_sel, ref_fv_imp, ref_fv_imp_sel, ref_fv_cur_guid, ref_fv_sel_guid,
            ref_fv_out_img, ref_fv_out_img_res, ref_fv_out_xfm, ref_fv_out_xfm2, i)

        ref_tex2d = _build_texture2d(sh, name, ref_tex_guid, ref_img)
        ref_tie = _build_texture_input_ext(sh, ref_ext_tex, ref_mi_guid)

        # Keyform grid: bind every parameter that deforms/fades this part; one grid cell (CArtMeshForm)
        # per cartesian combination of their keys. A part no parameter drives gets a single rest cell.
        ref_kfg_mesh, cell_forms = _mesh_keyform_grid(
            sh, rig, part, mesh, canvas_w, canvas_h, pidx, param_guid_ref)

        # A part parented to a deformer targets it; otherwise it hangs directly off the root deformer.
        ref_mesh_target = (
            deformer_guid[part.parent_deformer] if part.parent_deformer else ref_deformer_root)
        ref_mesh = _build_art_mesh(
            sh, name, mesh, canvas_w, canvas_h, ref_part_root, ref_kfg_mesh, ref_ext_mesh,
            ref_emesh, ref_coord, ref_tie, ref_drawable, ref_mesh_target, cell_forms,
            ref_tex2d)
        mesh_refs.append(ref_mesh)

        # Point the texture-input extension's _owner back at the mesh now that it exists.
        _sub(sh.objects[_idx(ref_tie)].find("ACExtension"), "CArtMeshSource", xs__n="_owner",
             xs__ref=ref_mesh)

        model_image_els.append(_model_image(
            ref_mi_guid, name, ref_fset, ref_fvid_mi_guid, ref_fvid_mi_layer, ref_li_guid,
            ref_layer, ref_img, ref_img_grp, canvas_w, canvas_h))

    # ---- finish the shared CLayeredImage / CLayerGroup / CModelImageGroup -------------------------
    _fill_layer_group(layer_group, ref_blend, ref_li, layer_refs)
    _fill_layered_image(layered_img, canvas_w, canvas_h, ref_li_guid, ref_lg, layer_refs)
    _fill_model_image_group(img_group, ref_li_guid, model_image_els)

    # ---- deformers (Phase 3) ---------------------------------------------------------------------
    deformer_sources = _build_deformers(
        sh, rig, canvas_w, canvas_h, ref_coord, param_guid_ref, pidx, ref_part_root, deformer_guid,
        ref_deformer_root)

    # The single root part owns every drawable and every deformer (its _childGuids reaches both).
    ref_part_form = sh.add("CFormGuid", uuid=_new_uuid(), note="PartRoot_form")[1]
    ref_kfg_part = _static_keyform_grid(sh, ref_part_form)
    deformer_child_refs = [deformer_guid[d.id] for d in rig.deformers]
    ref_part_src = _build_root_part(
        sh, ref_part_root, ref_kfg_part, ref_deformer_root, drawable_refs, deformer_child_refs,
        ref_part_form)

    # Physics needs CModelSource:14, which requires a real root parameter-group entity (v4 got away with a
    # bare guid + empty group set). Build it only on the physics path so physics-less rigs stay on the
    # validated v4 graph unchanged.
    has_physics = bool(rig.physics)
    ref_root_pg = None
    if has_physics:
        ref_root_pg = _build_root_param_group(sh, ref_param_group, param_guid_ref, rig)
        texture_files = texture_files + _icon_files()

    # ---- assemble main.xml -----------------------------------------------------------------------
    root = _e("root", fileFormatVersion="402030000")
    shared_el = _sub(root, "shared")
    for obj in sh.objects:
        shared_el.append(obj)

    main_el = _sub(root, "main")
    _build_model_source(
        main_el, rig, ref_model, canvas_w, canvas_h, ref_param_group, param_guid_ref, mesh_refs,
        ref_part_src, ref_li, ref_img_grp, deformer_sources, ref_root_pg)

    model_version = "14" if has_physics else "4"
    version_pis = [(n, model_version if n == "CModelSource" else v) for n, v in _VERSION_PIS]
    pi_lines = ['<?xml version="1.0" encoding="UTF-8"?>']
    pi_lines += [f"<?version {n}:{v}?>" for n, v in version_pis]
    pi_lines += [f"<?import {imp}?>" for imp in _IMPORT_PIS]
    xml_body = ET.tostring(root, encoding="unicode")
    full = "\n".join(pi_lines) + "\n" + xml_body
    return full.encode("utf-8"), texture_files


# --- helpers that build sub-graphs (kept small and named so the flow above reads top-down) ---------

def _safe_name(part_id: str, i: int) -> str:
    """A drawable/layer display name derived from the part id (ids are already unique in the rig)."""
    cleaned = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in part_id).strip("_")
    return cleaned or f"ArtMesh{i}"


# ``_Shared`` records objects in order, so an ``xs.id`` "#N" indexes directly into ``objects``.
def _idx(ref: str) -> int:
    return int(ref[1:])


def _build_layer(sh, name, i, w, h, ref_blend, ref_lg, ref_li, ref_img) -> str:
    layer, ref_layer = sh.add("CLayer")
    acil = _sub(layer, "ACImageLayer", xs__n="super")
    ale = _sub(acil, "ACLayerEntry", xs__n="super")
    _text(ale, "s", name, xs__n="name")
    _text(ale, "s", "", xs__n="memo")
    _text(ale, "b", "true", xs__n="isVisible")
    _text(ale, "b", "false", xs__n="isClipping")
    _sub(ale, "CBlend_Normal", xs__n="blend", xs__ref=ref_blend)
    _sub(ale, "CLayerGuid", xs__n="guid", uuid=_new_uuid(), note="(no debug info)")
    _sub(ale, "CLayerGroup", xs__n="group", xs__ref=ref_lg)
    _text(ale, "i", "255", xs__n="opacity255")
    _sub(ale, "hash_map", xs__n="_optionOfIOption", count="0", keyType="string")
    _sub(ale, "CLayeredImage", xs__n="_layeredImage", xs__ref=ref_li)
    _sub(layer, "CImageResource", xs__n="imageResource", xs__ref=ref_img)
    bounds = _sub(layer, "CRect", xs__n="boundsOnImageDoc")
    _text(bounds, "i", "0", xs__n="x")
    _text(bounds, "i", "0", xs__n="y")
    _text(bounds, "i", str(w), xs__n="width")
    _text(bounds, "i", str(h), xs__n="height")
    lid = _sub(layer, "CLayerIdentifier", xs__n="layerIdentifier")
    _text(lid, "s", name, xs__n="layerName")
    _text(lid, "s", f"00-00-{(i + 1) >> 8 & 0xFF:02d}-{(i + 1) & 0xFF:02d}", xs__n="layerId")
    _text(lid, "i", str(i + 1), xs__n="layerIdValue_testImpl")
    _sub(layer, "null", xs__n="icon16")
    _sub(layer, "null", xs__n="icon64")
    _sub(layer, "linked_map", xs__n="layerInfo", count="0", keyType="string")
    _sub(layer, "hash_map", xs__n="_optionOfIOption", count="0", keyType="string")
    return ref_layer


def _fill_layer_group(layer_group, ref_blend, ref_li, layer_refs) -> None:
    alg = _sub(layer_group, "ACLayerGroup", xs__n="super")
    ale = _sub(alg, "ACLayerEntry", xs__n="super")
    _text(ale, "s", "root", xs__n="name")
    _text(ale, "s", "", xs__n="memo")
    _text(ale, "b", "true", xs__n="isVisible")
    _text(ale, "b", "false", xs__n="isClipping")
    _sub(ale, "CBlend_Normal", xs__n="blend", xs__ref=ref_blend)
    _sub(ale, "CLayerGuid", xs__n="guid", uuid=_new_uuid(), note="(no debug info)")
    _sub(ale, "null", xs__n="group")
    _text(ale, "i", "255", xs__n="opacity255")
    _sub(ale, "hash_map", xs__n="_optionOfIOption", count="0", keyType="string")
    _sub(ale, "CLayeredImage", xs__n="_layeredImage", xs__ref=ref_li)
    children = _sub(alg, "carray_list", xs__n="_children", count=str(len(layer_refs)))
    for lref in layer_refs:
        _sub(children, "CLayer", xs__ref=lref)
    _sub(layer_group, "null", xs__n="layerIdentifier")


def _fill_layered_image(layered_img, w, h, ref_li_guid, ref_lg, layer_refs) -> None:
    _text(layered_img, "s", "synthetic.psd", xs__n="name")
    _text(layered_img, "s", "", xs__n="memo")
    _text(layered_img, "i", str(w), xs__n="width")
    _text(layered_img, "i", str(h), xs__n="height")
    _text(layered_img, "file", "synthetic.psd", xs__n="psdFile")
    _text(layered_img, "s", "", xs__n="description")
    _sub(layered_img, "CLayeredImageGuid", xs__n="guid", xs__ref=ref_li_guid)
    _sub(layered_img, "null", xs__n="psdBytes")
    _text(layered_img, "l", "0", xs__n="psdFileLastModified")
    _sub(layered_img, "CLayerGroup", xs__n="_rootLayer", xs__ref=ref_lg)
    layer_set = _sub(layered_img, "LayerSet", xs__n="layerSet")
    _sub(layer_set, "CLayeredImage", xs__n="_layeredImage", xs__ref=layered_img.get("xs.id"))
    ls_list = _sub(layer_set, "carray_list", xs__n="_layerEntryList", count=str(len(layer_refs) + 1))
    _sub(ls_list, "CLayerGroup", xs__ref=ref_lg)
    for lref in layer_refs:
        _sub(ls_list, "CLayer", xs__ref=lref)
    _sub(layered_img, "null", xs__n="icon16")
    _sub(layered_img, "null", xs__n="icon64")


def _build_filter_set(
    sh, ref_fdef_sel, ref_fdef_flt, ref_fvid_ilf_output, ref_fvid_ilf_input, ref_fvid_ilf_guid,
    ref_fvid_mi_layer, ref_fvid_mi_guid, ref_fvid_mi_out_img, ref_fvid_mi_out_xfm,
    ref_fvid_ilf_in_layer, ref_fv_sel, ref_fv_imp, ref_fv_imp_sel, ref_fv_cur_guid, ref_fv_sel_guid,
    ref_fv_out_img, ref_fv_out_img_res, ref_fv_out_xfm, ref_fv_out_xfm2, i,
) -> str:
    filter_set, ref_fset = sh.add("ModelImageFilterSet")
    _, ref_fiid0 = sh.add("FilterInstanceId", idstr=f"filter{i}_0")
    fi_sel, ref_fi_sel = sh.add("FilterInstance", filterName="CLayerSelector")
    fout, ref_fout = sh.add("FilterOutputValueConnector")
    _, ref_fiid1 = sh.add("FilterInstanceId", idstr=f"filter{i}_1")
    fi_flt, ref_fi_flt = sh.add("FilterInstance", filterName="CLayerFilter")

    _sub(fout, "AValueConnector", xs__n="super")
    _sub(fout, "FilterInstance", xs__n="instance", xs__ref=ref_fi_sel)
    _sub(fout, "FilterValueId", xs__n="id", xs__ref=ref_fvid_ilf_output)
    _sub(fout, "FilterValue", xs__n="valueDef", xs__ref=ref_fv_sel)

    _sub(fi_sel, "StaticFilterDefGuid", xs__n="filterDefGuid", xs__ref=ref_fdef_sel)
    _sub(fi_sel, "null", xs__n="filterDef")
    _sub(fi_sel, "FilterInstanceId", xs__n="filterId", xs__ref=ref_fiid0)
    ic = _sub(fi_sel, "hash_map", xs__n="inputConnectors", count="2")
    _env_input(ic, ref_fvid_ilf_input, ref_fvid_mi_layer)
    _env_input(ic, ref_fvid_ilf_guid, ref_fvid_mi_guid)
    oc = _sub(fi_sel, "hash_map", xs__n="outputConnectors", count="1")
    e = _sub(oc, "entry")
    _sub(e, "FilterValueId", xs__n="key", xs__ref=ref_fvid_ilf_output)
    _sub(e, "FilterOutputValueConnector", xs__n="value", xs__ref=ref_fout)
    _sub(fi_sel, "ModelImageFilterSet", xs__n="ownerFilterSet", xs__ref=ref_fset)

    _sub(fi_flt, "StaticFilterDefGuid", xs__n="filterDefGuid", xs__ref=ref_fdef_flt)
    _sub(fi_flt, "null", xs__n="filterDef")
    _sub(fi_flt, "FilterInstanceId", xs__n="filterId", xs__ref=ref_fiid1)
    icf = _sub(fi_flt, "hash_map", xs__n="inputConnectors", count="1")
    e = _sub(icf, "entry")
    _sub(e, "FilterValueId", xs__n="key", xs__ref=ref_fvid_ilf_in_layer)
    _sub(e, "FilterOutputValueConnector", xs__n="value", xs__ref=ref_fout)
    _sub(fi_flt, "hash_map", xs__n="outputConnectors", count="0", keyType="string")
    _sub(fi_flt, "ModelImageFilterSet", xs__n="ownerFilterSet", xs__ref=ref_fset)

    fs = _sub(filter_set, "FilterSet", xs__n="super")
    fm = _sub(fs, "linked_map", xs__n="filterMap", count="2")
    for fiid, fi in ((ref_fiid0, ref_fi_sel), (ref_fiid1, ref_fi_flt)):
        e = _sub(fm, "entry")
        _sub(e, "FilterInstanceId", xs__n="key", xs__ref=fiid)
        _sub(e, "FilterInstance", xs__n="value", xs__ref=fi)
    ei = _sub(fs, "linked_map", xs__n="_externalInputs", count="2")
    _ext_conn(ei, ref_fvid_mi_layer, ref_fv_imp, ref_fi_sel, ref_fv_imp_sel)
    _ext_conn(ei, ref_fvid_mi_guid, ref_fv_cur_guid, ref_fi_sel, ref_fv_sel_guid)
    eo = _sub(fs, "linked_map", xs__n="_externalOutputs", count="2")
    _ext_conn(eo, ref_fvid_mi_out_img, ref_fv_out_img, ref_fi_flt, ref_fv_out_img_res)
    _ext_conn(eo, ref_fvid_mi_out_xfm, ref_fv_out_xfm, ref_fi_flt, ref_fv_out_xfm2)
    return ref_fset


def _env_input(parent, key_ref, env_ref) -> None:
    e = _sub(parent, "entry")
    _sub(e, "FilterValueId", xs__n="key", xs__ref=key_ref)
    evc = _sub(e, "EnvValueConnector", xs__n="value")
    _sub(evc, "AValueConnector", xs__n="super")
    _sub(evc, "FilterValueId", xs__n="envValueId", xs__ref=env_ref)


def _ext_conn(parent, key_ref, env_def_ref, filter_ref, filter_val_ref) -> None:
    e = _sub(parent, "entry")
    _sub(e, "FilterValueId", xs__n="key", xs__ref=key_ref)
    ec = _sub(e, "EnvConnection", xs__n="value")
    _sub(ec, "FilterValue", xs__n="_envValueDef", xs__ref=env_def_ref)
    _sub(ec, "FilterInstance", xs__n="filter", xs__ref=filter_ref)
    _sub(ec, "FilterValue", xs__n="filterValueDef", xs__ref=filter_val_ref)


def _build_texture2d(sh, name, ref_tex_guid, ref_img) -> str:
    tex2d, ref_tex2d = sh.add("GTexture2D")
    gtex = _sub(tex2d, "GTexture", xs__n="super")
    _text(gtex, "s", name, xs__n="name")
    _sub(gtex, "WrapMode", xs__n="wrapMode", v="CLAMP_TO_BORDER")
    fm = _sub(gtex, "FilterMode", xs__n="filterMode")
    _sub(fm, "GTexture2D", xs__n="owner", xs__ref=ref_tex2d)
    _sub(fm, "MinFilter", xs__n="minFilter", v="LINEAR_MIPMAP_LINEAR")
    _sub(fm, "MagFilter", xs__n="magFilter", v="LINEAR")
    _sub(gtex, "GTextureGuid", xs__n="guid", xs__ref=ref_tex_guid)
    _sub(gtex, "Anisotropy", xs__n="anisotropy", v="ON")
    _sub(tex2d, "CImageResource", xs__n="srcImageResource", xs__ref=ref_img)
    _sub(tex2d, "CAffine", xs__n="transformImageResource01toLogical01", **_IDENTITY_AFFINE)
    _text(tex2d, "i", "1", xs__n="mipmapLevel")
    _text(tex2d, "b", "true", xs__n="isPremultiplied")
    return ref_tex2d


def _build_texture_input_ext(sh, ref_ext_tex, ref_mi_guid) -> str:
    tie, ref_tie = sh.add("CTextureInputExtension")
    timi, ref_timi = sh.add("CTextureInput_ModelImage")
    ati = _sub(timi, "ACTextureInput", xs__n="super")
    _sub(ati, "CAffine", xs__n="optionalTransformOnCanvas", **_IDENTITY_AFFINE)
    _sub(ati, "CTextureInputExtension", xs__n="_owner", xs__ref=ref_tie)
    _sub(timi, "CModelImageGuid", xs__n="_modelImageGuid", xs__ref=ref_mi_guid)

    sup = _sub(tie, "ACExtension", xs__n="super")
    _sub(sup, "CExtensionGuid", xs__n="guid", xs__ref=ref_ext_tex)
    # _owner (the CArtMeshSource) is patched in by the caller once the mesh exists.
    inputs = _sub(tie, "carray_list", xs__n="_textureInputs", count="1")
    _sub(inputs, "CTextureInput_ModelImage", xs__ref=ref_timi)
    _sub(tie, "CTextureInput_ModelImage", xs__n="currentTextureInputData", xs__ref=ref_timi)
    return ref_tie


def _static_keyform_grid(sh, ref_form) -> str:
    """A keyform grid with a single rest form and no parameter bindings (nothing deforms it). Used by the
    root part, which carries no parameter-driven motion."""
    grid, ref_grid = sh.add("KeyformGridSource")
    kfog = _sub(grid, "array_list", xs__n="keyformsOnGrid", count="1")
    kog = _sub(kfog, "KeyformOnGrid")
    ak = _sub(kog, "KeyformGridAccessKey", xs__n="accessKey")
    _sub(ak, "array_list", xs__n="_keyOnParameterList", count="0")
    _sub(kog, "CFormGuid", xs__n="keyformGuid", xs__ref=ref_form)
    _sub(grid, "array_list", xs__n="keyformBindings", count="0")
    return ref_grid


def _build_keyform_grid(sh, affecting, keys_per, param_guid_ref, note_prefix):
    """Build a ``KeyformGridSource`` bound to ``affecting`` parameters (one binding per axis).

    Returns ``(ref_grid, cells)`` where ``cells[i] = (ref_form_guid, axis_values)`` — one entry per grid
    cell (the cartesian product of the axes' keys, param[0] fastest-varying), ``axis_values`` being the
    parameter value on each axis at that cell. The caller allocates nothing else here but owns building
    the concrete forms (``CArtMeshForm`` / ``CWarpDeformerForm`` ...) that reference the returned form
    guids. Shared by the mesh grid and the warp-deformer grid so their binding structure is identical.
    """
    total = 1
    for ks in keys_per:
        total *= len(ks)

    grid, ref_grid = sh.add("KeyformGridSource")
    # Allocate a binding per axis up front so each cell's KeyOnParameter can reference it; fill them after
    # the grid exists (a binding points back at its grid).
    binding_els = [sh.add("KeyformBindingSource") for _ in affecting]
    binding_refs = [ref for _, ref in binding_els]

    cells = []
    kfog = _sub(grid, "array_list", xs__n="keyformsOnGrid", count=str(total))
    for idx in range(total):
        rem = idx
        axis_ki, axis_vals = [], []
        for pi in range(len(affecting)):
            ki = rem % len(keys_per[pi])
            rem //= len(keys_per[pi])
            axis_ki.append(ki)
            axis_vals.append(keys_per[pi][ki])
        ref_form = sh.add("CFormGuid", uuid=_new_uuid(), note=f"{note_prefix}_c{idx}")[1]
        cells.append((ref_form, axis_vals))

        kog = _sub(kfog, "KeyformOnGrid")
        ak = _sub(kog, "KeyformGridAccessKey", xs__n="accessKey")
        kop_list = _sub(ak, "array_list", xs__n="_keyOnParameterList", count=str(len(affecting)))
        for pi in range(len(affecting)):
            kop = _sub(kop_list, "KeyOnParameter")
            _sub(kop, "KeyformBindingSource", xs__n="binding", xs__ref=binding_refs[pi])
            _text(kop, "i", str(axis_ki[pi]), xs__n="keyIndex")
        _sub(kog, "CFormGuid", xs__n="keyformGuid", xs__ref=ref_form)

    kb = _sub(grid, "array_list", xs__n="keyformBindings", count=str(len(affecting)))
    for bref in binding_refs:
        _sub(kb, "KeyformBindingSource", xs__ref=bref)

    for (el, _), p, keys in zip(binding_els, affecting, keys_per):
        _fill_kf_binding(el, ref_grid, param_guid_ref[p.id], keys, p.id)

    return ref_grid, cells


def _mesh_keyform_grid(sh, rig, part, mesh, canvas_w, canvas_h, pidx, param_guid_ref):
    """Build a mesh's ``KeyformGridSource`` bound to the parameters that drive ``part``.

    Returns ``(ref_grid, cell_forms)`` where ``cell_forms`` is one ``(ref_form_guid, positions_px,
    opacity)`` per grid cell, parallel to the grid's ``keyformsOnGrid`` — the caller emits one
    ``CArtMeshForm`` per entry. Positions are in canvas pixels (``norm * canvas``).

    Axes are ordered ascending by parameter index (identical to the moc3 grid, so both backends read the
    same cell). Each cell's form = rest pose + the summed per-vertex offset of each axis at that cell's
    key, and its opacity = the part's base opacity times any keyed opacity overrides. A part no parameter
    drives falls back to a single rest cell with no bindings.
    """
    part_id = part.id
    nv = len(mesh.vertices)
    rest_px = [(x * canvas_w, y * canvas_h) for (x, y) in mesh.vertices]

    # Params that deform this part (magnitude-capped), plus any that only key its opacity; canonical order.
    affecting = _affecting_params(rig, part_id)
    opac_only = [p for p in _opacity_params(rig, part_id) if p not in affecting]
    affecting = sorted(affecting + opac_only, key=lambda p: pidx[p.id])

    if not affecting:
        ref_form = sh.add("CFormGuid", uuid=_new_uuid(), note=f"{part_id}_rest")[1]
        return _static_keyform_grid(sh, ref_form), [(ref_form, rest_px, part.opacity)]

    keys_per = [sorted(kf.value for kf in p.keyforms) for p in affecting]
    ref_grid, cells = _build_keyform_grid(sh, affecting, keys_per, param_guid_ref, part_id)

    cell_forms = []
    for ref_form, axis_vals in cells:
        pos = [list(v) for v in mesh.vertices]        # normalized rest; offsets are normalized deltas
        op = part.opacity
        for p, val in zip(affecting, axis_vals):
            for j, (dx, dy) in enumerate(_offset_at(p, val, part_id, nv)):
                pos[j][0] += dx
                pos[j][1] += dy
            ov = _opacity_at(p, val, part_id)
            if ov is not None:
                op *= ov
        cell_forms.append((ref_form, [(x * canvas_w, y * canvas_h) for x, y in pos], op))
    return ref_grid, cell_forms


# --- deformers (Phase 3) ---------------------------------------------------------------------------
# Warp deformers get full parameter-driven keyform grids (a lattice whose control points move per
# parameter — this is how the moc3 backend does head/body turn). Rotation deformers carry no angle in the
# IRR (only a pivot), so a faithful keyform would rotate by 0° = a no-op; they are emitted as
# structurally-valid *static* deformers instead (their children still render), a documented limitation
# until the IRR grows rotation-angle keyforms. NB: our authoring pipeline currently emits ``deformers=[]``
# (turn is synthesized inside each backend), so this path is exercised by tests, not yet by live output.

def _deformer_affecting(rig, def_id, pidx):
    """Parameters whose keyforms move deformer ``def_id`` (nonzero grid-vertex delta), canonical order."""
    out = []
    for p in rig.parameters:
        mag = 0.0
        for kf in p.keyforms:
            for dx, dy in kf.deformer_offsets.get(def_id, []):
                mag += abs(dx) + abs(dy)
        if mag > 1e-9:
            out.append(p)
    return sorted(out, key=lambda p: pidx[p.id])


def _deformer_offset_at(param, value, def_id, npts):
    """Per-grid-point (dx, dy) deltas for ``param`` at keyform ``value`` (zeros if none)."""
    for kf in param.keyforms:
        if kf.value == value:
            offs = kf.deformer_offsets.get(def_id)
            if offs and len(offs) == npts:
                return offs
            break
    return [(0.0, 0.0)] * npts


def _deformer_common(sh, source_el, deformer, ref_part_root, ref_grid, ref_self_guid, ref_target):
    """The ``ACDeformerSource`` header shared by warp and rotation deformers: name, parent part, keyform
    grid, guid, id and the target (parent) deformer that this one hangs beneath."""
    acdfs = _sub(source_el, "ACDeformerSource", xs__n="super")
    pc = _sub(acdfs, "ACParameterControllableSource", xs__n="super")
    _text(pc, "s", deformer.id, xs__n="localName")
    _text(pc, "b", "true", xs__n="isVisible")
    _text(pc, "b", "false", xs__n="isLocked")
    _sub(pc, "CPartGuid", xs__n="parentGuid", xs__ref=ref_part_root)
    _sub(pc, "KeyformGridSource", xs__n="keyformGridSource", xs__ref=ref_grid)
    morph = _sub(pc, "KeyFormMorphTargetSet", xs__n="keyformMorphTargetSet")
    _sub(morph, "carray_list", xs__n="_morphTargets", count="0")
    mbw = _sub(morph, "MorphTargetBlendWeightConstraintSet", xs__n="blendWeightConstraintSet")
    _sub(mbw, "carray_list", xs__n="_constraints", count="0")
    _sub(pc, "carray_list", xs__n="_extensions", count="0")
    _sub(pc, "null", xs__n="internalColor_direct_argb")
    _sub(pc, "null", xs__n="internalColor_indirect_argb")
    _sub(acdfs, "CDeformerGuid", xs__n="guid", xs__ref=ref_self_guid)
    _sub(acdfs, "CDeformerId", xs__n="id", idstr=deformer.id)
    _sub(acdfs, "CDeformerGuid", xs__n="targetDeformerGuid", xs__ref=ref_target)


def _deformer_form_super(parent, tag, ref_form, ref_source, ref_coord):
    """The ``ACDeformerForm`` (guid, source back-ref, opacity, colours, coord) inside a deformer keyform."""
    adf = _sub(parent, "ACDeformerForm", xs__n="super")
    acf = _sub(adf, "ACForm", xs__n="super")
    _sub(acf, "CFormGuid", xs__n="guid", xs__ref=ref_form)
    _text(acf, "b", "false", xs__n="isAnimatedForm")
    _text(acf, "b", "false", xs__n="isLocalAnimatedForm")
    _sub(acf, tag, xs__n="_source", xs__ref=ref_source)
    _sub(acf, "null", xs__n="name")
    _text(acf, "s", "", xs__n="notes")
    _text(adf, "f", "1.0", xs__n="opacity")
    _sub(adf, "CFloatColor", xs__n="multiplyColor", red="1.0", green="1.0", blue="1.0", alpha="1.0")
    _sub(adf, "CFloatColor", xs__n="screenColor", red="0.0", green="0.0", blue="0.0", alpha="1.0")
    _sub(adf, "CoordType", xs__n="coordType", xs__ref=ref_coord)


def _build_warp_deformer(sh, rig, d, cw, ch, ref_coord, param_guid_ref, pidx, ref_part_root,
                         ref_self_guid, ref_target):
    """A ``CWarpDeformerSource``: a ``grid_rows``x``grid_cols`` lattice whose control points move per the
    parameters that key ``deformer_offsets[d.id]``. One ``CWarpDeformerForm`` per keyform-grid cell."""
    npts = d.grid_rows * d.grid_cols
    base = d.grid_vertices  # normalized lattice, row-major
    affecting = _deformer_affecting(rig, d.id, pidx)

    if affecting:
        keys_per = [sorted(kf.value for kf in p.keyforms) for p in affecting]
        ref_grid, cells = _build_keyform_grid(sh, affecting, keys_per, param_guid_ref, d.id)
        cell_forms = []
        for ref_form, axis_vals in cells:
            pts = [list(v) for v in base]
            for p, val in zip(affecting, axis_vals):
                for j, (dx, dy) in enumerate(_deformer_offset_at(p, val, d.id, npts)):
                    pts[j][0] += dx
                    pts[j][1] += dy
            cell_forms.append((ref_form, [(x * cw, y * ch) for x, y in pts]))
    else:
        ref_form = sh.add("CFormGuid", uuid=_new_uuid(), note=f"{d.id}_rest")[1]
        ref_grid = _static_keyform_grid(sh, ref_form)
        cell_forms = [(ref_form, [(x * cw, y * ch) for x, y in base])]

    warp, ref_src = sh.add("CWarpDeformerSource")
    _deformer_common(sh, warp, d, ref_part_root, ref_grid, ref_self_guid, ref_target)
    _text(warp, "i", str(d.grid_cols - 1), xs__n="col")  # Cubism col/row = segments = points - 1
    _text(warp, "i", str(d.grid_rows - 1), xs__n="row")
    _text(warp, "b", "false", xs__n="isQuadTransform")
    kf_list = _sub(warp, "carray_list", xs__n="keyforms", count=str(len(cell_forms)))
    for ref_form, grid_px in cell_forms:
        form = _sub(kf_list, "CWarpDeformerForm")
        _deformer_form_super(form, "CWarpDeformerSource", ref_form, ref_src, ref_coord)
        flat = [c for xy in grid_px for c in xy]
        _text(form, "float-array", " ".join(f"{v:.4f}" for v in flat), xs__n="positions",
              count=str(2 * npts))
    return "CWarpDeformerSource", ref_src


def _build_rotation_deformer(sh, d, cw, ch, ref_coord, ref_part_root, ref_self_guid, ref_target):
    """A ``CRotationDeformerSource`` emitted as a static passthrough (angle 0, origin at the pivot). The
    IRR carries no rotation angle, so this deformer does not animate yet; its children still render."""
    ref_form = sh.add("CFormGuid", uuid=_new_uuid(), note=f"{d.id}_rest")[1]
    ref_grid = _static_keyform_grid(sh, ref_form)
    rot, ref_src = sh.add("CRotationDeformerSource")
    _deformer_common(sh, rot, d, ref_part_root, ref_grid, ref_self_guid, ref_target)
    _text(rot, "b", "true", xs__n="useBoneUi_testImpl")
    ox, oy = d.pivot
    kf_list = _sub(rot, "carray_list", xs__n="keyforms", count="1")
    form = _sub(kf_list, "CRotationDeformerForm", angle="0.0", originX=f"{ox * cw:.4f}",
                originY=f"{oy * ch:.4f}", scale="1.0", isReflectX="false", isReflectY="false")
    _deformer_form_super(form, "CRotationDeformerSource", ref_form, ref_src, ref_coord)
    _text(rot, "f", "200.0", xs__n="handleLengthOnCanvas")
    _text(rot, "f", "100.0", xs__n="circleRadiusOnCanvas")
    _text(rot, "f", "0.0", xs__n="baseAngle")
    return "CRotationDeformerSource", ref_src


def _build_deformers(sh, rig, cw, ch, ref_coord, param_guid_ref, pidx, ref_part_root,
                     deformer_guid, ref_deformer_root):
    """Build every ``rig.deformers`` source. Returns ``[(tag, ref_source), ...]`` for the
    ``CDeformerSourceSet``. Each deformer targets its parent deformer (or the root deformer)."""
    out = []
    for d in rig.deformers:
        ref_self = deformer_guid[d.id]
        ref_target = deformer_guid[d.parent] if d.parent is not None else ref_deformer_root
        if d.type is DeformerType.warp:
            out.append(_build_warp_deformer(
                sh, rig, d, cw, ch, ref_coord, param_guid_ref, pidx, ref_part_root, ref_self, ref_target))
        else:
            out.append(_build_rotation_deformer(
                sh, d, cw, ch, ref_coord, ref_part_root, ref_self, ref_target))
    return out


def _fill_kf_binding(el, ref_grid, ref_param_guid, keys, description) -> None:
    """One parameter axis of a keyform grid: its keys (parameter stop values) and LINEAR interpolation."""
    _sub(el, "KeyformGridSource", xs__n="_gridSource", xs__ref=ref_grid)
    _sub(el, "CParameterGuid", xs__n="parameterGuid", xs__ref=ref_param_guid)
    keys_arr = _sub(el, "array_list", xs__n="keys", count=str(len(keys)))
    for k in keys:
        _text(keys_arr, "f", f"{k:.4f}")
    _sub(el, "InterpolationType", xs__n="interpolationType", v="LINEAR")
    _sub(el, "ExtendedInterpolationType", xs__n="extendedInterpolationType", v="LINEAR")
    _text(el, "i", "1", xs__n="insertPointCount")
    _text(el, "f", "1.0", xs__n="extendedInterpolationScale")
    _text(el, "s", description, xs__n="description")


def _build_art_mesh(
    sh, name, mesh, canvas_w, canvas_h, ref_part_root, ref_kfg_mesh, ref_ext_mesh, ref_emesh,
    ref_coord, ref_tie, ref_drawable, ref_deformer_target, cell_forms, ref_tex2d,
) -> str:
    mesh_src, ref_mesh = sh.add("CArtMeshSource")

    # Geometry: model-space [0,1] (y-down) -> canvas pixels; UVs pass straight through (v-down). The
    # top-level ``positions`` is the rest/base mesh; per-parameter deformation lives in ``cell_forms``.
    positions = [c for (x, y) in mesh.vertices for c in (x * canvas_w, y * canvas_h)]
    uvs = [c for uv in mesh.uvs for c in uv]
    n = len(mesh.vertices)
    edges = _edges_from_triangles(mesh.triangles)

    ds = _sub(mesh_src, "ACDrawableSource", xs__n="super")
    pc = _sub(ds, "ACParameterControllableSource", xs__n="super")
    _text(pc, "s", name, xs__n="localName")
    _text(pc, "b", "true", xs__n="isVisible")
    _text(pc, "b", "false", xs__n="isLocked")
    _sub(pc, "CPartGuid", xs__n="parentGuid", xs__ref=ref_part_root)
    _sub(pc, "KeyformGridSource", xs__n="keyformGridSource", xs__ref=ref_kfg_mesh)
    morph = _sub(pc, "KeyFormMorphTargetSet", xs__n="keyformMorphTargetSet")
    _sub(morph, "carray_list", xs__n="_morphTargets", count="0")
    mbw = _sub(morph, "MorphTargetBlendWeightConstraintSet", xs__n="blendWeightConstraintSet")
    _sub(mbw, "carray_list", xs__n="_constraints", count="0")

    ext_list = _sub(pc, "carray_list", xs__n="_extensions", count="3")
    eme = _sub(ext_list, "CEditableMeshExtension")
    eme_sup = _sub(eme, "ACExtension", xs__n="super")
    _sub(eme_sup, "CExtensionGuid", xs__n="guid", xs__ref=ref_ext_mesh)
    _sub(eme_sup, "CArtMeshSource", xs__n="_owner", xs__ref=ref_mesh)
    em = _sub(eme, "GEditableMesh2", xs__n="editableMesh", nextPointUid=str(n),
              useDelaunayTriangulation="false")
    _text(em, "float-array", " ".join(f"{v:.4f}" for v in positions), xs__n="point", count=str(2 * n))
    _text(em, "byte-array", " ".join(["20"] * n), xs__n="pointPriority", count=str(n))
    edge_flat = [idx for e in edges for idx in e]
    _text(em, "short-array", " ".join(str(v) for v in edge_flat), xs__n="edge", count=str(len(edge_flat)))
    _text(em, "byte-array", " ".join(["30"] * len(edges)), xs__n="edgePriority", count=str(len(edges)))
    _text(em, "int-array", " ".join(str(k) for k in range(n)), xs__n="pointUid", count=str(n))
    _sub(em, "GEditableMeshGuid", xs__n="meshGuid", xs__ref=ref_emesh)
    _sub(em, "CoordType", xs__n="coordType", xs__ref=ref_coord)
    _text(eme, "b", "false", xs__n="isLocked")

    _sub(ext_list, "CTextureInputExtension", xs__ref=ref_tie)

    mge = _sub(ext_list, "CMeshGeneratorExtension")
    mge_sup = _sub(mge, "ACExtension", xs__n="super")
    _sub(mge_sup, "CExtensionGuid", xs__n="guid", uuid=_new_uuid(), note="(no debug info)")
    _sub(mge_sup, "CArtMeshSource", xs__n="_owner", xs__ref=ref_mesh)
    mgs = _sub(mge, "MeshGenerateSetting", xs__n="meshGenerateSetting")
    for k, val in (("polygonOuterDensity", "100"), ("polygonInnerDensity", "100"),
                   ("polygonMargin", "20"), ("polygonInnerMargin", "20"), ("polygonMinMargin", "5"),
                   ("polygonMinBoundsPt", "5"), ("thresholdAlpha", "0")):
        _text(mgs, "i", val, xs__n=k)

    _sub(pc, "null", xs__n="internalColor_direct_argb")
    _sub(ds, "CDrawableId", xs__n="id", idstr=name)
    _sub(ds, "CDrawableGuid", xs__n="guid", xs__ref=ref_drawable)
    _sub(ds, "CDeformerGuid", xs__n="targetDeformerGuid", xs__ref=ref_deformer_target)
    _sub(ds, "carray_list", xs__n="clipGuidList", count="0")
    _text(ds, "b", "false", xs__n="invertClippingMask")

    indices = [idx for tri in mesh.triangles for idx in tri]
    _text(mesh_src, "int-array", " ".join(str(v) for v in indices), xs__n="indices",
          count=str(len(indices)))

    # One CArtMeshForm per keyform-grid cell: its own form guid, deformed positions and opacity.
    kf_list = _sub(mesh_src, "carray_list", xs__n="keyforms", count=str(len(cell_forms)))
    for ref_form, cell_px, cell_opacity in cell_forms:
        art_form = _sub(kf_list, "CArtMeshForm")
        adf = _sub(art_form, "ACDrawableForm", xs__n="super")
        acf = _sub(adf, "ACForm", xs__n="super")
        _sub(acf, "CFormGuid", xs__n="guid", xs__ref=ref_form)
        _text(acf, "b", "false", xs__n="isAnimatedForm")
        _text(acf, "b", "false", xs__n="isLocalAnimatedForm")
        _sub(acf, "CArtMeshSource", xs__n="_source", xs__ref=ref_mesh)
        _sub(acf, "null", xs__n="name")
        _text(acf, "s", "", xs__n="notes")
        _text(adf, "i", "500", xs__n="drawOrder")
        _text(adf, "f", f"{cell_opacity:.4f}", xs__n="opacity")
        _sub(adf, "CFloatColor", xs__n="multiplyColor", red="1.0", green="1.0", blue="1.0", alpha="1.0")
        _sub(adf, "CFloatColor", xs__n="screenColor", red="0.0", green="0.0", blue="0.0", alpha="1.0")
        _sub(adf, "CoordType", xs__n="coordType", xs__ref=ref_coord)
        flat = [c for xy in cell_px for c in xy]
        _text(art_form, "float-array", " ".join(f"{v:.4f}" for v in flat), xs__n="positions",
              count=str(2 * n))

    _text(mesh_src, "float-array", " ".join(f"{v:.4f}" for v in positions), xs__n="positions",
          count=str(2 * n))
    _text(mesh_src, "float-array", " ".join(f"{v:.6f}" for v in uvs), xs__n="uvs", count=str(2 * n))
    _sub(mesh_src, "GTexture2D", xs__n="texture", xs__ref=ref_tex2d)
    _sub(mesh_src, "ColorComposition", xs__n="colorComposition", v="NORMAL")
    _text(mesh_src, "b", "false", xs__n="culling")
    _sub(mesh_src, "TextureState", xs__n="textureState", v="MODEL_IMAGE")
    _text(mesh_src, "s", "", xs__n="userData")
    return ref_mesh


def _model_image(
    ref_mi_guid, name, ref_fset, ref_fvid_mi_guid, ref_fvid_mi_layer, ref_li_guid, ref_layer,
    ref_img, ref_img_grp, canvas_w, canvas_h,
) -> ET.Element:
    """An inline CModelImage: renders this part's CLayer from the synthetic PSD through the filter set."""
    mi = _e("CModelImage", modelImageVersion="0")
    _sub(mi, "CModelImageGuid", xs__n="guid", xs__ref=ref_mi_guid)
    _text(mi, "s", name, xs__n="name")
    _sub(mi, "ModelImageFilterSet", xs__n="inputFilter", xs__ref=ref_fset)

    mife = _sub(mi, "ModelImageFilterEnv", xs__n="inputFilterEnv")
    fe = _sub(mife, "FilterEnv", xs__n="super")
    _sub(fe, "null", xs__n="parentEnv")
    env_map = _sub(fe, "hash_map", xs__n="envValues", count="2")
    e1 = _sub(env_map, "entry")
    _sub(e1, "FilterValueId", xs__n="key", xs__ref=ref_fvid_mi_guid)
    evs1 = _sub(e1, "EnvValueSet", xs__n="value")
    _sub(evs1, "FilterValueId", xs__n="id", xs__ref=ref_fvid_mi_guid)
    _sub(evs1, "CLayeredImageGuid", xs__n="value", xs__ref=ref_li_guid)
    _text(evs1, "l", "0", xs__n="updateTimeMs")
    e2 = _sub(env_map, "entry")
    _sub(e2, "FilterValueId", xs__n="key", xs__ref=ref_fvid_mi_layer)
    evs2 = _sub(e2, "EnvValueSet", xs__n="value")
    _sub(evs2, "FilterValueId", xs__n="id", xs__ref=ref_fvid_mi_layer)
    lsm = _sub(evs2, "CLayerSelectorMap", xs__n="value")
    itli = _sub(lsm, "linked_map", xs__n="_imageToLayerInput", count="1")
    itli_e = _sub(itli, "entry")
    _sub(itli_e, "CLayeredImageGuid", xs__n="key", xs__ref=ref_li_guid)
    itli_v = _sub(itli_e, "array_list", xs__n="value", count="1")
    lid = _sub(itli_v, "CLayerInputData")
    _sub(lid, "CLayer", xs__n="layer", xs__ref=ref_layer)
    _sub(lid, "CAffine", xs__n="affine", **_IDENTITY_AFFINE)
    _sub(lid, "null", xs__n="clippingOnTexturePx")
    _text(evs2, "l", "0", xs__n="updateTimeMs")

    _sub(mi, "CImageResource", xs__n="_filteredImage", xs__ref=ref_img)
    _sub(mi, "null", xs__n="icon16")
    _sub(mi, "CAffine", xs__n="_materialLocalToCanvasTransform", **_IDENTITY_AFFINE)
    _sub(mi, "CModelImageGroup", xs__n="_group", xs__ref=ref_img_grp)
    lrig = _sub(mi, "carray_list", xs__n="linkedRawImageGuids", count="1")
    _sub(lrig, "CLayeredImageGuid", xs__ref=ref_li_guid)

    cim = _sub(mi, "CCachedImageManager", xs__n="cachedImageManager")
    _sub(cim, "CachedImageType", xs__n="defaultCacheType", v="SCALE_1")
    _sub(cim, "CImageResource", xs__n="rawImage", xs__ref=ref_img)
    ci_list = _sub(cim, "array_list", xs__n="cachedImages", count="1")
    ci = _sub(ci_list, "CCachedImage")
    _sub(ci, "CImageResource", xs__n="_cachedImageResource", xs__ref=ref_img)
    _text(ci, "b", "true", xs__n="isSharedImage")
    _sub(ci, "CSize", xs__n="rawImageSize", width=str(canvas_w), height=str(canvas_h))
    _text(ci, "i", "1", xs__n="reductionRatio")
    _text(ci, "i", "1", xs__n="mipmapLevel")
    _text(ci, "b", "false", xs__n="hasMargin")
    _text(ci, "b", "false", xs__n="isCleaned")
    _sub(ci, "CAffine", xs__n="transformRawImageToCachedImage", **_IDENTITY_AFFINE)
    _text(cim, "i", "1", xs__n="requiredMipmapLevel")
    _text(mi, "s", "", xs__n="memo")
    return mi


def _fill_model_image_group(img_group, ref_li_guid, model_image_els) -> None:
    _text(img_group, "s", "", xs__n="memo")
    _text(img_group, "s", "image2live2d_export", xs__n="groupName")
    linked = _sub(img_group, "carray_list", xs__n="_linkedRawImageGuids", count="1")
    _sub(linked, "CLayeredImageGuid", xs__ref=ref_li_guid)
    mi_list = _sub(img_group, "carray_list", xs__n="_modelImages", count=str(len(model_image_els)))
    for mi in model_image_els:
        mi_list.append(mi)


def _build_root_part(sh, ref_part_root, ref_kfg_part, ref_deformer_root, drawable_refs,
                     deformer_refs, ref_part_form) -> str:
    part_src, ref_part_src = sh.add("CPartSource")
    pc = _sub(part_src, "ACParameterControllableSource", xs__n="super")
    _text(pc, "s", "Root Part", xs__n="localName")
    _text(pc, "b", "true", xs__n="isVisible")
    _text(pc, "b", "false", xs__n="isLocked")
    _sub(pc, "null", xs__n="parentGuid")
    _sub(pc, "KeyformGridSource", xs__n="keyformGridSource", xs__ref=ref_kfg_part)
    morph = _sub(pc, "KeyFormMorphTargetSet", xs__n="keyformMorphTargetSet")
    _sub(morph, "carray_list", xs__n="_morphTargets", count="0")
    mbw = _sub(morph, "MorphTargetBlendWeightConstraintSet", xs__n="blendWeightConstraintSet")
    _sub(mbw, "carray_list", xs__n="_constraints", count="0")
    _sub(pc, "carray_list", xs__n="_extensions", count="0")
    _sub(pc, "null", xs__n="internalColor_direct_argb")
    _sub(part_src, "CPartGuid", xs__n="guid", xs__ref=ref_part_root)
    _sub(part_src, "CPartId", xs__n="id", idstr="PartRoot")
    _text(part_src, "b", "false", xs__n="enableDrawOrderGroup")
    _text(part_src, "i", "500", xs__n="defaultOrder_forEditor")
    _text(part_src, "b", "false", xs__n="isSketch")
    _sub(part_src, "CColor", xs__n="partsEditColor")
    child_guids = _sub(part_src, "carray_list", xs__n="_childGuids",
                       count=str(len(drawable_refs) + len(deformer_refs)))
    for dref in drawable_refs:
        _sub(child_guids, "CDrawableGuid", xs__ref=dref)
    for gref in deformer_refs:
        _sub(child_guids, "CDeformerGuid", xs__ref=gref)
    _sub(part_src, "CDeformerGuid", xs__n="targetDeformerGuid", xs__ref=ref_deformer_root)
    kf_list = _sub(part_src, "carray_list", xs__n="keyforms", count="1")
    part_form = _sub(kf_list, "CPartForm")
    acf = _sub(part_form, "ACForm", xs__n="super")
    _sub(acf, "CFormGuid", xs__n="guid", xs__ref=ref_part_form)
    _text(acf, "b", "false", xs__n="isAnimatedForm")
    _text(acf, "b", "false", xs__n="isLocalAnimatedForm")
    _sub(acf, "CPartSource", xs__n="_source", xs__ref=ref_part_src)  # self-reference
    _sub(acf, "null", xs__n="name")
    _text(acf, "s", "", xs__n="notes")
    _text(part_form, "i", "500", xs__n="drawOrder")
    return ref_part_src


def _build_model_source(
    main_el, rig, ref_model, canvas_w, canvas_h, ref_param_group, param_guid_ref, mesh_refs,
    ref_part_src, ref_li, ref_img_grp, deformer_sources, ref_root_pg,
) -> None:
    model = _sub(main_el, "CModelSource", isDefaultKeyformLocked="true")
    _sub(model, "CModelGuid", xs__n="guid", xs__ref=ref_model)
    _text(model, "s", rig.meta.name or "image2live2d Export", xs__n="name")
    edition = _sub(model, "EditorEdition", xs__n="editorEdition")
    _text(edition, "i", "15", xs__n="edition")

    canvas = _sub(model, "CImageCanvas", xs__n="canvas")
    _text(canvas, "i", str(canvas_w), xs__n="pixelWidth")
    _text(canvas, "i", str(canvas_h), xs__n="pixelHeight")
    _sub(canvas, "CColor", xs__n="background")

    param_set = _sub(model, "CParameterSourceSet", xs__n="parameterSourceSet")
    psources = _sub(param_set, "carray_list", xs__n="_sources", count=str(len(rig.parameters)))
    for p in rig.parameters:
        _param_source(psources, p, ref_param_group, param_guid_ref[p.id])

    tex_mgr = _sub(model, "CTextureManager", xs__n="textureManager")
    tex_list = _sub(tex_mgr, "TextureImageGroup", xs__n="textureList")
    _sub(tex_list, "carray_list", xs__n="children", count="0")
    ri = _sub(tex_mgr, "carray_list", xs__n="_rawImages", count="1")
    liw = _sub(ri, "LayeredImageWrapper")
    _sub(liw, "CLayeredImage", xs__n="image", xs__ref=ref_li)
    _text(liw, "l", "0", xs__n="importedTimeMSec")
    _text(liw, "l", "0", xs__n="lastModifiedTimeMSec")
    _text(liw, "b", "false", xs__n="isReplaced")
    mig = _sub(tex_mgr, "carray_list", xs__n="_modelImageGroups", count="1")
    _sub(mig, "CModelImageGroup", xs__ref=ref_img_grp)
    _sub(tex_mgr, "carray_list", xs__n="_textureAtlases", count="0")
    _text(tex_mgr, "b", "true", xs__n="isTextureInputModelImageMode")
    _text(tex_mgr, "i", "1", xs__n="previewReductionRatio")
    _sub(tex_mgr, "carray_list", xs__n="artPathBrushUsingLayeredImageIds", count="0")

    _text(model, "b", "false", xs__n="useLegacyDrawOrder__testImpl")
    draw_set = _sub(model, "CDrawableSourceSet", xs__n="drawableSourceSet")
    dsources = _sub(draw_set, "carray_list", xs__n="_sources", count=str(len(mesh_refs)))
    for mref in mesh_refs:
        _sub(dsources, "CArtMeshSource", xs__ref=mref)

    deformer_set = _sub(model, "CDeformerSourceSet", xs__n="deformerSourceSet")
    dsrc = _sub(deformer_set, "carray_list", xs__n="_sources", count=str(len(deformer_sources)))
    for tag, ref in deformer_sources:
        _sub(dsrc, tag, xs__ref=ref)
    affecter_set = _sub(model, "CAffecterSourceSet", xs__n="affecterSourceSet")
    _sub(affecter_set, "carray_list", xs__n="_sources", count="0")

    part_set = _sub(model, "CPartSourceSet", xs__n="partSourceSet")
    psrc = _sub(part_set, "carray_list", xs__n="_sources", count="1")
    _sub(psrc, "CPartSource", xs__ref=ref_part_src)

    # Physics (Phase 4): pendulum settings driving hair/skirt output params. Only parsed at
    # CModelSource:14, so this appears only alongside the v14 tail below (ref_root_pg is set).
    if rig.physics:
        _build_physics(model, rig, param_guid_ref)

    _sub(model, "CPartSource", xs__n="rootPart", xs__ref=ref_part_src)

    pg_set = _sub(model, "CParameterGroupSet", xs__n="parameterGroupSet")
    if ref_root_pg is not None:
        groups = _sub(pg_set, "carray_list", xs__n="_groups", count="1")
        _sub(groups, "CParameterGroup", xs__ref=ref_root_pg)
    else:
        _sub(pg_set, "carray_list", xs__n="_groups", count="0")

    if ref_root_pg is not None:  # v14 requires the rootParameterGroup pointer (entity ref, not guid)
        _sub(model, "CParameterGroup", xs__n="rootParameterGroup", xs__ref=ref_root_pg)

    mi_info = _sub(model, "CModelInfo", xs__n="modelInfo")
    _text(mi_info, "f", "1.0", xs__n="pixelsPerUnit")
    origin = _sub(mi_info, "CPoint", xs__n="originInPixels")
    _text(origin, "i", "0", xs__n="x")
    _text(origin, "i", "0", xs__n="y")
    if ref_root_pg is not None:  # v14 required inner collection
        epg = _sub(mi_info, "CEffectParameterGroups", xs__n="_effectParameterGroups")
        _sub(epg, "hash_map", xs__n="_parameterGroups", count="0", keyType="string")

    if ref_root_pg is not None:
        _v14_tail(model)

    _text(model, "i", "3000", xs__n="targetVersionNo")
    _text(model, "i", "5000000", xs__n="latestVersionOfLastModelerNo")


def _param_source(parent, param, ref_param_group, ref_guid) -> None:
    ps = _sub(parent, "CParameterSource")
    _text(ps, "i", "1", xs__n="decimalPlaces")
    _sub(ps, "CParameterGuid", xs__n="guid", xs__ref=ref_guid)
    _text(ps, "f", "0.1", xs__n="snapEpsilon")
    _text(ps, "f", f"{param.min:.4f}", xs__n="minValue")
    _text(ps, "f", f"{param.max:.4f}", xs__n="maxValue")
    _text(ps, "f", f"{param.default:.4f}", xs__n="defaultValue")
    _text(ps, "b", "false", xs__n="isRepeat")
    _sub(ps, "CParameterId", xs__n="id", idstr=param.id)
    _sub(ps, "Type", xs__n="paramType", v="NORMAL")
    _text(ps, "s", param.id, xs__n="name")
    _text(ps, "s", "", xs__n="description")
    _text(ps, "b", "false", xs__n="combined")
    _sub(ps, "CParameterGroupGuid", xs__n="parentGroupGuid", xs__ref=ref_param_group)


# --- physics + the CModelSource:14 tail (Phase 4) --------------------------------------------------

def _build_root_param_group(sh, ref_param_group_guid, param_guid_ref, rig) -> str:
    """The root ``CParameterGroup`` *entity* v14 requires (v4 got by with just the guid + an empty group
    set). A flat tree: every parameter is a direct child. Returns its shared ref."""
    group, ref = sh.add("CParameterGroup")
    _text(group, "s", "Root Parameter Group", xs__n="name")
    _text(group, "s", "", xs__n="description")
    _text(group, "b", "false", xs__n="folderIsOpened")
    _sub(group, "CParameterGroupGuid", xs__n="guid", xs__ref=ref_param_group_guid)
    _sub(group, "null", xs__n="parentGroupGuid")
    children = _sub(group, "carray_list", xs__n="_childGuids", count=str(len(rig.parameters)))
    for p in rig.parameters:
        _sub(children, "CParameterGuid", xs__ref=param_guid_ref[p.id])
    _sub(group, "CParameterGroupId", xs__n="id", idstr="ParamGroupRoot")
    for ch, val in (("Red", "1.0"), ("Green", "1.0"), ("Blue", "1.0"), ("Alpha", "1.0")):
        _text(group, "f", val, xs__n=f"visibilityColor{ch}")
    return ref


def _build_physics(model, rig, param_guid_ref) -> None:
    """Emit ``physicsSettingsSourceSet`` — one ``CPhysicsSettingsSource`` per IRR ``PhysicsRig``."""
    pset = _sub(model, "CPhysicsSettingsSourceSet", xs__n="physicsSettingsSourceSet")
    plist = _sub(pset, "carray_list", xs__n="_sourceCubismPhysics", count=str(len(rig.physics)))
    for i, ph in enumerate(rig.physics, start=1):
        _build_physics_setting(plist, ph, i, param_guid_ref)
    _sub(pset, "CPhysicsSettingsGuid", xs__n="selectedCubismPhysics", uuid=_new_uuid(), note="selection")
    _sub(pset, "null", xs__n="settingFPS")


def _build_physics_setting(plist, ph, idx, param_guid_ref) -> None:
    """One pendulum: inputs from the drivers (translate/gravity-angle, dropping inert ones), a single
    output tapping the swinging tip, the vertex chain, and the normalization ranges — all mirroring
    :mod:`..physics3` so the editor physics matches the shipped physics3.json."""
    src = _sub(plist, "CPhysicsSettingsSource")
    _text(src, "s", ph.output_param, xs__n="name")
    _sub(src, "CPhysicsSettingsGuid", xs__n="guid", uuid=_new_uuid(), note=ph.output_param)
    _sub(src, "CPhysicsSettingId", xs__n="id", idstr=f"PhysicsSetting{idx}")

    # One input per driver that can actually swing a pendulum; "X" -> SRC_TO_X (translate), "Angle" ->
    # SRC_TO_G_ANGLE (tip gravity), None -> dropped (a Y translation is a no-op for an angle output).
    inputs = [(d, "SRC_TO_X" if t == "X" else "SRC_TO_G_ANGLE")
              for d in ph.all_drivers() if (t := _input_type(d, pitch_angle=ph.pitch_angle))]
    inode = _sub(src, "carray_list", xs__n="inputs", count=str(len(inputs)))
    for d, src_type in inputs:
        inp = _sub(inode, "CPhysicsInput")
        _sub(inp, "CPhysicsDataGuid", xs__n="guid", uuid=_new_uuid(), note=f"in_{d}")
        _sub(inp, "CParameterGuid", xs__n="source", xs__ref=param_guid_ref[d])
        _text(inp, "f", "0.0", xs__n="angleScale")
        ts = _sub(inp, "GVector2", xs__n="translationScale")
        _text(ts, "f", "0.0", xs__n="x")
        _text(ts, "f", "0.0", xs__n="y")
        _text(inp, "f", f"{_INPUT_WEIGHT:.1f}", xs__n="weight")
        _sub(inp, "CPhysicsSourceType", xs__n="type", v=src_type)
        _text(inp, "b", "false", xs__n="isReverse")

    verts = _vertices(ph.mass, ph.drag, ph.length)
    onode = _sub(src, "carray_list", xs__n="outputs", count="1")
    out = _sub(onode, "CPhysicsOutput")
    _sub(out, "CPhysicsDataGuid", xs__n="guid", uuid=_new_uuid(), note=f"out_{ph.output_param}")
    _sub(out, "CParameterGuid", xs__n="destination", xs__ref=param_guid_ref[ph.output_param])
    _text(out, "i", str(len(verts) - 1), xs__n="vertexIndex")  # the swinging tip (0-based)
    ots = _sub(out, "GVector2", xs__n="translationScale")
    _text(ots, "f", "0.0", xs__n="x")
    _text(ots, "f", "0.0", xs__n="y")
    scale = 1.4 if ph.output_param.startswith("ParamHair") else 1.0
    _text(out, "f", f"{scale:.1f}", xs__n="angleScale")
    _text(out, "f", f"{_OUTPUT_WEIGHT:.1f}", xs__n="weight")
    _sub(out, "CPhysicsSourceType", xs__n="type", v="SRC_TO_G_ANGLE")
    _text(out, "b", "false", xs__n="isReverse")

    vnode = _sub(src, "carray_list", xs__n="vertices", count=str(len(verts)))
    for j, vd in enumerate(verts):
        v = _sub(vnode, "CPhysicsVertex")
        _sub(v, "CPhysicsDataGuid", xs__n="guid", uuid=_new_uuid(), note=f"v{j}")
        pos = _sub(v, "GVector2", xs__n="position")
        _text(pos, "f", f"{vd['Position']['X']:.4f}", xs__n="x")
        _text(pos, "f", f"{vd['Position']['Y']:.4f}", xs__n="y")
        _text(v, "f", f"{vd['Mobility']:.4f}", xs__n="mobility")
        _text(v, "f", f"{vd['Delay']:.4f}", xs__n="delay")
        _text(v, "f", f"{vd['Acceleration']:.4f}", xs__n="acceleration")
        _text(v, "f", f"{vd['Radius']:.4f}", xs__n="radius")

    pos_n, ang_n = _NORM["Position"], _NORM["Angle"]
    _text(src, "f", f"{pos_n['Maximum']:.1f}", xs__n="normalizedPositionValueMax")
    _text(src, "f", f"{pos_n['Minimum']:.1f}", xs__n="normalizedPositionValueMin")
    _text(src, "f", f"{pos_n['Default']:.1f}", xs__n="normalizedPositionDefaultValue")
    _text(src, "f", f"{ang_n['Maximum']:.1f}", xs__n="normalizedAngleValueMax")
    _text(src, "f", f"{ang_n['Minimum']:.1f}", xs__n="normalizedAngleValueMin")
    _text(src, "f", f"{ang_n['Default']:.1f}", xs__n="normalizedAngleDefaultValue")


_ICON_SPECS = (("_icon64", 64, "cmo3_icon_64.png"), ("_icon32", 32, "cmo3_icon_32.png"),
               ("_icon16", 16, "cmo3_icon_16.png"))


def _v14_tail(model) -> None:
    """The remaining fields the CModelSource:14 reader dereferences (or NPEs). All empty stubs except the
    preview-icon references, whose PNGs :func:`_icon_files` adds to the archive."""
    for field, size, path in _ICON_SPECS:
        icon = _sub(model, "CImageIcon", xs__n=field)
        img = _sub(icon, "CWritableImage", xs__n="image", width=str(size), height=str(size),
                   type="INT_ARGB")
        _sub(img, "file", xs__n="image", path=path)
    _sub(model, "hash_map", xs__n="modelOptions", count="0", keyType="string")
    gms = _sub(model, "CGameMotionSet", xs__n="gameMotionSet")
    _sub(gms, "carray_list", xs__n="gameMotions", count="0")
    _sub(gms, "carray_list", xs__n="gameMotionGroups", count="0")
    mvs = _sub(model, "ModelViewerSetting", xs__n="modelViewerSetting")
    _sub(mvs, "array_list", xs__n="trackCursorSettings", count="0")
    guides = _sub(model, "CGuidesSetting", xs__n="guides")
    _sub(guides, "carray_list", xs__n="guidesModeling", count="0")
    brush = _sub(model, "CArtPathBrushSetting", xs__n="artPathBrushesSetting")
    _sub(brush, "carray_list", xs__n="brushes", count="0")
    rp = _sub(model, "CRandomPoseSettingManager", xs__n="randomPoseSetting")
    _sub(rp, "array_list", xs__n="_settings", count="0")
    _text(rp, "i", "0", xs__n="currentIndex")


def _icon_files() -> list[tuple[str, bytes]]:
    """The blank preview-icon PNGs the v14 ``CImageIcon`` fields reference, for the CAFF archive."""
    return [(path, _blank_png(size, size)) for _, size, path in _ICON_SPECS]


def _blank_png(w: int, h: int) -> bytes:
    """A minimal transparent RGBA PNG (stdlib only)."""
    import struct
    import zlib

    def chunk(ctype: bytes, data: bytes) -> bytes:
        body = ctype + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
    raw = (b"\x00" + b"\x00\x00\x00\x00" * w) * h
    return b"\x89PNG\r\n\x1a\n" + ihdr + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
