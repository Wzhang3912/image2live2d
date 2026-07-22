"""Build a `.moc3` (v3.00) from geometry + parameters — Route A / Phase 4B, stage S2.

This is the **authoring** side that the S1 codec (`moc3_binary`) round-trips. It assembles the moc3
structure-of-arrays for a **deformer-free** model: art meshes whose vertex positions are driven
directly by parameter keyforms (our IRR already bakes motion as per-vertex keyforms, so no warp/
rotation deformers are needed). Blend shapes, glue, and colour keyforms are omitted (v3.00 minimal).

The binding chain (reverse-engineered from the Haru sample):
  parameter ─owns─▶ parameterBinding (its key values in `keys`)
  drawable  ─▶ keyformBinding ─▶ parameterBindingIndices ─▶ parameterBindings   (which params drive it)
  drawable  ─▶ its run of keyforms; keyformCount == ∏ (key counts of its params)  (1 if none)
  each keyform ─▶ a slice of `keyformPositions` (vertexCount XY)

Index-unit conventions (from Haru): `keysSourcesBeginIndices` in key units; `positionIndex...Begin` in
index units; `uvSourcesBeginIndices` and (assumed, pending runtime confirmation) `keyformPosition
SourcesBeginIndices` in **float-component units** (= 2 × vertex offset). See KEYFORM_POS_UNIT below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field

from .moc3_binary import COUNT_KEYS, FIELDS, Moc3, V3_00

# keyformPositionSourcesBeginIndices are in FLOAT-COMPONENT units (2× the pair index) — CONFIRMED by
# decoding the real Haru sample: its begin[0]=35568 overflows the pair array (34664) but is a valid
# float index (÷2 = 17784 pairs). Same convention as uvSourcesBeginIndices.
KEYFORM_POS_UNIT = 2   # 2 = float-component units, 1 = XY-pair units

# Each keyform's position block is padded with (0,0) up to a multiple of KEYFORM_ALIGN_PAIRS pairs
# (= 64 bytes, Cubism's SIMD alignment) — VERIFIED against Haru: mesh#0 (54 verts) and mesh#1 (51 verts)
# BOTH stride 56 pairs = align8(vc); 285 art-mesh + 317 warp keyform strides matched with 0 exceptions,
# and total keyformPositions == Σ align8(vc)·keyformCount exactly. The native Cubism runtime computes a
# keyform's position address as keyform_index × align8(vertexCount) (SIMD stride) and IGNORES the stored
# begin index; tightly-packed keyforms therefore render assembled in the web Core (which reads the stored
# begins) but SCATTER in the native Viewer/VTube Studio — exactly the core/Viewer split we chased.
KEYFORM_ALIGN_PAIRS = 8


def _pad_keyform(keyform_pos: list) -> None:
    """Pad the shared keyform-position list up to the next KEYFORM_ALIGN_PAIRS boundary with (0,0)."""
    while len(keyform_pos) % KEYFORM_ALIGN_PAIRS:
        keyform_pos.append((0.0, 0.0))


@dataclass
class EmitMesh:
    id: str
    part_index: int
    texture_no: int
    uvs: list[tuple[float, float]]
    triangles: list[tuple[int, int, int]]
    param_indices: list[int]                       # which parameters drive this mesh (indices into params)
    keyforms: list[list[tuple[float, float]]]      # vertex positions per grid keyform (len == ∏ key counts, or 1)
    opacities: list[float] = dc_field(default_factory=list)  # absolute opacity per grid keyform (parallel to
    #                                                       keyforms; empty -> fully opaque everywhere)


@dataclass
class EmitParam:
    id: str
    min: float
    max: float
    default: float
    keys: list[float]                              # the parameter values at which keyforms are defined


@dataclass
class EmitPart:
    id: str
    draw_order: float = 500.0


@dataclass
class EmitWarp:
    """A warp (grid) deformer. Children (art meshes whose ids are in ``child_ids``) are re-parented to
    it and warped by its deformed grid — the real Live2D way to turn a head without tearing the neck."""
    id: str
    parent_part_index: int
    param_indices: list[int]                       # driving params (e.g. ParamAngleX/Y/Z)
    rows: int
    columns: int
    keyforms: list[list[tuple[float, float]]]      # grid control points per grid keyform ((rows+1)*(cols+1) pts)
    child_ids: set                                 # mesh ids re-parented under this deformer


def _empty_sections() -> dict:
    return {section: {} for section, _, _, _ in FIELDS}


def build_moc3(canvas: dict, params: list[EmitParam], parts: list[EmitPart],
               meshes: list[EmitMesh], warps: list["EmitWarp"] | None = None) -> Moc3:
    warps = warps or []
    counts = {k: 0 for k in COUNT_KEYS}
    S = _empty_sections()

    # ---- keys + parameterBindings (one binding per parameter) ----------------------------------
    keys: list[float] = []
    pb_keys_begin: list[int] = []
    pb_keys_count: list[int] = []
    for p in params:
        pb_keys_begin.append(len(keys))
        pb_keys_count.append(len(p.keys))
        keys.extend(p.keys)
    S["keys"]["values"] = keys
    S["parameterBindings"]["keysSourcesBeginIndices"] = pb_keys_begin
    S["parameterBindings"]["keysSourcesCounts"] = pb_keys_count

    # ---- parameters ---------------------------------------------------------------------------
    S["parameters"] = {
        "runtimeSpace0": b"\0" * (8 * len(params)),
        "ids": [p.id for p in params],
        "maxValues": [p.max for p in params],
        "minValues": [p.min for p in params],
        "defaultValues": [p.default for p in params],
        "isRepeat": [0] * len(params),
        "decimalPlaces": [4] * len(params),
        "parameterBindingSourcesBeginIndices": list(range(len(params))),  # param i -> binding i
        "parameterBindingSourcesCounts": [1] * len(params),
    }

    # ---- keyformBindings (+ parameterBindingIndices) : one binding per drawable ----------------
    pbi: list[int] = []                                        # parameterBindingIndices (flat)
    kb_begin: list[int] = []
    kb_count: list[int] = []

    def add_binding(param_idx: list[int]) -> int:
        kb_begin.append(len(pbi))
        kb_count.append(len(param_idx))
        pbi.extend(param_idx)                                  # binding index == parameter index
        return len(kb_begin) - 1

    part_bindings = [add_binding([]) for _ in parts]           # parts: empty binding -> 1 keyform each

    # ---- parts + partKeyforms -----------------------------------------------------------------
    S["parts"] = {
        "runtimeSpace0": b"\0" * (8 * len(parts)),
        "ids": [pt.id for pt in parts],
        "keyformBindingSourcesIndices": part_bindings,
        "keyformSourcesBeginIndices": list(range(len(parts))),
        "keyformSourcesCounts": [1] * len(parts),
        "isVisible": [1] * len(parts),
        "isEnabled": [1] * len(parts),
        "parentPartIndices": [-1] * len(parts),
    }
    S["partKeyforms"]["drawOrders"] = [pt.draw_order for pt in parts]

    # ---- art meshes + keyforms + geometry -----------------------------------------------------
    uvs_flat: list[tuple[float, float]] = []
    pos_indices: list[int] = []
    keyform_pos: list[tuple[float, float]] = []               # concatenated XY across all keyforms
    amk_opacity: list[float] = []
    amk_draworder: list[float] = []
    amk_pos_begin: list[int] = []

    am = {k: [] for k in (
        "ids", "keyformBindingSourcesIndices", "keyformSourcesBeginIndices", "keyformSourcesCounts",
        "isVisible", "isEnabled", "parentPartIndices", "parentDeformerIndices", "textureNos",
        "drawableFlags", "vertexCounts", "uvSourcesBeginIndices", "positionIndexSourcesBeginIndices",
        "positionIndexSourcesCounts", "drawableMaskSourcesBeginIndices", "drawableMaskSourcesCounts")}

    for m in meshes:
        vc = len(m.uvs)
        binding = add_binding(m.param_indices)
        am["ids"].append(m.id)
        am["keyformBindingSourcesIndices"].append(binding)
        am["keyformSourcesBeginIndices"].append(len(amk_opacity))
        am["keyformSourcesCounts"].append(len(m.keyforms))
        am["isVisible"].append(1)
        am["isEnabled"].append(1)
        am["parentPartIndices"].append(m.part_index)
        am["parentDeformerIndices"].append(-1)
        am["textureNos"].append(m.texture_no)
        # csmIsDoubleSided (1<<2): render both faces so meshes show regardless of triangle winding. Without
        # it, a runtime with back-face culling on (e.g. Cubism Viewer) drops any reverse-wound mesh -> the
        # whole model renders blank while our own (cull-disabled) renderer still shows it.
        am["drawableFlags"].append(1 << 2)
        am["vertexCounts"].append(vc)
        am["uvSourcesBeginIndices"].append(2 * len(uvs_flat))          # float-component units
        am["positionIndexSourcesBeginIndices"].append(len(pos_indices))
        am["positionIndexSourcesCounts"].append(3 * len(m.triangles))
        am["drawableMaskSourcesBeginIndices"].append(0)
        am["drawableMaskSourcesCounts"].append(0)
        uvs_flat.extend(m.uvs)
        for tri in m.triangles:
            pos_indices.extend(tri)
        for ki, kf in enumerate(m.keyforms):                           # one keyform per grid point
            # Per-keyform opacity: a drawable can fade in/out across a parameter axis (e.g. a synthesised
            # closed-eye lash line that appears only as ParamEyeOpen -> 0). ``opacities`` is parallel to
            # ``keyforms``; absent (the common case) -> fully opaque, exactly the old behaviour.
            amk_opacity.append(m.opacities[ki] if m.opacities else 1.0)
            amk_draworder.append(parts[m.part_index].draw_order)
            amk_pos_begin.append(KEYFORM_POS_UNIT * len(keyform_pos))   # block start is 8-pair aligned
            keyform_pos.extend(kf)
            _pad_keyform(keyform_pos)                                   # 64-byte SIMD alignment (native)

    # ---- warp deformers (grid) : head-turn via a real deformer -------------------------------------
    # Children set parentDeformerIndices -> the deformer; the deformer grid deforms per param combo and
    # Cubism warps the children through it. Grid control points live in the shared keyformPositions.
    def_ids: list[str] = []
    def_kb: list[int] = []
    def_parent_part: list[int] = []
    def_parent_def: list[int] = []
    def_type: list[int] = []
    def_specific: list[int] = []
    wd_kb: list[int] = []
    wd_kf_begin: list[int] = []
    wd_kf_count: list[int] = []
    wd_vcount: list[int] = []
    wd_rows: list[int] = []
    wd_cols: list[int] = []
    wdk_opacity: list[float] = []
    wdk_pos_begin: list[int] = []
    mesh_index = {m.id: i for i, m in enumerate(meshes)}

    for w in warps:
        binding = add_binding(w.param_indices)
        def_index = len(def_ids)
        def_ids.append(w.id)
        def_kb.append(binding)
        def_parent_part.append(w.parent_part_index)
        def_parent_def.append(-1)                       # root deformer
        def_type.append(0)                              # 0 = warp
        def_specific.append(len(wd_kb))                 # index into warpDeformers
        wd_kb.append(binding)
        wd_kf_begin.append(len(wdk_opacity))
        wd_kf_count.append(len(w.keyforms))
        wd_vcount.append((w.rows + 1) * (w.columns + 1))
        wd_rows.append(w.rows)
        wd_cols.append(w.columns)
        for grid in w.keyforms:                         # one grid per param-combo keyform
            wdk_opacity.append(1.0)
            wdk_pos_begin.append(KEYFORM_POS_UNIT * len(keyform_pos))   # block start is 8-pair aligned
            keyform_pos.extend(grid)
            _pad_keyform(keyform_pos)                    # 64-byte SIMD alignment (native runtime stride)
        for cid in w.child_ids:                         # re-parent children under this deformer
            mi = mesh_index.get(cid)
            if mi is not None:
                am["parentDeformerIndices"][mi] = def_index

    # runtime spaces (4 for art meshes) are zeroed scratch
    for rs in ("runtimeSpace0", "runtimeSpace1", "runtimeSpace2", "runtimeSpace3"):
        S["artMeshes"][rs] = b"\0" * (8 * len(meshes))
    S["artMeshes"].update(am)

    if warps:
        S["deformers"] = {
            "runtimeSpace0": b"\0" * (8 * len(def_ids)), "ids": def_ids,
            "keyformBindingSourcesIndices": def_kb, "isVisible": [1] * len(def_ids),
            "isEnabled": [1] * len(def_ids), "parentPartIndices": def_parent_part,
            "parentDeformerIndices": def_parent_def, "types": def_type,
            "specificSourcesIndices": def_specific}
        S["warpDeformers"] = {
            "keyformBindingSourcesIndices": wd_kb, "keyformSourcesBeginIndices": wd_kf_begin,
            "keyformSourcesCounts": wd_kf_count, "vertexCounts": wd_vcount,
            "rows": wd_rows, "columns": wd_cols}
        S["warpDeformerKeyforms"] = {"opacities": wdk_opacity,
                                     "keyformPositionSourcesBeginIndices": wdk_pos_begin}

    S["artMeshKeyforms"] = {"opacities": amk_opacity, "drawOrders": amk_draworder,
                            "keyformPositionSourcesBeginIndices": amk_pos_begin}
    S["keyformPositions"]["xys"] = keyform_pos
    S["uvs"]["uvs"] = uvs_flat
    S["positionIndices"]["indices"] = pos_indices
    S["parameterBindingIndices"]["bindingSourcesIndices"] = pbi
    S["keyformBindings"]["parameterBindingIndexSourcesBeginIndices"] = kb_begin
    S["keyformBindings"]["parameterBindingIndexSourcesCounts"] = kb_count

    # ---- draw order group: one group listing every art mesh (required by Cubism Core's consistency
    #      check for render ordering; art meshes are already in draw order, index 0..n-1) -----------
    nm = len(meshes)
    if nm:
        draw_orders = [int(pt.draw_order) for pt in parts] or [500]
        S["drawOrderGroups"] = {
            "objectSourcesBeginIndices": [0], "objectSourcesCounts": [nm],
            "objectSourcesTotalCounts": [nm],
            "maximumDrawOrders": [max(max(draw_orders) + 1, 1000)],
            "minimumDrawOrders": [min(min(draw_orders), 0)]}
        S["drawOrderGroupObjects"] = {
            "types": [0] * nm, "indices": list(range(nm)), "selfIndices": [-1] * nm}

    # ---- counts (float-count fields are 2× the pair count) ------------------------------------
    counts.update({
        "parts": len(parts), "artMeshes": len(meshes), "parameters": len(params),
        "partKeyforms": len(parts), "artMeshKeyforms": len(amk_opacity),
        "keyformPositions": 2 * len(keyform_pos), "parameterBindingIndices": len(pbi),
        "keyformBindings": len(kb_begin), "parameterBindings": len(params), "keys": len(keys),
        "uvs": 2 * len(uvs_flat), "positionIndices": len(pos_indices),
        "drawOrderGroups": 1 if len(meshes) else 0, "drawOrderGroupObjects": len(meshes),
        "deformers": len(def_ids), "warpDeformers": len(wd_kb),
        "warpDeformerKeyforms": len(wdk_opacity),
    })

    # fill any unset field with an empty default so the writer is happy
    for section, field, kind, _ in FIELDS:
        S[section].setdefault(field, b"" if kind == "rt" else [])

    return Moc3(version=V3_00, big_endian=False, canvas=canvas, counts=counts, sections=S)


def default_canvas(width: float = 2.0, height: float = 2.0, ppu: float = 100.0) -> dict:
    """A centered y-up canvas matching our normalized model space."""
    return {"pixelsPerUnit": ppu, "originX": width / 2, "originY": height / 2,
            "width": width, "height": height, "flags": 0}


# --------------------------------------------------------------------------------------------------
# Full IRR Rig -> .moc3  (S3)
# --------------------------------------------------------------------------------------------------
# Cap on how many parameters may drive one art mesh. Keyforms per mesh = product of the driving
# params' key counts (cartesian grid), so this bounds file size; excess (smallest-magnitude) params are
# dropped and logged. 6 params x 3 keys = 729 keyforms/mesh — comfortably enough for our rigs.
MAX_PARAMS_PER_MESH = 6


def _affecting_params(rig, part_id, forced=()):
    """Parameters whose keyforms move ``part_id`` (nonzero per-vertex delta), with a magnitude score
    so we can cap the least-important ones. Unlike the nijilive/preview path, moc3 is deformer-free, so
    we bake EVERY parameter's per-vertex offsets into keyforms (head/body turns included).

    ``forced`` ids are always included first (even with zero additive offset) — used for group-rotation
    params that drive a baked directional head/body shift rather than per-vertex offsets, so their key
    axes must exist in the mesh's keyform grid."""
    forced_ids = [fid for fid in forced if any(p.id == fid for p in rig.parameters)]
    forced_params = [p for p in rig.parameters if p.id in forced_ids]
    out = []
    for p in rig.parameters:
        if p.id in forced_ids:
            continue
        mag = 0.0
        for kf in p.keyforms:
            for dx, dy in kf.mesh_offsets.get(part_id, []):
                mag += abs(dx) + abs(dy)
        if mag > 1e-9:
            out.append((p, mag))
    out.sort(key=lambda pm: -pm[1])                       # strongest first
    return (forced_params + [p for p, _ in out])[:MAX_PARAMS_PER_MESH]


def _offset_at(param, value, part_id, nverts):
    """Per-vertex (dx, dy) deltas for ``param`` at keyform ``value`` (zeros if none)."""
    for kf in param.keyforms:
        if kf.value == value:
            offs = kf.mesh_offsets.get(part_id)
            if offs and len(offs) == nverts:
                return offs
            break
    return [(0.0, 0.0)] * nverts


def _opacity_at(param, value, part_id):
    """Absolute opacity override for ``part_id`` at ``param``'s keyform ``value``, or ``None`` if this
    param does not key the part's opacity there (then the part keeps its base opacity along this axis)."""
    for kf in param.keyforms:
        if kf.value == value:
            return kf.opacity_overrides.get(part_id)
    return None


def _opacity_params(rig, part_id):
    """Parameters that key ``part_id``'s opacity (via ``opacity_overrides``). These must join the mesh's
    keyform grid even with zero per-vertex offset, or the opacity fade has no axis to vary along."""
    out = []
    for p in rig.parameters:
        if any(part_id in kf.opacity_overrides for kf in p.keyforms):
            out.append(p)
    return out


def rig_to_moc3(rig, *, log=lambda m: None, atlas_uv=None):
    """Emit a complete v3.00 ``.moc3`` from an IRR ``Rig`` (deformer-free; every parameter baked into
    art-mesh keyforms). Model space: our normalized y-up [0,1] is mapped to Cubism's centered space
    ``(x-0.5, 0.5-y)`` (Cubism drawable Y is down relative to our y-up)."""
    drawn = [p for p in rig.parts_in_draw_order() if rig.mesh_for(p.id) is not None]
    tex_ids = []
    for p in drawn:
        if p.texture_id not in tex_ids:
            tex_ids.append(p.texture_id)

    params = [EmitParam(p.id, p.min, p.max, p.default,
                        sorted(kf.value for kf in p.keyforms)) for p in rig.parameters]
    pidx = {p.id: i for i, p in enumerate(rig.parameters)}
    pmap = {p.id: p for p in rig.parameters}
    parts = [EmitPart(p.id, float(p.draw_order)) for p in drawn]

    # HEAD-TURN via a real WARP DEFORMER (the Live2D-correct way, built after the mesh loop below): a
    # grid over the head+neck region whose head rows translate/roll with ParamAngleX/Y/Z while the
    # shoulder rows stay and the neck rows interpolate. Cubism warps the child drawables (head + neck)
    # through it -> smooth turn, the neck stretches, nothing tears, and the face never shears. Art meshes
    # keep their own (eye/mouth/face-feature) keyforms in the deformer's local space.
    from ..nijilive.puppet import head_group_ids  # noqa: PLC0415  (avoid top import cycle)
    from ...core.rig.head_rigidity import PROTECT, regions_from, rigidity_field  # noqa: PLC0415
    from ...irr.schema import SemanticRole as _SR  # noqa: PLC0415
    _dm = [(p, rig.mesh_for(p.id)) for p in drawn]
    head_ids = head_group_ids(_dm)
    neck_ids = {p.id for p, _ in _dm if p.semantic_role is _SR.neck}
    # Head turn = pseudo-3D SQUASH + roll ABOUT THE NECK BASE (the nijilive / inochi2d model), NOT a
    # translation. Yaw squashes the head horizontally, pitch squashes it vertically, roll tilts it — all
    # about the neck-base pivot, so the head stays ANCHORED at the neck and the neck itself never moves or
    # stretches. (Translating the head, the old approach, dragged the pinned neck into a rubber stretch.)
    # Magnitudes match the nijilive head-group rotation (backends/nijilive/puppet.py _HEAD_ROT).
    HEAD_YAW = 0.52            # radians of pseudo-3D yaw (horizontal squash) at full ParamAngleX
    HEAD_PITCH = 0.42          # radians of pseudo-3D pitch (vertical squash) at full ParamAngleY
    HEAD_ROLL = 0.35           # radians of in-plane roll at full ParamAngleZ
    # The head turns as a rigid unit (translate + roll) exactly like the niji runtime. No neck "lift" is
    # applied: an earlier version raised the head on turn to expose the neck, but that stretched the neck
    # and shrank the head vs the .inp/niji render, so it's removed — the head stays full size through turns.

    def _norm_frac(pid, val):
        p = pmap.get(pid)
        if p is None:
            return 0.0
        return val / max(abs(p.min), abs(p.max), 1e-6)

    # Cubism vertex positions are in UNITS (not pixels), y-UP (head +y, feet -y), centered at the origin.
    # Real .moc3 models keep vertices in a small unit range (~±1) with pixelsPerUnit ≈ the art pixel size;
    # official runtimes (Cubism Viewer, VTube Studio, the SDK) frame their default camera on that ±1
    # range. Emitting ±500-unit vertices (the old scale) left the model 1000× outside the default camera,
    # so the Cubism Viewer rendered blank even though the data was valid. Map normalized [0,1] -> ±1 unit.
    # Y is flipped (0.5 - y) because IRR space is y-DOWN (head at y=0) while Cubism is y-UP.
    CANVAS_PX = 1000.0          # canvas SIZE in pixels (metadata; with ppu below -> MODEL_SPAN units)
    MODEL_SPAN = 2.0            # model spans MODEL_SPAN units across the canvas (~±1) — standard scale

    def to_moc(x, y):
        return ((x - 0.5) * MODEL_SPAN, (0.5 - y) * MODEL_SPAN)

    # ---- head WARP DEFORMER grid, computed UP FRONT ------------------------------------------------
    # KEY: Cubism stores a warp deformer's CHILD vertices in the grid's normalized [0,1] space (verified
    # against Haru), while the grid control points are in model space. So children map (u,v)->grid; at
    # rest (identity grid = to_moc of the rest lattice) that reproduces to_moc(x,y) exactly, and other
    # keyforms warp them. We build the grid so the head rows translate/roll and the shoulder row stays.
    warps = []
    turn_ids = [pid for pid in ("ParamAngleX", "ParamAngleY", "ParamAngleZ") if pid in pmap]
    use_warp = bool(head_ids and turn_ids)
    warp_child_conv = {}                                   # mesh id -> (emitter x,y) -> grid-local [0,1]
    fixed_turn_ids = (set(head_ids) | neck_ids) if use_warp else set()  # parts whose turn is NOT baked

    if use_warp:
        # ONE head-group deformer (face + ALL hair + eyes + accessories), driven by a pseudo-3D SQUASH+roll
        # about the NECK BASE — the nijilive / inochi2d head model. The head stays anchored at the neck, so
        # the NECK NEVER MOVES (it is not a child of any turn deformer and its turn keyforms aren't baked).
        # Earlier we translated the head, which dragged the pinned neck into a rubber stretch and pulled the
        # lower hair with it. Squash-about-pivot keeps the whole head (hair included) turning in place.
        _hair = {_SR.hair_front, _SR.hair_side, _SR.hair_back, _SR.accessory}
        _fxs = [v[0] for p, m in _dm if p.id in head_ids and p.semantic_role not in _hair for v in m.vertices]
        _fys = [v[1] for p, m in _dm if p.id in head_ids and p.semantic_role not in _hair for v in m.vertices]
        _fref = sum(_fys) / len(_fys) if _fys else 0.5
        # The head DOME (face parts only — long hair would inflate it): centre + x-radius of the ellipsoid
        # the turn happens on. Used to recover each grid point's DEPTH, which is what a turn needs and a
        # squash lacks. The vertical semi-axis (_sry, below) is set separately so the dome reaches the
        # neck. Falls back to the head-group bbox when there are no face parts.
        if _fxs and _fys:
            _sx, _sy = (min(_fxs) + max(_fxs)) / 2.0, (min(_fys) + max(_fys)) / 2.0
            _srad = max((max(_fxs) - min(_fxs)) / 2.0, 1e-6)
        else:
            _sx = _sy = _srad = None
        _bys = [v[1] for p, m in _dm if p.id not in head_ids and p.id not in neck_ids for v in m.vertices]
        _bref = sum(_bys) / len(_bys) if _bys else _fref + 1.0
        _dir = 1.0 if _bref >= _fref else -1.0                          # +1 if body is at larger y
        # Pivot = the base of the FACE toward the body (the neck junction). Computed from face parts,
        # NOT the whole head group: floor-length hair (a gown veil, drill-curls to the feet) drags the
        # head-group bbox all the way down, so a group-based pivot sits at the hair TIP near the feet.
        # The yaw/pitch SQUASH is about the face-dome centre (_sx) so it survived that, but the ROLL is a
        # rotation about THIS pivot — and rotating the head about a point by its feet slides it clean off
        # the neck (seen at the roll extreme on floor-length-hair characters). The neck is the base of the
        # FACE, which long hair can't move. Fall back to the head group only when there are no face parts.
        _hxs = [v[0] for p, m in _dm if p.id in head_ids for v in m.vertices]
        _hys = [v[1] for p, m in _dm if p.id in head_ids for v in m.vertices]
        _pys = _fys if _fys else _hys
        _pivot = (_sx if _sx is not None else (sum(_hxs) / len(_hxs) if _hxs else 0.5),
                  (max(_pys) if _dir > 0 else min(_pys)) if _pys else 0.5)
        # Depth model = a VERTICALLY-ELONGATED ellipsoid, not a sphere (RIVAL_HARVEST_BACKLOG T6). A
        # sphere puts its top/bottom poles inside the face (a face is taller than it is wide), so those
        # rows get ~0 depth and PITCH pinches them together — a vertical squash, not a nod (measured: top
        # sweeps +0.057, chin -0.032, opposite ways). Stretching the vertical semi-axis to reach the NECK
        # makes depth ~constant down each column through the face (a cylinder there → the column nods as a
        # rigid unit), while the pole still lands at the neck so the neck keeps its ~0-depth anchor.
        _sry = max(abs(_pivot[1] - _sy), _srad) if _srad is not None else None
        _kp = [sorted(kf.value for kf in pmap[pid].keyforms) for pid in turn_ids]
        _tot = 1
        for _k in _kp:
            _tot *= len(_k)

        def _emit_warp(warp_id, part_ids, parent_idx):
            verts = [v for p, m in _dm if p.id in part_ids for v in m.vertices]
            if not verts:
                return
            xs = [v[0] for v in verts]
            ys = [v[1] for v in verts]
            bx0, bx1, by0, by1 = min(xs), max(xs), min(ys), max(ys)
            mx = (bx1 - bx0) * 0.12 + 1e-3   # margin: children stay inside
            my = (by1 - by0) * 0.12 + 1e-3
            bx0 -= mx
            bx1 += mx
            by0 -= my
            by1 += my
            bw = (bx1 - bx0) or 1e-6
            bh = (by1 - by0) or 1e-6
            ROWS = COLS = 12                                             # fine grid -> smooth squash
            rest = [(bx0 + bw * c / COLS, by0 + bh * r / ROWS)
                    for r in range(ROWS + 1) for c in range(COLS + 1)]   # row-major, r outer / c inner
            # Protected-region rigidity (RIVAL_HARVEST_BACKLOG T5): keep the eyes (and, partly, the
            # nose/mouth) from foreshortening with the head. Field is constant across keyforms — it's the
            # rest grid's geometry — so compute it once. Feature bboxes come from this warp's own children.
            _rbb: dict[str, tuple[float, float, float, float]] = {}
            for _p, _m in _dm:
                if _p.id in part_ids and _p.semantic_role.value in PROTECT:
                    _rxs = [v[0] for v in _m.vertices]
                    _rys = [v[1] for v in _m.vertices]
                    if _rxs:
                        _rbb[_p.semantic_role.value] = (min(_rxs), min(_rys), max(_rxs), max(_rys))
            rigid = rigidity_field(rest, regions_from(_rbb))
            gkfs = []
            for _idx in range(_tot):
                _rem = _idx
                fr = {}
                for _pi, _pid in enumerate(turn_ids):
                    _ki = _rem % len(_kp[_pi])
                    _rem //= len(_kp[_pi])
                    fr[_pid] = _norm_frac(_pid, _kp[_pi][_ki])
                yaw = fr.get("ParamAngleX", 0.0) * HEAD_YAW
                pitch = fr.get("ParamAngleY", 0.0) * HEAD_PITCH
                roll = fr.get("ParamAngleZ", 0.0) * HEAD_ROLL
                cyaw, cpit = math.cos(yaw), math.cos(pitch)             # pseudo-3D squash factors
                syaw, spit = math.sin(yaw), math.sin(pitch)
                cr, sr = math.cos(roll), math.sin(roll)

                def _squash(px_, py_):
                    # A point at depth z rotating about the head axis moves x' = x·cos(a) + z·sin(a).
                    # We had the cos term and not the sin one, which is a pure horizontal SCALE — so the
                    # two eyes drifted APART toward the centre line instead of sweeping together, and a
                    # full ±30° yaw read as the face getting narrower (2.8% of model width, vs 17% for
                    # roll). Recover z from the head dome and add the missing term.
                    #
                    # This also anchors the neck for free, which is why translating used to stretch it:
                    # the neck junction sits at the dome's bottom pole where z≈0, so it gets no sweep on
                    # its own, while the face (z≈radius) gets the full one. No taper needed — the dome
                    # already says where the head is deep and where it is edge-on.
                    if _srad is None:
                        z = 0.0
                    else:
                        # Ellipsoid depth: x-radius _srad, vertical semi-axis _sry (reaches the neck).
                        _q = 1.0 - ((px_ - _sx) / _srad) ** 2 - ((py_ - _sy) / _sry) ** 2
                        z = _srad * math.sqrt(_q) if _q > 0.0 else 0.0  # 0 outside the dome (hair, edges)
                    return (_sx + (px_ - _sx) * cyaw + z * syaw,
                            _sy + (py_ - _sy) * cpit + z * spit)

                grid = []
                for (px_, py_), (w, ccx, ccy) in zip(rest, rigid):
                    tx, ty = _squash(px_, py_)
                    if w > 0.0:
                        # Rigid target: translate the point by the feature centroid's OWN squash
                        # displacement, so the whole feature shifts as one and never narrows (T5). Blend
                        # by w — eyes 1.0 (fully rigid), nose/mouth 0.30. Roll below is exempt (a rotation
                        # preserves shape). At yaw=pitch=0 the squash is identity, so rest is untouched.
                        rtx, rty = _squash(ccx, ccy)
                        tx = tx * (1.0 - w) + (px_ + (rtx - ccx)) * w
                        ty = ty * (1.0 - w) + (py_ + (rty - ccy)) * w
                    dx = tx - _pivot[0]
                    dy = ty - _pivot[1]
                    rx = _pivot[0] + dx * cr - dy * sr
                    ry = _pivot[1] + dx * sr + dy * cr
                    grid.append(to_moc(rx, ry))
                gkfs.append(grid)
            warps.append(EmitWarp(id=warp_id, parent_part_index=parent_idx,
                                  param_indices=[pidx[pid] for pid in turn_ids],
                                  rows=ROWS, columns=COLS, keyforms=gkfs, child_ids=set(part_ids)))
            for cid in part_ids:
                warp_child_conv[cid] = (lambda x, y, _bx0=bx0, _by0=by0, _bw=bw, _bh=bh:
                                        ((x - _bx0) / _bw, (y - _by0) / _bh))

        _hpi = next((i for i, p in enumerate(drawn) if p.id in head_ids), 0)
        _emit_warp("D_HEAD", set(head_ids), _hpi)      # head + hair + eyes turn as one; neck stays fixed

    meshes = []
    for part_index, part in enumerate(drawn):
        mesh = rig.mesh_for(part.id)
        nv = len(mesh.vertices)
        is_child = use_warp and part.id in warp_child_conv   # child of a warp -> grid-local [0,1] coords
        affecting = _affecting_params(rig, part.id)       # face-feature keyforms (head turn = deformer)
        if part.id in fixed_turn_ids:
            # Head parts get the turn from the D_HEAD warp; the neck must stay FIXED under head-turn (it is
            # not a turn deformer child). Either way, do NOT bake ParamAngleX/Y/Z into this mesh's keyforms:
            # for head parts that would double-apply the turn (scale/shear), and for the neck it would make
            # it move when it shouldn't. Keep only the mesh's OTHER params (blink, mouth, brows, gaze, ...).
            affecting = [p for p in affecting if p.id not in turn_ids]
        # Lay out the keyform grid + binding in CANONICAL (ascending param-index) order. Selection above
        # is by magnitude (keeps the strongest params under the cap); the ORDER here must be canonical
        # because the native Cubism Viewer indexes a drawable's keyform grid by ascending parameter order,
        # NOT the order params are listed in the binding. Emitting a magnitude-scrambled order transposes
        # the grid axes for multi-param meshes (arms/legs, 6 params) -> they read the wrong keyform cell
        # and scatter in Cubism Viewer, while the web Cubism Core (which follows the listed order) still
        # renders them assembled — that core/Viewer split is exactly what this ordering bug looks like.
        dropped = sum(1 for p in rig.parameters
                      if any(any(dx or dy for dx, dy in kf.mesh_offsets.get(part.id, []))
                             for kf in p.keyforms)) - len(affecting)
        if dropped > 0:
            log(f"{part.id}: capped to {MAX_PARAMS_PER_MESH} params ({dropped} weaker dropped)")
        # Params that key this part's OPACITY (fade in/out) must also get a grid axis, even with no
        # per-vertex offset — otherwise the opacity has no parameter to vary along. Union with the
        # offset-driven params, then lay the whole grid out in canonical (ascending param-index) order.
        opac = [p for p in _opacity_params(rig, part.id) if p not in affecting]
        affecting = sorted(affecting + opac, key=lambda p: pidx[p.id])
        keys_per = [sorted(kf.value for kf in p.keyforms) for p in affecting]

        # cartesian grid, param[0] fastest-varying (matches PurismCore index_stride convention)
        total = 1
        for ks in keys_per:
            total *= len(ks)
        keyforms = []
        opacities = []
        keyed_opacity = False                             # did any keyform actually override opacity?
        for idx in range(total):
            pos = [list(v) for v in mesh.vertices]        # rest (our space)
            op = part.opacity                             # base; each keying param multiplies its factor
            rem = idx
            for pi, p in enumerate(affecting):
                ki = rem % len(keys_per[pi])
                rem //= len(keys_per[pi])
                for j, (dx, dy) in enumerate(_offset_at(p, keys_per[pi][ki], part.id, nv)):
                    pos[j][0] += dx
                    pos[j][1] += dy
                ov = _opacity_at(p, keys_per[pi][ki], part.id)
                if ov is not None:
                    op *= ov
                    keyed_opacity = True
            conv = warp_child_conv[part.id] if is_child else to_moc
            keyforms.append([conv(x, y) for x, y in pos])
            opacities.append(op)

        if atlas_uv is not None and part.id in atlas_uv:
            r = atlas_uv[part.id]
            uvs = [_remap_uv(u, v, r) for u, v in mesh.uvs]   # remap into the shared atlas cell
            texture_no = 0                                     # single atlas texture
        else:
            uvs = [(u, v) for u, v in mesh.uvs]
            texture_no = tex_ids.index(part.texture_id)
        # UVs are emitted RAW (v-down, v=0 at the top) to match the atlas: build_atlas (PIL) packs part
        # content into the TOP of the atlas and produces v-down cell coords, so a part's content lives at
        # low v. The real Cubism Viewer samples textures top-origin, so raw UVs land on the content.
        # DETERMINISTIC CHECK (renderer-independent): the atlas opaque region and the emitted UV v-range
        # must OVERLAP. A `(u, 1-v)` flip pushed UVs to v∈[0.87,1.0] while content sits at v∈[0.0,0.13];
        # they don't overlap → every part samples the empty bottom → the whole model renders blank in
        # Cubism Viewer. (An earlier headless "oracle" flipped V the opposite way from the real Viewer and
        # sent us in a circle — trust the atlas-vs-UV overlap, not that oracle.)
        meshes.append(EmitMesh(
            id=part.id, part_index=part_index, texture_no=texture_no,
            uvs=uvs, triangles=[tuple(t) for t in mesh.triangles],
            param_indices=[pidx[p.id] for p in affecting], keyforms=keyforms,
            # Only carry opacities when a keyform actually overrode this part's opacity; otherwise leave
            # empty so the emitter falls back to fully-opaque, exactly the pre-opacity behaviour. (Keyed by
            # the override actually firing, NOT by whether the driving param was opacity-only — an eye-lid
            # part is driven by ParamEyeOpen for BOTH its collapse and its fade, so ``opac`` is empty for
            # it yet its opacity very much varies.)
            opacities=opacities if keyed_opacity else []))

    # Canvas info drives the official runtime's default camera. Vertices span MODEL_SPAN units; the canvas
    # must span the same in units: canvas_units = width_px / pixelsPerUnit == MODEL_SPAN. With
    # width_px = CANVAS_PX that means pixelsPerUnit = CANVAS_PX / MODEL_SPAN. Origin at the pixel centre
    # (originX/ppu == MODEL_SPAN/2 units) so the centered model sits in the middle of the canvas.
    canvas = {"pixelsPerUnit": CANVAS_PX / MODEL_SPAN, "originX": CANVAS_PX / 2, "originY": CANVAS_PX / 2,
              "width": CANVAS_PX, "height": CANVAS_PX, "flags": 0}
    return build_moc3(canvas, params, parts, meshes, warps=warps)


def build_atlas(rig, asset_root, *, atlas_size=4096, pad=2):
    """Pack every drawable part's texture (tight-cropped to its content) into ONE atlas PNG and return
    ``(atlas_image, uv_remap)`` where ``uv_remap[part_id]`` maps that part's full-canvas UVs into its
    atlas cell. Real Live2D models use a shared atlas (not one texture per part); this makes our models
    render in standard web runtimes (pixi-live2d-display) and is smaller/faster. Requires Pillow/numpy.

    ``uv_remap[part_id] = (cx0, cy0, cx1, cy1, ax0, ay0, ax1, ay1)`` — source content rect and atlas
    cell, both normalized, v-down (matching mesh UV convention)."""
    import numpy as np
    from PIL import Image
    from pathlib import Path as _P

    root = _P(asset_root)
    drawn = [p for p in rig.parts_in_draw_order() if rig.mesh_for(p.id) is not None]
    tex_of = {t.id: t for t in rig.textures}

    # 1) load each part image, compute tight content bbox (pixels), collect crops
    crops = []   # (part_id, PIL crop, cx0,cy0,cx1,cy1 normalized v-down)
    for p in drawn:
        tex = tex_of[p.texture_id]
        img = Image.open(root / tex.path).convert("RGBA")
        W, H = img.size
        a = np.array(img)[:, :, 3]
        ys, xs = np.where(a > 8)
        if len(xs) == 0:
            x0, y0, x1, y1 = 0, 0, W - 1, H - 1
        else:
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        crop = img.crop((x0, y0, x1 + 1, y1 + 1))
        crops.append([p.id, crop, x0 / W, y0 / H, (x1 + 1) / W, (y1 + 1) / H])

    # 2) shelf-pack (tallest first); scale everything down uniformly if it overflows the atlas
    order = sorted(range(len(crops)), key=lambda i: -crops[i][1].height)
    def try_pack(scale):
        placements = {}
        x = pad
        y = pad
        row_h = 0
        for i in order:
            cw = max(1, int(crops[i][1].width * scale))
            ch = max(1, int(crops[i][1].height * scale))
            if x + cw + pad > atlas_size:
                x = pad
                y += row_h + pad
                row_h = 0
            if y + ch + pad > atlas_size:
                return None
            placements[i] = (x, y, cw, ch)
            x += cw + pad
            row_h = max(row_h, ch)
        return placements
    scale = 1.0
    placements = try_pack(scale)
    while placements is None and scale > 0.1:
        scale *= 0.8
        placements = try_pack(scale)
    if placements is None:
        raise ValueError("atlas packing failed (too many/large parts)")

    # 3) composite + build uv remap
    atlas = Image.new("RGBA", (atlas_size, atlas_size), (0, 0, 0, 0))
    uv_remap = {}
    for i, (pid, crop, cx0, cy0, cx1, cy1) in enumerate(crops):
        x, y, cw, ch = placements[i]
        rc = crop.resize((cw, ch), Image.LANCZOS) if (cw, ch) != crop.size else crop
        atlas.alpha_composite(rc, (x, y))
        uv_remap[pid] = (cx0, cy0, cx1, cy1,
                         x / atlas_size, y / atlas_size, (x + cw) / atlas_size, (y + ch) / atlas_size)
    return atlas, uv_remap


def _remap_uv(u, v, r):
    cx0, cy0, cx1, cy1, ax0, ay0, ax1, ay1 = r
    du = (u - cx0) / (cx1 - cx0) if cx1 > cx0 else 0.0
    dv = (v - cy0) / (cy1 - cy0) if cy1 > cy0 else 0.0
    return (ax0 + du * (ax1 - ax0), ay0 + dv * (ay1 - ay0))


def native_moc_writer(rig, template_path=None):
    """A ``MocWriter`` (see ``moc3.py``) that generates a ``.moc3`` from the IRR **from scratch** — no
    Cubism template needed. Inject into ``Live2DEmitter(moc_writer=native_moc_writer)`` to produce a
    complete, renderable Live2D bundle. ``template_path`` is ignored (accepted for the seam's signature)."""
    from .moc3_binary import write_moc3
    return write_moc3(rig_to_moc3(rig))
