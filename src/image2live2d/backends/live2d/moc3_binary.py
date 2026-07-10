"""Low-level ``.moc3`` binary codec (reader + writer) — Route A / Phase 4B, stage S1.

``.moc3`` is Live2D's closed rig format. Its byte layout was reverse-engineered by the OpenL2D project
(``moc3ingbird`` / ``moc3.hexpat``, FDPL-licensed); this module implements that layout as a plain
Python read/write codec so we can (a) parse a real model and (b) serialize one back — the foundation of
generating a Live2D model from our IRR without the Cubism Editor. See docs/PHASE4B_MOC3_FEASIBILITY.md.

Design: a moc3 is a **structure-of-arrays**. A fixed section-offset table at 0x40 holds one ``u32``
absolute file offset per (section, field); each points to an array of ``count`` elements (counts live
in the Count Info Table). We model the file as ``{section: {field: [values]}}`` + counts + canvas +
header, which is enough to round-trip and (later) to author from scratch.

Scope: **moc3 v3.00** (version byte 1) — the minimal layout (no v3.3/v4.2/v5 extension sections),
still loadable by current Cubism 4/5 runtimes. v4/v5 add trailing sections we can layer on later.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field as dfield

MAGIC = b"MOC3"
V3_00 = 1
HEADER_SIZE = 0x40
TABLE_BASE = 0x40           # section offset table starts here
RUNTIME_MAP = 0x2C0         # runtime address map (overwritten at load time)
# The runtime overlays its parsed `struct psm__sections` onto the buffer starting at ~0x2C0, and it is
# larger than the hexpat's 0x480 "runtime" comment implies. Real Cubism files start data at 0x7C0, so
# we must too — starting at 0x740 gets count_info clobbered by the runtime scratch (empirically: revive
# zeroed our counts). See docs/PHASE4B_MOC3_FEASIBILITY.md S2.
DATA_BASE = 0x7C0          # first data array (matches Cubism's own layout)
MOC_ALIGN = 64             # csmAlignofMoc — Cubism requires the TOTAL moc size to be a multiple of this

# Count Info Table field order (v3.00). Index = position of the u32 count from the countInfo offset.
COUNT_KEYS = [
    "parts", "deformers", "warpDeformers", "rotationDeformers", "artMeshes", "parameters",
    "partKeyforms", "warpDeformerKeyforms", "rotationDeformerKeyforms", "artMeshKeyforms",
    "keyformPositions", "parameterBindingIndices", "keyformBindings", "parameterBindings", "keys",
    "uvs", "positionIndices", "drawableMasks", "drawOrderGroups", "drawOrderGroupObjects",
    "glue", "glueInfo", "glueKeyforms",
]

# Section-offset-table field order (v3.00). Each tuple: (section, field, kind, count_key).
# kind: id | s32 | u32 | f32 | s16 | xy | uv | rt  (rt = runtime scratch, count*8 zero bytes)
# For xy/uv the element count is count_key//2 (the count is a float count); others use count_key.
FIELDS: list[tuple[str, str, str, str]] = [
    # parts
    ("parts", "runtimeSpace0", "rt", "parts"),
    ("parts", "ids", "id", "parts"),
    ("parts", "keyformBindingSourcesIndices", "s32", "parts"),
    ("parts", "keyformSourcesBeginIndices", "s32", "parts"),
    ("parts", "keyformSourcesCounts", "s32", "parts"),
    ("parts", "isVisible", "u32", "parts"),
    ("parts", "isEnabled", "u32", "parts"),
    ("parts", "parentPartIndices", "s32", "parts"),
    # deformers
    ("deformers", "runtimeSpace0", "rt", "deformers"),
    ("deformers", "ids", "id", "deformers"),
    ("deformers", "keyformBindingSourcesIndices", "s32", "deformers"),
    ("deformers", "isVisible", "u32", "deformers"),
    ("deformers", "isEnabled", "u32", "deformers"),
    ("deformers", "parentPartIndices", "s32", "deformers"),
    ("deformers", "parentDeformerIndices", "s32", "deformers"),
    ("deformers", "types", "u32", "deformers"),
    ("deformers", "specificSourcesIndices", "s32", "deformers"),
    # warp deformers
    ("warpDeformers", "keyformBindingSourcesIndices", "s32", "warpDeformers"),
    ("warpDeformers", "keyformSourcesBeginIndices", "s32", "warpDeformers"),
    ("warpDeformers", "keyformSourcesCounts", "s32", "warpDeformers"),
    ("warpDeformers", "vertexCounts", "s32", "warpDeformers"),
    ("warpDeformers", "rows", "u32", "warpDeformers"),
    ("warpDeformers", "columns", "u32", "warpDeformers"),
    # rotation deformers
    ("rotationDeformers", "keyformBindingSourcesIndices", "s32", "rotationDeformers"),
    ("rotationDeformers", "keyformSourcesBeginIndices", "s32", "rotationDeformers"),
    ("rotationDeformers", "keyformSourcesCounts", "s32", "rotationDeformers"),
    ("rotationDeformers", "baseAngles", "f32", "rotationDeformers"),
    # art meshes
    ("artMeshes", "runtimeSpace0", "rt", "artMeshes"),
    ("artMeshes", "runtimeSpace1", "rt", "artMeshes"),
    ("artMeshes", "runtimeSpace2", "rt", "artMeshes"),
    ("artMeshes", "runtimeSpace3", "rt", "artMeshes"),
    ("artMeshes", "ids", "id", "artMeshes"),
    ("artMeshes", "keyformBindingSourcesIndices", "s32", "artMeshes"),
    ("artMeshes", "keyformSourcesBeginIndices", "s32", "artMeshes"),
    ("artMeshes", "keyformSourcesCounts", "s32", "artMeshes"),
    ("artMeshes", "isVisible", "u32", "artMeshes"),
    ("artMeshes", "isEnabled", "u32", "artMeshes"),
    ("artMeshes", "parentPartIndices", "s32", "artMeshes"),
    ("artMeshes", "parentDeformerIndices", "s32", "artMeshes"),
    ("artMeshes", "textureNos", "u32", "artMeshes"),
    ("artMeshes", "drawableFlags", "u32", "artMeshes"),
    ("artMeshes", "vertexCounts", "s32", "artMeshes"),
    ("artMeshes", "uvSourcesBeginIndices", "s32", "artMeshes"),
    ("artMeshes", "positionIndexSourcesBeginIndices", "s32", "artMeshes"),
    ("artMeshes", "positionIndexSourcesCounts", "s32", "artMeshes"),
    ("artMeshes", "drawableMaskSourcesBeginIndices", "s32", "artMeshes"),
    ("artMeshes", "drawableMaskSourcesCounts", "s32", "artMeshes"),
    # parameters
    ("parameters", "runtimeSpace0", "rt", "parameters"),
    ("parameters", "ids", "id", "parameters"),
    ("parameters", "maxValues", "f32", "parameters"),
    ("parameters", "minValues", "f32", "parameters"),
    ("parameters", "defaultValues", "f32", "parameters"),
    ("parameters", "isRepeat", "u32", "parameters"),
    ("parameters", "decimalPlaces", "u32", "parameters"),
    ("parameters", "parameterBindingSourcesBeginIndices", "s32", "parameters"),
    ("parameters", "parameterBindingSourcesCounts", "s32", "parameters"),
    # part keyforms
    ("partKeyforms", "drawOrders", "f32", "partKeyforms"),
    # warp deformer keyforms
    ("warpDeformerKeyforms", "opacities", "f32", "warpDeformerKeyforms"),
    ("warpDeformerKeyforms", "keyformPositionSourcesBeginIndices", "s32", "warpDeformerKeyforms"),
    # rotation deformer keyforms
    ("rotationDeformerKeyforms", "opacities", "f32", "rotationDeformerKeyforms"),
    ("rotationDeformerKeyforms", "angles", "f32", "rotationDeformerKeyforms"),
    ("rotationDeformerKeyforms", "originX", "f32", "rotationDeformerKeyforms"),
    ("rotationDeformerKeyforms", "originY", "f32", "rotationDeformerKeyforms"),
    ("rotationDeformerKeyforms", "scales", "f32", "rotationDeformerKeyforms"),
    ("rotationDeformerKeyforms", "isReflectX", "u32", "rotationDeformerKeyforms"),
    ("rotationDeformerKeyforms", "isReflectY", "u32", "rotationDeformerKeyforms"),
    # art mesh keyforms
    ("artMeshKeyforms", "opacities", "f32", "artMeshKeyforms"),
    ("artMeshKeyforms", "drawOrders", "f32", "artMeshKeyforms"),
    ("artMeshKeyforms", "keyformPositionSourcesBeginIndices", "s32", "artMeshKeyforms"),
    # keyform positions (XY pairs: element count = keyformPositions // 2)
    ("keyformPositions", "xys", "xy", "keyformPositions"),
    # parameter binding indices
    ("parameterBindingIndices", "bindingSourcesIndices", "s32", "parameterBindingIndices"),
    # keyform bindings
    ("keyformBindings", "parameterBindingIndexSourcesBeginIndices", "s32", "keyformBindings"),
    ("keyformBindings", "parameterBindingIndexSourcesCounts", "s32", "keyformBindings"),
    # parameter bindings
    ("parameterBindings", "keysSourcesBeginIndices", "s32", "parameterBindings"),
    ("parameterBindings", "keysSourcesCounts", "s32", "parameterBindings"),
    # keys
    ("keys", "values", "f32", "keys"),
    # uvs (UV pairs: element count = uvs // 2)
    ("uvs", "uvs", "uv", "uvs"),
    # position indices
    ("positionIndices", "indices", "s16", "positionIndices"),
    # drawable masks
    ("drawableMasks", "artMeshSourcesIndices", "s32", "drawableMasks"),
    # draw order groups
    ("drawOrderGroups", "objectSourcesBeginIndices", "s32", "drawOrderGroups"),
    ("drawOrderGroups", "objectSourcesCounts", "s32", "drawOrderGroups"),
    ("drawOrderGroups", "objectSourcesTotalCounts", "s32", "drawOrderGroups"),
    ("drawOrderGroups", "maximumDrawOrders", "u32", "drawOrderGroups"),
    ("drawOrderGroups", "minimumDrawOrders", "u32", "drawOrderGroups"),
    # draw order group objects
    ("drawOrderGroupObjects", "types", "u32", "drawOrderGroupObjects"),
    ("drawOrderGroupObjects", "indices", "s32", "drawOrderGroupObjects"),
    ("drawOrderGroupObjects", "selfIndices", "s32", "drawOrderGroupObjects"),
    # glue
    ("glue", "runtimeSpace0", "rt", "glue"),
    ("glue", "ids", "id", "glue"),
    ("glue", "keyformBindingSourcesIndices", "s32", "glue"),
    ("glue", "keyformSourcesBeginIndices", "s32", "glue"),
    ("glue", "keyformSourcesCounts", "s32", "glue"),
    ("glue", "artMeshIndicesA", "s32", "glue"),
    ("glue", "artMeshIndicesB", "s32", "glue"),
    ("glue", "glueInfoSourcesBeginIndices", "s32", "glue"),
    ("glue", "glueInfoSourcesCounts", "s32", "glue"),
    # glue info
    ("glueInfo", "weights", "f32", "glueInfo"),
    ("glueInfo", "positionIndices", "s16", "glueInfo"),
    # glue keyforms
    ("glueKeyforms", "intensities", "f32", "glueKeyforms"),
]

_ELEM_SIZE = {"id": 64, "s32": 4, "u32": 4, "f32": 4, "s16": 2, "xy": 8, "uv": 8}


def _n_elems(kind: str, count: int) -> int:
    return count // 2 if kind in ("xy", "uv") else count


@dataclass
class Moc3:
    version: int = V3_00
    big_endian: bool = False
    canvas: dict = dfield(default_factory=dict)          # pixelsPerUnit, originX/Y, width, height, flags
    counts: dict = dfield(default_factory=dict)          # {count_key: int}
    sections: dict = dfield(default_factory=dict)        # {section: {field: [values]}}

    def get(self, section: str, field: str):
        return self.sections.get(section, {}).get(field, [])


# --------------------------------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------------------------------
def read_moc3(data: bytes) -> Moc3:
    if data[:4] != MAGIC:
        raise ValueError(f"not a moc3 (magic={data[:4]!r})")
    version = data[4]
    big_endian = bool(data[5])
    end = ">" if big_endian else "<"
    if version != V3_00:
        raise NotImplementedError(f"moc3 codec currently targets v3.00 (=1); file is version {version}")

    def u32(off: int) -> int:
        return struct.unpack_from(end + "I", data, off)[0]

    count_off = u32(TABLE_BASE)
    canvas_off = u32(TABLE_BASE + 4)
    counts = {k: u32(count_off + 4 * i) for i, k in enumerate(COUNT_KEYS)}

    ppu, ox, oy, cw, ch = struct.unpack_from(end + "5f", data, canvas_off)
    flags = data[canvas_off + 20]
    canvas = {"pixelsPerUnit": ppu, "originX": ox, "originY": oy,
              "width": cw, "height": ch, "flags": flags}

    sections: dict = {}
    for slot, (section, field, kind, ckey) in enumerate(FIELDS):
        ptr = u32(TABLE_BASE + 4 * (2 + slot))       # +2: countInfo, canvasInfo come first
        count = counts[ckey]
        n = _n_elems(kind, count)
        sec = sections.setdefault(section, {})
        if n == 0:
            sec[field] = b"" if kind == "rt" else []
            continue
        sec[field] = _read_array(data, ptr, kind, n, end)
    return Moc3(version=version, big_endian=big_endian, canvas=canvas, counts=counts, sections=sections)


def _read_array(data: bytes, ptr: int, kind: str, n: int, end: str):
    if kind == "rt":
        return data[ptr:ptr + n * 8]
    if kind == "id":
        return [data[ptr + 64 * i: ptr + 64 * i + 64].split(b"\0", 1)[0].decode("ascii", "replace")
                for i in range(n)]
    if kind == "s16":
        return list(struct.unpack_from(end + f"{n}h", data, ptr))
    if kind == "s32":
        return list(struct.unpack_from(end + f"{n}i", data, ptr))
    if kind == "u32":
        return list(struct.unpack_from(end + f"{n}I", data, ptr))
    if kind == "f32":
        return list(struct.unpack_from(end + f"{n}f", data, ptr))
    if kind in ("xy", "uv"):
        flat = struct.unpack_from(end + f"{2 * n}f", data, ptr)
        return [(flat[2 * i], flat[2 * i + 1]) for i in range(n)]
    raise ValueError(kind)


# --------------------------------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------------------------------
def write_moc3(moc: Moc3) -> bytes:
    if moc.version != V3_00:
        raise NotImplementedError("writer targets v3.00")
    end = ">" if moc.big_endian else "<"
    buf = bytearray()

    # header (64B)
    buf += MAGIC + bytes([moc.version, 1 if moc.big_endian else 0]) + b"\0" * 58
    # reserve section offset table (0x40..0x2c0) + runtime map (0x2c0..0x740)
    n_slots = 2 + len(FIELDS)
    buf += b"\0" * (DATA_BASE - len(buf))
    assert len(buf) == DATA_BASE

    ptrs: list[int] = []

    def align(a: int = 8):   # PurismCore/Cubism require every section 8-byte aligned
        while len(buf) % a:
            buf.append(0)

    def emit_array(kind: str, values) -> int:
        align(8)
        off = len(buf)
        if kind == "rt":
            buf.extend(values if values else b"")
        elif kind == "id":
            for s in values:
                b = s.encode("ascii")[:63]
                buf.extend(b + b"\0" * (64 - len(b)))
        elif kind == "s16":
            buf.extend(struct.pack(end + f"{len(values)}h", *values))
        elif kind == "s32":
            buf.extend(struct.pack(end + f"{len(values)}i", *values))
        elif kind == "u32":
            buf.extend(struct.pack(end + f"{len(values)}I", *values))
        elif kind == "f32":
            buf.extend(struct.pack(end + f"{len(values)}f", *values))
        elif kind in ("xy", "uv"):
            flat = [c for xy in values for c in xy]
            buf.extend(struct.pack(end + f"{len(flat)}f", *flat))
        else:
            raise ValueError(kind)
        return off

    # count info + canvas
    align(8)
    count_off = len(buf)
    buf.extend(struct.pack(end + f"{len(COUNT_KEYS)}I", *[moc.counts[k] for k in COUNT_KEYS]))
    buf.extend(b"\0" * (128 - len(COUNT_KEYS) * 4))       # count table region padded to 128B
    align(8)
    canvas_off = len(buf)
    buf.extend(struct.pack(end + "5f", moc.canvas["pixelsPerUnit"], moc.canvas["originX"],
                           moc.canvas["originY"], moc.canvas["width"], moc.canvas["height"]))
    buf.append(int(moc.canvas.get("flags", 0)))
    buf.extend(b"\0" * 43)

    # data arrays (in table order — layout need not match Cubism's; validity is what matters)
    for section, field, kind, ckey in FIELDS:
        values = moc.sections.get(section, {}).get(field, b"" if kind == "rt" else [])
        n = _n_elems(kind, moc.counts[ckey])
        if n == 0:
            align(8)                       # empty sections still need a monotonic, aligned offset
            ptrs.append(len(buf))          # (PurismCore rejects off=0 after a later section)
            continue
        ptrs.append(emit_array(kind, values))

    # backfill the offset table
    struct.pack_into(end + "I", buf, TABLE_BASE, count_off)
    struct.pack_into(end + "I", buf, TABLE_BASE + 4, canvas_off)
    for i, p in enumerate(ptrs):
        struct.pack_into(end + "I", buf, TABLE_BASE + 4 * (2 + i), p)

    # Cubism's csmReviveMocInPlace requires the TOTAL moc size to be a multiple of csmAlignofMoc (64)
    # or it rejects the file with `"size" is invalid` — our 8-aligned section packing doesn't guarantee
    # this (it's size-dependent: some models happen to land on 64, others don't). Pad the tail to 64.
    # (PurismCore doesn't enforce this, so a file can pass moc3info yet be rejected by real Cubism Core.)
    while len(buf) % MOC_ALIGN:
        buf.append(0)
    return bytes(buf)
