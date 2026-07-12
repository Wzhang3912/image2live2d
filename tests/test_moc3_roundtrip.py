"""Route A / Phase 4B S1 — **byte-for-byte** validation of the moc3 codec against REAL official
Live2D models (Haru/Hiyori/…), the strongest possible proof the reader+writer understand the format.

Real ``.moc3`` files are copyrighted Live2D assets (and git-ignored), so they can't ship in the repo —
this test is **asset-gated**: it discovers reference models on the local machine and *skips* when none
are found. Point it at your own models with the ``MOC3_REFERENCE_MODELS`` env var (``:``-separated
files or directories); it also best-effort discovers the free Cubism samples bundled with VTube Studio.

For each real v3.00 model it asserts three things, which together mean *every byte is understood*:

  1. **Coverage** — every byte of the file is either inside a region we parse (header, offset table,
     count table, canvas, or a ``FIELDS`` array at its real offset) or is zero padding. A non-zero
     unaccounted byte would mean the reader silently drops data the runtime reads.
  2. **Re-encode** — each parsed array re-serializes byte-for-byte identical to the original bytes at
     its real offset (reader parsed it right *and* the writer's encoders match Cubism's).
  3. **Reconstruction** — rebuilding the whole file from parsed values at their original offsets
     reproduces the original bytes exactly.
"""

from __future__ import annotations

import glob
import os
import struct
from pathlib import Path

import pytest

from image2live2d.backends.live2d.moc3_binary import (
    COUNT_KEYS, FIELDS, MAGIC, TABLE_BASE, _ELEM_SIZE, _n_elems, read_moc3,
)

_END = "<"


def _discover_models() -> list[Path]:
    """Real v3.00 ``.moc3`` files to validate against, from the env var + best-effort known locations."""
    candidates: list[str] = []
    env = os.environ.get("MOC3_REFERENCE_MODELS", "")
    for entry in (p for p in env.split(os.pathsep) if p):
        candidates.append(entry) if entry.endswith(".moc3") else candidates.extend(
            glob.glob(os.path.join(entry, "**", "*.moc3"), recursive=True))
    # Best-effort: free Cubism sample models bundled with a local VTube Studio install (any OS path).
    home = str(Path.home())
    for pat in (
        f"{home}/Library/Application Support/Steam/steamapps/common/VTube Studio/**/Live2DModels/**/*.moc3",
        f"{home}/.steam/steam/steamapps/common/VTube Studio/**/Live2DModels/**/*.moc3",
        "out/moc3_research/haru.moc3",  # the RE reference kept in this repo's (git-ignored) out/ dir
    ):
        candidates.extend(glob.glob(pat, recursive=True))

    out: list[Path] = []
    seen: set[str] = set()
    for c in candidates:
        p = Path(c)
        if not p.is_file() or str(p) in seen:
            continue
        seen.add(str(p))
        try:
            head = p.read_bytes()[:6]
        except OSError:
            continue
        if head[:4] == MAGIC and head[4] == 1:  # v3.00 only (our codec's target)
            out.append(p)
    return out


_MODELS = _discover_models()


def _array_nbytes(kind: str, n: int) -> int:
    return n * 8 if kind == "rt" else n * _ELEM_SIZE[kind]


def _encode_array(kind: str, values) -> bytes:
    """Mirror of the writer's per-array serialization (kept local so the test is self-contained)."""
    if kind == "rt":
        return bytes(values) if values else b""
    if kind == "id":
        out = bytearray()
        for s in values:
            b = s.encode("ascii")[:63]
            out += b + b"\0" * (64 - len(b))
        return bytes(out)
    if kind == "u8":
        return struct.pack(f"{len(values)}B", *values)
    if kind == "s16":
        return struct.pack(_END + f"{len(values)}h", *values)
    if kind == "s32":
        return struct.pack(_END + f"{len(values)}i", *values)
    if kind == "u32":
        return struct.pack(_END + f"{len(values)}I", *values)
    if kind == "f32":
        return struct.pack(_END + f"{len(values)}f", *values)
    if kind in ("xy", "uv"):
        flat = [c for xy in values for c in xy]
        return struct.pack(_END + f"{len(flat)}f", *flat)
    raise ValueError(kind)


@pytest.mark.skipif(not _MODELS, reason="no real .moc3 reference models found "
                    "(set MOC3_REFERENCE_MODELS=path1:path2 or install VTube Studio's free samples)")
@pytest.mark.parametrize("path", _MODELS, ids=lambda p: p.stem)
def test_real_moc3_roundtrips_byte_for_byte(path: Path):
    data = path.read_bytes()
    moc = read_moc3(data)

    def u32(off: int) -> int:
        return struct.unpack_from(_END + "I", data, off)[0]

    count_off = u32(TABLE_BASE)
    canvas_off = u32(TABLE_BASE + 4)
    slot_offsets = [u32(TABLE_BASE + 4 * (2 + i)) for i in range(len(FIELDS))]

    # (1) coverage: every non-zero byte must be inside a region we account for.
    covered = bytearray(len(data))

    def mark(start: int, length: int) -> None:
        for i in range(start, min(start + length, len(data))):
            covered[i] = 1

    mark(0, 0x40)                                     # header
    mark(TABLE_BASE, 4 * (2 + len(FIELDS)))           # offset table
    mark(count_off, len(COUNT_KEYS) * 4)              # count table
    mark(canvas_off, 24)                              # canvas: 5 floats + flag byte
    for slot, (section, field, kind, ckey) in enumerate(FIELDS):
        n = _n_elems(kind, moc.counts[ckey])
        if n:
            mark(slot_offsets[slot], _array_nbytes(kind, n))
    unaccounted_nonzero = [i for i in range(len(data)) if not covered[i] and data[i]]
    assert not unaccounted_nonzero, (
        f"{path.name}: {len(unaccounted_nonzero)} non-zero bytes not covered by the field map "
        f"(reader is dropping data); first at 0x{unaccounted_nonzero[0]:x}")

    # (2) re-encode: each array must serialize identically to the original bytes in place.
    for slot, (section, field, kind, ckey) in enumerate(FIELDS):
        n = _n_elems(kind, moc.counts[ckey])
        if not n:
            continue
        off = slot_offsets[slot]
        orig = data[off: off + _array_nbytes(kind, n)]
        assert _encode_array(kind, moc.sections[section][field]) == orig, \
            f"{path.name}: {section}.{field} ({kind}) does not re-encode identically"

    # (3) full-file reconstruction from parsed values at their original offsets == the original bytes.
    recon = bytearray(len(data))
    recon[0:4] = MAGIC
    recon[4] = moc.version
    recon[5] = 1 if moc.big_endian else 0
    struct.pack_into(_END + "I", recon, TABLE_BASE, count_off)
    struct.pack_into(_END + "I", recon, TABLE_BASE + 4, canvas_off)
    for i, p in enumerate(slot_offsets):
        struct.pack_into(_END + "I", recon, TABLE_BASE + 4 * (2 + i), p)
    struct.pack_into(_END + f"{len(COUNT_KEYS)}I", recon, count_off,
                     *[moc.counts[k] for k in COUNT_KEYS])
    c = moc.canvas
    struct.pack_into(_END + "5f", recon, canvas_off,
                     c["pixelsPerUnit"], c["originX"], c["originY"], c["width"], c["height"])
    recon[canvas_off + 20] = int(c.get("flags", 0))
    for slot, (section, field, kind, ckey) in enumerate(FIELDS):
        n = _n_elems(kind, moc.counts[ckey])
        if n:
            b = _encode_array(kind, moc.sections[section][field])
            recon[slot_offsets[slot]: slot_offsets[slot] + len(b)] = b
    assert bytes(recon) == data, f"{path.name}: full-file reconstruction differs from the original"
