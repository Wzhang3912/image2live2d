"""CAFF — the Cubism Archive File Format container that wraps a ``.cmo3`` project.

A ``.cmo3`` (an *editable* Cubism Editor project, as opposed to the runtime ``.moc3``) is a CAFF
archive: a custom big-endian binary container bundling one ``main.xml`` (the whole model graph) plus
the model's PNG textures. This module reads and writes that container; the ``main.xml`` payload itself
is built elsewhere (that is the bulk of the work — see the cmo3 package).

The container is deliberately simple and needs no proprietary tooling:

* **No real cryptography.** Integers can be XOR-"obfuscated" with a key that is stored *in plaintext* in
  the header and is allowed to be ``0`` (no obfuscation at all). The XOR is applied at the integer level
  (a 32-bit key XORed against each int, sign-extended for 64-bit), not per byte — see ``_Codec``.
* **No archive checksum.** The only integrity marker is two guard bytes ``[98, 99]`` at EOF; individual
  PNG/ZIP payloads carry their own CRCs.
* Entries are stored ``RAW`` (verbatim) or ``FAST``/``SMALL`` (the content wrapped in a ZIP archive
  holding a single member named ``contents``).

Byte layout is the reverse-engineered CAFF format (big-endian throughout). We verify our writer by
round-tripping through our own reader and by byte-comparing against a known-good reference archive.
"""

from __future__ import annotations

import io
import struct
import zipfile
from dataclasses import dataclass, field

MAGIC = b"CAFF"
FORMAT_ID = b"----"                 # four dashes = the cmo3 format within CAFF
GUARD = bytes([98, 99])             # end-of-archive marker (the only container-level integrity check)

COMPRESS_RAW = 16                   # stored verbatim
COMPRESS_FAST = 33                  # ZIP-wrapped, deflate (a member named "contents")
COMPRESS_SMALL = 37                 # ZIP-wrapped, higher compression (same container shape as FAST)

NO_PREVIEW = 127                    # preview image-format / colour-type sentinel = "none"
_ZIP_MEMBER = "contents"            # the single member name inside a FAST/SMALL entry's ZIP


def _int64_mask(key: int) -> int:
    """The 64-bit XOR mask derived from the 32-bit ``key`` (low word = key, high word sign-extends a
    negative key) — matches the editor's ``CreateInt64Mask``."""
    lower = key & 0xFFFFFFFF
    upper = 0xFFFFFFFF if key < 0 else lower
    return ((upper << 32) | lower) & 0xFFFFFFFFFFFFFFFF


class _Writer:
    """Big-endian binary writer with integer-level XOR obfuscation and back-patching."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def __len__(self) -> int:
        return len(self._buf)

    def byte(self, v: int, key: int = 0) -> None:
        self._buf.append((v ^ key) & 0xFF)

    def raw(self, data: bytes, key: int = 0) -> None:
        if key:
            k = key & 0xFF
            self._buf.extend(b ^ k for b in data)
        else:
            self._buf.extend(data)

    def zeros(self, n: int) -> None:
        self._buf.extend(b"\x00" * n)

    def int16(self, v: int, key: int = 0) -> None:
        self._buf.extend(struct.pack(">H", (v ^ (key & 0xFFFF)) & 0xFFFF))

    def int32(self, v: int, key: int = 0) -> None:
        self._buf.extend(struct.pack(">I", (v ^ key) & 0xFFFFFFFF))

    def int64(self, v: int, key: int = 0) -> None:
        self._buf.extend(struct.pack(">Q", (v ^ _int64_mask(key)) & 0xFFFFFFFFFFFFFFFF))

    def varint(self, v: int, key: int = 0) -> None:
        """1-4 byte variable-length int, 7 bits/byte, high bit = continuation, big-endian order."""
        if v < 0 or v >= (1 << 28):
            raise ValueError(f"varint out of range: {v}")
        shifts = [s for s in (21, 14, 7) if v >> s]
        for s in shifts:
            self.byte(((v >> s) & 0x7F) | 0x80, key)
        self.byte(v & 0x7F, key)

    def string(self, s: str, key: int = 0) -> None:
        b = s.encode("utf-8")
        self.varint(len(b), key)
        self.raw(b, key)

    def patch_int64(self, at: int, v: int, key: int = 0) -> None:
        struct.pack_into(">Q", self._buf, at, (v ^ _int64_mask(key)) & 0xFFFFFFFFFFFFFFFF)

    def bytes(self) -> bytes:
        return bytes(self._buf)


class _Reader:
    """Mirror of ``_Writer`` for round-trip verification and reading existing archives."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def _take(self, n: int) -> bytes:
        b = self.data[self.pos:self.pos + n]
        self.pos += n
        return b

    def skip(self, n: int) -> None:
        self.pos += n

    def byte(self, key: int = 0) -> int:
        return (self._take(1)[0] ^ key) & 0xFF

    def raw(self, n: int, key: int = 0) -> bytes:
        b = self._take(n)
        return bytes(x ^ (key & 0xFF) for x in b) if key else b

    def int16(self, key: int = 0) -> int:
        return (struct.unpack(">H", self._take(2))[0] ^ (key & 0xFFFF)) & 0xFFFF

    def int32(self, key: int = 0) -> int:
        return (struct.unpack(">I", self._take(4))[0] ^ key) & 0xFFFFFFFF

    def int64(self, key: int = 0) -> int:
        return (struct.unpack(">Q", self._take(8))[0] ^ _int64_mask(key)) & 0xFFFFFFFFFFFFFFFF

    def varint(self, key: int = 0) -> int:
        v = 0
        for _ in range(4):
            b = self.byte(key)
            v = (v << 7) | (b & 0x7F)
            if not (b & 0x80):
                return v
        raise ValueError("varint too long")

    def string(self, key: int = 0) -> str:
        n = self.varint(key)
        return self.raw(n, key).decode("utf-8") if n > 0 else ""


@dataclass
class CaffEntry:
    """One archived file: its in-archive ``path``, decompressed ``content``, and how it is stored.

    ``tag`` labels the model payload (``"main_xml"``); textures use ``""``. ``obfuscated`` XORs the stored
    bytes with the archive key; ``compress`` is one of ``COMPRESS_RAW`` / ``COMPRESS_FAST`` /
    ``COMPRESS_SMALL``."""

    path: str
    content: bytes
    tag: str = ""
    obfuscated: bool = True
    compress: int = COMPRESS_RAW
    _stored: bytes = field(default=b"", repr=False, compare=False)


def _zip_wrap(content: bytes, *, small: bool) -> bytes:
    """Wrap ``content`` as a ZIP holding a single ``contents`` member (the FAST/SMALL storage form)."""
    buf = io.BytesIO()
    level = 9 if small else 6
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=level) as zf:
        zf.writestr(_ZIP_MEMBER, content)
    return buf.getvalue()


def _zip_unwrap(stored: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(stored)) as zf:
        return zf.read(_ZIP_MEMBER)


def pack_caff(entries: list[CaffEntry], *, key: int = 0) -> bytes:
    """Serialise ``entries`` into a complete CAFF (``.cmo3``) archive.

    ``key`` is the int32 XOR obfuscation key written in plaintext into the header (``0`` = no
    obfuscation). Entry order is preserved. Start positions are back-patched after the payloads are laid
    down, exactly as the editor writes them.
    """
    w = _Writer()

    # --- header (never obfuscated) ---
    w.raw(MAGIC)
    w.zeros(3)                       # archive version [0,0,0]
    w.raw(FORMAT_ID)
    w.zeros(3)                       # format version [0,0,0]
    w.int32(key)                     # obfuscation key, in plaintext
    w.zeros(8)                       # reserved

    # --- preview image: none ---
    w.byte(NO_PREVIEW)               # image format
    w.byte(NO_PREVIEW)               # colour type
    w.zeros(2)                       # padding
    w.int16(0)                       # width
    w.int16(0)                       # height
    w.int64(0)                       # preview start position (0 = none)
    w.int32(0)                       # preview size
    w.zeros(8)                       # reserved

    # --- file table (obfuscated from here on) ---
    w.int32(len(entries), key)

    for e in entries:
        e._stored = e.content if e.compress == COMPRESS_RAW else _zip_wrap(
            e.content, small=e.compress == COMPRESS_SMALL)

    start_addrs: list[int] = []
    for e in entries:
        w.string(e.path, key)
        w.string(e.tag, key)
        start_addrs.append(len(w))
        w.int64(0, key)              # placeholder start position, patched below
        w.int32(len(e._stored), key)
        w.byte(1 if e.obfuscated else 0, key)
        w.byte(e.compress, key)
        w.zeros(8)                   # per-entry reserved

    # --- payloads, recording real start offsets ---
    starts: list[int] = []
    for e in entries:
        starts.append(len(w))
        w.raw(e._stored, key if e.obfuscated else 0)

    w.raw(GUARD)

    for addr, start in zip(start_addrs, starts):
        w.patch_int64(addr, start, key)

    return w.bytes()


def unpack_caff(data: bytes) -> list[CaffEntry]:
    """Parse a CAFF archive back into its entries (content decompressed). Raises ``ValueError`` if the
    magic or guard bytes are wrong. Used to round-trip-verify ``pack_caff`` and to read existing files."""
    if data[:4] != MAGIC:
        raise ValueError(f"not a CAFF archive: {data[:4]!r}")
    if data[-2:] != GUARD:
        raise ValueError(f"missing CAFF guard bytes: {data[-2:]!r}")

    r = _Reader(data)
    r.skip(4)                        # magic
    r.skip(3)                        # archive version
    r.skip(4)                        # format id
    r.skip(3)                        # format version
    key = r.int32()
    if key >= 0x80000000:            # header key is a signed int32
        key -= 0x100000000
    r.skip(8)                        # reserved
    r.skip(2)                        # preview format + colour type
    r.skip(2)                        # padding
    r.int16()                        # preview width
    r.int16()                        # preview height
    r.int64()                        # preview start
    r.int32()                        # preview size
    r.skip(8)                        # reserved

    count = r.int32(key)
    heads = []
    for _ in range(count):
        path = r.string(key)
        tag = r.string(key)
        start = r.int64(key)
        size = r.int32(key)
        obf = r.byte(key) != 0
        compress = r.byte(key)
        r.skip(8)
        heads.append((path, tag, start, size, obf, compress))

    out: list[CaffEntry] = []
    for path, tag, start, size, obf, compress in heads:
        r.pos = start
        stored = r.raw(size, key if obf else 0)
        content = stored if compress == COMPRESS_RAW else _zip_unwrap(stored)
        out.append(CaffEntry(path=path, content=content, tag=tag, obfuscated=obf, compress=compress))
    return out
