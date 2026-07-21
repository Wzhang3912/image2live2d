"""Assemble a complete ``.cmo3`` (editable Cubism Editor project) from an IRR ``Rig``.

This ties the two halves together: :func:`.model_xml.build_main_xml` produces the model graph and the
list of texture PNGs, and :func:`.caff.pack_caff` wraps ``main.xml`` + those PNGs into the CAFF
container. The result is a single ``.cmo3`` byte string ready to write to disk.
"""

from __future__ import annotations

from pathlib import Path

from ....irr.schema import Rig, Texture
from .caff import COMPRESS_FAST, COMPRESS_RAW, CaffEntry, pack_caff
from .model_xml import build_main_xml

# The Editor writes obfuscated archives with a fixed non-zero key; matching it keeps our output shaped
# like a real project (and exercises the XOR path). The key is plaintext in the header, so it is not a
# secret — see :mod:`.caff`.
DEFAULT_KEY = 42


def rig_to_cmo3(rig: Rig, asset_root: str | Path | None = None, *, key: int = DEFAULT_KEY) -> bytes:
    """Serialize ``rig`` to ``.cmo3`` bytes.

    ``asset_root`` is the directory the rig's texture paths are relative to; each drawable part's PNG is
    read from there. If ``asset_root`` is ``None`` a solid-magenta placeholder PNG is substituted for
    every texture (useful for structural tests without asset files on disk).
    """
    root = Path(asset_root) if asset_root is not None else None

    def load_png(tex: Texture) -> bytes:
        if root is not None:
            data = (root / tex.path).read_bytes()
            if data[:8] == b"\x89PNG\r\n\x1a\n":
                return data
            raise ValueError(f"texture {tex.path!r} is not a PNG")
        return _placeholder_png(tex.width, tex.height)

    xml_bytes, texture_files = build_main_xml(rig, load_png)

    entries = [
        CaffEntry(path, png, tag="", obfuscated=True, compress=COMPRESS_RAW)
        for path, png in texture_files
    ]
    # main.xml deflates well and the Editor stores it FAST; textures are already PNG-compressed (RAW).
    entries.append(CaffEntry("main.xml", xml_bytes, tag="main_xml", obfuscated=True,
                             compress=COMPRESS_FAST))
    return pack_caff(entries, key=key)


def _placeholder_png(w: int, h: int) -> bytes:
    """A minimal solid-magenta RGBA PNG (stdlib only) — a stand-in when no real texture is available."""
    import struct
    import zlib

    def chunk(ctype: bytes, data: bytes) -> bytes:
        body = ctype + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    w, h = max(1, int(w)), max(1, int(h))
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
    row = b"\x00" + bytes([255, 0, 255, 255]) * w
    idat = chunk(b"IDAT", zlib.compress(row * h))
    return b"\x89PNG\r\n\x1a\n" + ihdr + idat + chunk(b"IEND", b"")
