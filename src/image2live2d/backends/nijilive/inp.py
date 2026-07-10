"""INP binary container (read + write).

INP = "Inochi2D Puppet", the open container format used by nijilive. Layout (all multi-byte
values **big-endian**), verified against the Inochi2D INP spec and the nijilive source:

    [magic 8B = "TRNSRTS\\0"]
    [payload_len u32][payload JSON bytes]
    ["TEX_SECT"][tex_count u32]  (per texture: [len u32][encoding u8][data])
    (optional) ["EXT_SECT"][ext_count u32]  (per entry: [name_len u32][name][payload_len u32][payload])

This module handles ONLY the container framing; the JSON puppet payload is built elsewhere
(``puppet.py``). Keeping them separate means the framing is testable on its own and stable even as
the puppet schema evolves.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

MAGIC = b"TRNSRTS\0"
TEX_SECT = b"TEX_SECT"
EXT_SECT = b"EXT_SECT"


class TextureEncoding(IntEnum):
    PNG = 0
    TGA = 1
    BC7 = 2


@dataclass
class Texture:
    data: bytes
    encoding: TextureEncoding = TextureEncoding.PNG


@dataclass
class ExtEntry:
    name: str
    payload: bytes


@dataclass
class InpFile:
    """An in-memory INP: a JSON payload + ordered textures (+ optional vendor data)."""

    payload: bytes  # UTF-8 encoded JSON puppet
    textures: list[Texture] = field(default_factory=list)
    ext: list[ExtEntry] = field(default_factory=list)

    def to_bytes(self) -> bytes:
        out = bytearray()
        out += MAGIC
        out += struct.pack(">I", len(self.payload))
        out += self.payload

        out += TEX_SECT
        out += struct.pack(">I", len(self.textures))
        for tex in self.textures:
            out += struct.pack(">I", len(tex.data))
            out += struct.pack(">B", int(tex.encoding))
            out += tex.data

        if self.ext:
            out += EXT_SECT
            out += struct.pack(">I", len(self.ext))
            for entry in self.ext:
                name = entry.name.encode("utf-8")
                out += struct.pack(">I", len(name))
                out += name
                out += struct.pack(">I", len(entry.payload))
                out += entry.payload
        return bytes(out)

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_bytes(self.to_bytes())
        return p

    @classmethod
    def read(cls, path: str | Path) -> "InpFile":
        return cls.from_bytes(Path(path).read_bytes())

    @classmethod
    def from_bytes(cls, raw: bytes) -> "InpFile":
        r = _Reader(raw)
        if r.take(8) != MAGIC:
            raise ValueError("not an INP file (bad magic)")
        payload = r.take(r.u32())

        if r.take(8) != TEX_SECT:
            raise ValueError("expected TEX_SECT")
        textures: list[Texture] = []
        for _ in range(r.u32()):
            length = r.u32()
            enc = TextureEncoding(r.u8())
            textures.append(Texture(data=r.take(length), encoding=enc))

        ext: list[ExtEntry] = []
        if r.remaining() >= 8 and r.peek(8) == EXT_SECT:
            r.take(8)
            for _ in range(r.u32()):
                name = r.take(r.u32()).decode("utf-8")
                ext.append(ExtEntry(name=name, payload=r.take(r.u32())))

        return cls(payload=payload, textures=textures, ext=ext)


class _Reader:
    def __init__(self, buf: bytes) -> None:
        self._buf = buf
        self._pos = 0

    def take(self, n: int) -> bytes:
        if self._pos + n > len(self._buf):
            raise ValueError("unexpected end of INP data")
        chunk = self._buf[self._pos : self._pos + n]
        self._pos += n
        return chunk

    def peek(self, n: int) -> bytes:
        return self._buf[self._pos : self._pos + n]

    def u32(self) -> int:
        return struct.unpack(">I", self.take(4))[0]

    def u8(self) -> int:
        return struct.unpack(">B", self.take(1))[0]

    def remaining(self) -> int:
        return len(self._buf) - self._pos
