"""Tests for the INP binary container framing (round-trip + spec conformance)."""

from __future__ import annotations

import struct

from image2live2d.backends.nijilive.inp import (
    EXT_SECT,
    MAGIC,
    TEX_SECT,
    ExtEntry,
    InpFile,
    Texture,
    TextureEncoding,
)


def test_magic_and_header_layout():
    inp = InpFile(payload=b'{"a":1}')
    raw = inp.to_bytes()
    assert raw[:8] == MAGIC == b"TRNSRTS\0"
    payload_len = struct.unpack(">I", raw[8:12])[0]
    assert payload_len == len(inp.payload)
    assert raw[12 : 12 + payload_len] == inp.payload
    # texture section header immediately follows the payload
    assert raw[12 + payload_len : 12 + payload_len + 8] == TEX_SECT


def test_roundtrip_with_textures():
    inp = InpFile(
        payload=b'{"meta":{"name":"x"}}',
        textures=[
            Texture(data=b"\x89PNGfake", encoding=TextureEncoding.PNG),
            Texture(data=b"tgadata", encoding=TextureEncoding.TGA),
        ],
    )
    restored = InpFile.from_bytes(inp.to_bytes())
    assert restored.payload == inp.payload
    assert len(restored.textures) == 2
    assert restored.textures[0].data == b"\x89PNGfake"
    assert restored.textures[0].encoding is TextureEncoding.PNG
    assert restored.textures[1].encoding is TextureEncoding.TGA


def test_roundtrip_with_ext_section():
    inp = InpFile(
        payload=b"{}",
        textures=[Texture(data=b"img")],
        ext=[ExtEntry(name="com.image2live2d.irr", payload=b"hello")],
    )
    raw = inp.to_bytes()
    assert EXT_SECT in raw
    restored = InpFile.from_bytes(raw)
    assert len(restored.ext) == 1
    assert restored.ext[0].name == "com.image2live2d.irr"
    assert restored.ext[0].payload == b"hello"


def test_no_ext_section_when_empty():
    raw = InpFile(payload=b"{}", textures=[Texture(data=b"i")]).to_bytes()
    assert EXT_SECT not in raw


def test_file_write_read(tmp_path):
    path = tmp_path / "model.inp"
    InpFile(payload=b'{"ok":true}', textures=[Texture(data=b"img")]).write(path)
    restored = InpFile.read(path)
    assert restored.payload == b'{"ok":true}'
    assert restored.textures[0].data == b"img"
