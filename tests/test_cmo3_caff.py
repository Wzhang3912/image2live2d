"""CAFF container for .cmo3 export (Phase 0). The container that wraps main.xml + textures into an
editable Cubism Editor project. Verified here by round-trip (pack -> unpack) across keys, compression
modes, and obfuscation; the byte-for-byte match against the reference packer is checked in a local
oracle, not in CI (the reference is an external read-only clone)."""

from __future__ import annotations

import struct

import pytest

from image2live2d.backends.live2d.cmo3.caff import (
    COMPRESS_FAST,
    COMPRESS_RAW,
    COMPRESS_SMALL,
    GUARD,
    MAGIC,
    CaffEntry,
    pack_caff,
    unpack_caff,
)


def _sample(compress=COMPRESS_RAW):
    return [
        CaffEntry("main.xml", b"<model>" + b"hello" * 200 + b"</model>", tag="main_xml",
                  obfuscated=True, compress=compress),
        CaffEntry("texture_00.png", bytes(range(256)) * 3, tag="", obfuscated=True,
                  compress=COMPRESS_RAW),
    ]


def test_archive_has_magic_and_guard():
    data = pack_caff(_sample(), key=42)
    assert data[:4] == MAGIC
    assert data[4:7] == b"\x00\x00\x00"        # archive version
    assert data[7:11] == b"----"               # cmo3 format id
    assert data[-2:] == GUARD


@pytest.mark.parametrize("key", [0, 1, 42, -7, 0x7FFFFFFF])
def test_round_trip_preserves_entries(key):
    entries = _sample()
    back = unpack_caff(pack_caff(entries, key=key))
    assert [(e.path, e.tag, e.content) for e in back] == \
           [(e.path, e.tag, e.content) for e in entries]


@pytest.mark.parametrize("compress", [COMPRESS_RAW, COMPRESS_FAST, COMPRESS_SMALL])
def test_round_trip_across_compression(compress):
    # a payload with real redundancy so FAST/SMALL actually deflate it
    entries = [CaffEntry("main.xml", b"<a>" + b"ABCD" * 4000 + b"</a>", tag="main_xml",
                         obfuscated=True, compress=compress)]
    back = unpack_caff(pack_caff(entries, key=42))
    assert back[0].content == entries[0].content
    assert back[0].compress == compress


def test_key_zero_is_no_obfuscation():
    # with key 0 an unobfuscated raw payload appears verbatim in the archive bytes
    payload = b"VERBATIM-PAYLOAD-MARKER"
    data = pack_caff([CaffEntry("t.png", payload, obfuscated=False, compress=COMPRESS_RAW)], key=0)
    assert payload in data


def test_obfuscated_payload_is_scrambled_but_recovers():
    payload = b"SECRET" * 20
    data = pack_caff([CaffEntry("t.bin", payload, obfuscated=True, compress=COMPRESS_RAW)], key=42)
    assert payload not in data                  # XOR-scrambled in the archive
    assert unpack_caff(data)[0].content == payload  # ...but recovered on read


def test_file_count_is_obfuscated_with_the_key():
    # the file-count int32 (first obfuscated field, at offset 0x36) is XORed with the key
    entries = _sample()
    off = 0x36
    plain = struct.unpack_from(">I", pack_caff(entries, key=0), off)[0]
    keyed = struct.unpack_from(">I", pack_caff(entries, key=42), off)[0]
    assert plain == len(entries)
    assert keyed == (len(entries) ^ 42)


def test_bad_magic_and_guard_rejected():
    good = pack_caff(_sample(), key=0)
    with pytest.raises(ValueError):
        unpack_caff(b"ZZZZ" + good[4:])
    with pytest.raises(ValueError):
        unpack_caff(good[:-2] + b"\x00\x00")
