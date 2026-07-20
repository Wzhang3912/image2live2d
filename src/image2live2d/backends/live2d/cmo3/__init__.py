"""``.cmo3`` — editable Cubism Editor project export (CAFF container + main.xml model graph).

Phase 0 is the container: :mod:`.caff`. Later phases build the ``main.xml`` model graph from an IRR
``Rig`` and assemble the full editable project.
"""

from __future__ import annotations

from .caff import (
    COMPRESS_FAST,
    COMPRESS_RAW,
    COMPRESS_SMALL,
    CaffEntry,
    pack_caff,
    unpack_caff,
)

__all__ = [
    "CaffEntry",
    "pack_caff",
    "unpack_caff",
    "COMPRESS_RAW",
    "COMPRESS_FAST",
    "COMPRESS_SMALL",
]
