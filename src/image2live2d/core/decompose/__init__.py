"""Stage 2 — Decompose. Single image -> semantic, inpainted, depth-ordered layers.

Implementation lives in :mod:`.sources`; this package re-exports its public API.
"""

from __future__ import annotations

from .sources import (
    decompose,
    from_layer_dir,
    from_psd,
    from_service,
    parse_layer_name,
    png_size,
    RawLayer,
    raws_to_stack,
    role_from_layer_name,
    RoleMapper,
)

__all__ = [
    "decompose",
    "from_layer_dir",
    "from_psd",
    "from_service",
    "parse_layer_name",
    "png_size",
    "RawLayer",
    "raws_to_stack",
    "role_from_layer_name",
    "RoleMapper",
]
