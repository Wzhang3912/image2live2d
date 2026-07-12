"""IRR -> ``.moc3`` binary — the ``MocWriter`` seam of Route A.

``.moc3`` is the only closed file in a Live2D model. It **can** now be generated **from scratch** —
the v3 format is fully reverse-engineered and ``moc3_emit.native_moc_writer`` builds a valid file
directly from the IRR (no Cubism template needed). Generated files load and render a full character in
Cubism Viewer 5.3 and VTube Studio (see docs/PHASE4B_MOC3_FEASIBILITY.md and the ``moc3-official-
runtime-conventions`` memory). This supersedes the earlier assumption that a hand-rigged template was
required; template-binary mutation (the CartoonAlive playbook) remains a *valid alternative* writer but
is no longer the only path.

This module keeps a swappable ``MocWriter`` seam so the caller chooses the strategy:
  - **from-scratch** (default, headless): inject ``moc3_emit.native_moc_writer`` — no template.
  - **template mutation**: inject a writer that mutates a Cubism-authored template ``.moc3``.

The only genuine gate left is **legal**, not technical: authoring/shipping ``.moc3`` needs a Live2D
publishing license before commercial use (R&D is fine; Route B / nijilive ``.inp`` stays fully clean).
If no ``MocWriter`` is injected the bundle is written JSON-only and renders the moment a ``.moc3`` is
supplied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ...irr.schema import Rig

# A MocWriter mutates a template .moc3 to fit `rig` and returns the resulting .moc3 bytes.
MocWriter = Callable[[Rig, Path], bytes]


def write_moc3_from_template(
    rig: Rig,
    template_path: str | Path | None,
    *,
    writer: MocWriter | None = None,
) -> bytes:
    """Produce ``.moc3`` bytes for ``rig`` by mutating a template (gated).

    If a ``writer`` is injected, delegate to it. Otherwise raise ``NotImplementedError`` — the bundle
    assembler catches this and writes a JSON-only bundle (renderable once a ``.moc3`` is supplied).
    """
    if writer is not None:
        # A writer may be template-based (template-binary mutation) or template-free (native
        # from-scratch generation, e.g. moc3_emit.native_moc_writer). Pass the template if we have one.
        return writer(rig, Path(template_path) if template_path is not None else None)
    raise NotImplementedError(
        "no MocWriter injected — pass moc3_emit.native_moc_writer to generate a .moc3 from scratch, "
        "or a template-mutation writer. The other 4 model files are emitted regardless."
    )
