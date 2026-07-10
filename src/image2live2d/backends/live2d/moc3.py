"""IRR -> ``.moc3`` binary — the **gated seam** of Route A.

``.moc3`` is the only closed file in a Live2D model and **cannot be generated from scratch** (the
Cubism SDK only displays; community RE writers are dead/partial — see docs/RIGGING_DEEP_DIVE.md). The
viable path is **template-binary mutation** (the CartoonAlive playbook): take a hand-rigged Cubism
template ``.moc3`` whose part/deformer/parameter layout matches the IRR archetype, then overwrite its
numeric arrays (vertex positions, deform offsets, parameter keyform values) to fit this character.

That needs two things this repo can't provide headlessly (see Phase 4 task #29):
  1. a Cubism-authored template ``.moc3`` to mutate, and
  2. a Live2D publishing license (template mutation is legally gray).

So this is a swappable seam, exactly like ``decompose`` and the landmark ML detectors: everything
around it (the open JSON emitters, the bundle assembly) is real and tested; only this call is gated.
Inject a real ``MocWriter`` once a template + license are in hand.
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
        "writing .moc3 needs a Cubism template + a MocWriter (template-binary mutation) — see "
        "docs/PHASE4_PLAN.md §4B / task #29. The other 4 model files are emitted regardless."
    )
