"""Stage 6 — assemble the IRR.

Turns the upstream products (a decomposed ``LayerStack``, its meshes, the rig-authoring output, and
physics) into a single validated ``Rig``. This is pure wiring: one ``Texture`` and one ``Part`` per
layer (1:1, since each decomposed layer owns its own PNG), plus the meshes/deformers/parameters
produced earlier. Construction runs the IRR's integrity validator, so a bad binding fails loudly
here rather than in an emitter.
"""

from __future__ import annotations

from pathlib import Path

from .types import LayerStack
from ..irr.schema import (
    Animation,
    Deformer,
    Mesh,
    Meta,
    Parameter,
    Part,
    PhysicsRig,
    Rig,
    Texture,
)


def texture_id_for(layer_id: str) -> str:
    return f"tex_{layer_id}"


def textures_for(stack: LayerStack) -> list[Texture]:
    """One ``Texture`` per layer; ``path`` is the PNG basename (resolved against the emitter's
    ``asset_root``)."""
    return [
        Texture(
            id=texture_id_for(layer.id),
            path=Path(layer.texture_path).name,
            width=layer.width,
            height=layer.height,
        )
        for layer in stack.layers
    ]


def parts_for(stack: LayerStack) -> list[Part]:
    """One ``Part`` per layer, carrying its semantic role and draw order."""
    return [
        Part(
            id=layer.id,
            semantic_role=layer.semantic_role,
            texture_id=texture_id_for(layer.id),
            draw_order=layer.draw_order,
        )
        for layer in stack.layers
    ]


def assemble_rig(
    *,
    name: str,
    source: str | None,
    stack: LayerStack,
    meshes: list[Mesh],
    deformers: list[Deformer],
    parameters: list[Parameter],
    physics: list[PhysicsRig],
    archetype: str | None = None,
    animations: list[Animation] | None = None,
) -> Rig:
    """Assemble and validate the complete ``Rig`` (IRR)."""
    return Rig(
        meta=Meta(name=name, source_image=source, archetype=archetype),
        textures=textures_for(stack),
        parts=parts_for(stack),
        meshes=meshes,
        deformers=deformers,
        parameters=parameters,
        physics=physics,
        animations=animations or [],
    )
