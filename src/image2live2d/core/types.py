"""Intermediate data types passed *between* pipeline stages (upstream of the IRR).

These are deliberately lightweight: heavy arrays (images, masks) are referenced by path or held as
opaque objects so the contract stays importable without the ML extras installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..irr.schema import SemanticRole


@dataclass
class ImageAsset:
    """A loaded source image on disk."""

    path: Path
    width: int
    height: int


@dataclass
class PreparedImage:
    """Source image after preprocessing (bg removed, normalized, character-cropped)."""

    path: Path
    width: int
    height: int
    alpha_path: Path | None = None  # foreground mask, if produced


@dataclass
class Layer:
    """One decomposed, inpainted layer from See-through."""

    id: str
    semantic_role: SemanticRole
    texture_path: Path
    draw_order: int
    width: int
    height: int
    # bounding box of the layer within the model canvas, normalized [0,1]
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)


@dataclass
class LayerStack:
    """The full decomposition result: ordered, semantically-labeled, inpainted layers."""

    layers: list[Layer] = field(default_factory=list)
    canvas_width: int = 0
    canvas_height: int = 0

    def by_role(self, role: SemanticRole) -> list[Layer]:
        return [layer for layer in self.layers if layer.semantic_role is role]
