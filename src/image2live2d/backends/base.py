"""Emitter contract. The *only* backend-specific surface in the system.

Everything upstream produces a ``Rig`` (IRR); an ``Emitter`` serializes that ``Rig`` to a concrete
on-disk model. Route B (nijilive ``.inp``) ships first; Route A (Live2D ``.moc3``) is a second
``Emitter`` that reuses the entire upstream pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..irr.schema import Rig


class Emitter(ABC):
    """Serialize a ``Rig`` to a backend-specific model bundle."""

    name: str
    extension: str

    @abstractmethod
    def emit(self, rig: Rig, out_dir: Path) -> Path:
        """Write the model bundle into ``out_dir`` and return the path to the primary file."""
        raise NotImplementedError
