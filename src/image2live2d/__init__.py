"""image2live2d — automatically convert an image into a reusable, riggable 2D puppet.

The package is organized around the Intermediate Rig Representation (``image2live2d.irr``):
``core`` holds the shared pipeline stages that build a ``Rig``; ``backends`` holds the thin
emitters that serialize a ``Rig`` to a concrete format (nijilive ``.inp`` / Live2D ``.moc3``).
"""

__version__ = "0.2.0"

from .api import (
    ConversionResult,
    convert_layers,
    convert_psd,
    convert_stack,
    rig_from_layer_dir,
    rig_from_psd,
)

__all__ = [
    "__version__",
    "ConversionResult",
    "convert_layers",
    "convert_psd",
    "convert_stack",
    "rig_from_layer_dir",
    "rig_from_psd",
]
