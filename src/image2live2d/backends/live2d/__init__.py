"""Route A backend — Live2D ``.moc3`` (closed format, large ecosystem). Added in Phase 4.

The four sibling JSON files (model3 / physics3 / motion3 / cdi3) are emitted from the IRR headless;
the ``.moc3`` binary is a gated seam (template-binary mutation). See docs/PHASE4_PLAN.md.
"""

from .emitter import Live2DBundle, Live2DEmitter
from .moc3 import MocWriter, write_moc3_from_template

__all__ = ["Live2DEmitter", "Live2DBundle", "MocWriter", "write_moc3_from_template"]
