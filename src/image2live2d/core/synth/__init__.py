"""Stage 3b — Synthesis. Paint the parts a decomposer cannot give us.

A decomposition only ever returns what is *visible* in the source art. Some parts a rig needs are, by
definition, not visible: the inside of a closed mouth is the obvious one. This package makes them.
"""

from __future__ import annotations

from .mouth import synthesize_mouth_cavity

__all__ = ["synthesize_mouth_cavity"]
