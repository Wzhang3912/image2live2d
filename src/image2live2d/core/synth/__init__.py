"""Stage 3b — Synthesis. Paint the parts a decomposer cannot give us.

A decomposition only ever returns what is *visible* in the source art. Some parts a rig needs are, by
definition, not visible: the inside of a closed mouth is the obvious one. This package makes them.
"""

from __future__ import annotations

from .blush import synthesize_blush
from .eye import synthesize_closed_eyes
from .mouth import synthesize_mouth_cavity

__all__ = ["synthesize_blush", "synthesize_closed_eyes", "synthesize_mouth_cavity"]
