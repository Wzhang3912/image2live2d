"""IRR -> Cubism ``.model3.json`` (open JSON) — the bundle manifest.

References the other files (``.moc3``, textures, ``.physics3``, ``.cdi3``, motions) and declares the
EyeBlink / LipSync parameter groups and clickable hit areas. This is what a Live2D runtime
(pixi-live2d-display, VTube Studio) loads first.
"""

from __future__ import annotations

from ...irr.schema import Rig
from . import mapping

MODEL_VERSION = 3


def model3(
    rig: Rig,
    *,
    moc: str,
    textures: list[str],
    physics: str | None = None,
    display_info: str | None = None,
    motions: dict[str, list[str]] | None = None,
) -> dict:
    """Build the ``.model3.json`` document.

    ``moc``/``textures``/``physics``/``display_info`` are relative file paths; ``motions`` maps a
    motion group name (e.g. ``"Idle"``) to a list of ``.motion3.json`` paths.
    """
    refs: dict = {"Moc": moc, "Textures": list(textures)}
    if physics:
        refs["Physics"] = physics
    if display_info:
        refs["DisplayInfo"] = display_info
    if motions:
        refs["Motions"] = {
            group: [{"File": f} for f in files] for group, files in motions.items()
        }

    return {
        "Version": MODEL_VERSION,
        "FileReferences": refs,
        "Groups": mapping.model_groups(rig),
        "HitAreas": mapping.hit_areas(rig),
    }
