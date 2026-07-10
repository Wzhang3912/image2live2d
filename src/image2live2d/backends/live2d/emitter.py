"""IRR -> Live2D model bundle (``.model3.json`` + siblings, and a ``.moc3`` when available).

The four sibling JSON files (model3 / physics3 / motion3 / cdi3) are open and written from the IRR
unconditionally. The ``.moc3`` itself is the gated piece (``moc3.write_moc3_from_template``): if no
``MocWriter`` is injected, the bundle is written **JSON-only** — every reference is in place and the
model renders the moment a ``.moc3`` is dropped in (or a writer is supplied). ``emit`` returns the
``.model3.json`` path (what a Live2D runtime loads).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..base import Emitter
from ..nijilive.puppet import solid_png  # stdlib PNG placeholder (shared helper)
from ...irr.schema import Rig
from . import cdi3 as _cdi3
from . import model3 as _model3
from . import motion3 as _motion3
from . import physics3 as _physics3
from .moc3 import MocWriter, write_moc3_from_template


@dataclass
class Live2DBundle:
    """What ``emit`` produced — useful for tests/CLI to see whether the gated ``.moc3`` was written."""

    model3_path: Path
    files: list[str]
    moc_written: bool


class Live2DEmitter(Emitter):
    name = "live2d"
    extension = ".model3.json"

    def __init__(
        self,
        *,
        asset_root: str | Path | None = None,
        moc_writer: MocWriter | None = None,
        moc_template: str | Path | None = None,
    ) -> None:
        self.asset_root = Path(asset_root) if asset_root is not None else None
        self.moc_writer = moc_writer
        self.moc_template = moc_template

    def emit(self, rig: Rig, out_dir: Path) -> Path:
        return self.build(rig, out_dir).model3_path

    def build(self, rig: Rig, out_dir: str | Path) -> Live2DBundle:
        out = Path(out_dir)
        (out / "textures").mkdir(parents=True, exist_ok=True)
        name = rig.meta.name or "model"
        written: list[str] = []

        def write_json(rel: str, doc: dict) -> None:
            (out / rel).write_text(json.dumps(doc, indent=2))
            written.append(rel)

        # Textures (copy from asset_root, else a stdlib placeholder PNG so the bundle is complete).
        texture_files: list[str] = []
        for i, tex in enumerate(rig.textures):
            rel = f"textures/{i:03d}_{tex.id}.png"
            (out / rel).write_bytes(self._texture_bytes(tex))
            texture_files.append(rel)
            written.append(rel)

        physics_file: str | None = None
        if rig.physics:
            physics_file = f"{name}.physics3.json"
            write_json(physics_file, _physics3.physics3(rig))

        cdi_file = f"{name}.cdi3.json"
        write_json(cdi_file, _cdi3.cdi3(rig))

        motions: dict[str, list[str]] = {}
        for anim in rig.animations:
            rel = f"{name}.{anim.name}.motion3.json"
            write_json(rel, _motion3.motion3(anim))
            motions.setdefault(anim.name.capitalize(), []).append(rel)

        # Gated .moc3 (template mutation). JSON-only bundle if no writer is injected.
        moc_file = f"{name}.moc3"
        moc_written = False
        try:
            data = write_moc3_from_template(rig, self.moc_template, writer=self.moc_writer)
            (out / moc_file).write_bytes(data)
            written.append(moc_file)
            moc_written = True
        except NotImplementedError:
            pass

        doc = _model3.model3(
            rig,
            moc=moc_file,
            textures=texture_files,
            physics=physics_file,
            display_info=cdi_file,
            motions=motions or None,
        )
        model3_file = f"{name}.model3.json"
        write_json(model3_file, doc)

        return Live2DBundle(model3_path=out / model3_file, files=written, moc_written=moc_written)

    def _texture_bytes(self, tex) -> bytes:
        if self.asset_root is not None:
            path = self.asset_root / tex.path
            if path.is_file():
                return path.read_bytes()
        return solid_png(tex.width, tex.height)
