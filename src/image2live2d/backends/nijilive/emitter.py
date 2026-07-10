"""IRR -> nijilive ``.inp``.

nijilive's format is open and writable, so this emitter runs fully headless — no GUI, no closed
binary. This is why Route B is the de-risking first target: serialization is essentially free,
letting Phase 1 focus on rig quality.

Pipeline: ``Rig`` --(puppet.build_puppet)--> puppet dict + texture PNGs --(inp.InpFile)--> ``.inp``.
The original IRR is embedded in an ``EXT_SECT`` vendor entry for traceability/round-tripping.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..base import Emitter
from ...irr.schema import Rig
from .inp import ExtEntry, InpFile, Texture, TextureEncoding
from .puppet import DEFAULT_SCALE, build_puppet

IRR_EXT_NAME = "com.image2live2d.irr.v1"


class NijiliveEmitter(Emitter):
    name = "nijilive"
    extension = ".inp"

    def __init__(
        self,
        *,
        asset_root: str | Path | None = None,
        scale: float = DEFAULT_SCALE,
        embed_irr: bool = True,
    ) -> None:
        self.asset_root = asset_root
        self.scale = scale
        self.embed_irr = embed_irr

    def emit(self, rig: Rig, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        build = build_puppet(rig, asset_root=self.asset_root, scale=self.scale)
        payload = json.dumps(build.puppet, separators=(",", ":")).encode("utf-8")
        textures = [Texture(data=blob, encoding=TextureEncoding.PNG) for blob in build.textures]

        ext: list[ExtEntry] = []
        if self.embed_irr:
            ext.append(ExtEntry(name=IRR_EXT_NAME, payload=rig.model_dump_json().encode("utf-8")))

        path = out_dir / f"{rig.meta.name}{self.extension}"
        InpFile(payload=payload, textures=textures, ext=ext).write(path)
        return path
