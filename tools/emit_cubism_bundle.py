"""Emit a complete, Cubism-Viewer-loadable Live2D bundle from a layer directory.

The ``--live2d`` flag on the CLI writes a *stub* .moc3 (it expects a template to mutate). The web
app instead builds a real binary .moc3 from scratch, plus the single shared texture atlas that
standard Cubism runtimes expect. This tool exposes that same path headlessly, so you can drop the
result straight into Cubism Viewer / VTube Studio without going through the browser.

Usage::

    python tools/emit_cubism_bundle.py out/char_layers_fixed out/eyeball/char_fixed_v10_cubism

Produces ``model.moc3``, ``model.model3.json``, ``model.physics3.json``, ``model.cdi3.json``,
``model.<anim>.motion3.json`` (idle + expressions) and ``textures/atlas.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from image2live2d.backends.live2d import cdi3 as _cdi3
from image2live2d.backends.live2d import model3 as _model3
from image2live2d.backends.live2d import motion3 as _motion3
from image2live2d.backends.live2d import physics3 as _physics3
from image2live2d.backends.live2d.moc3_binary import write_moc3
from image2live2d.backends.live2d.moc3_emit import build_atlas, rig_to_moc3
from image2live2d.core import decompose
from image2live2d.pipeline import rig_from_stack


def emit_bundle(layer_dir: Path, out: Path) -> Path:
    stack = decompose.from_layer_dir(layer_dir)
    rig = rig_from_stack(stack, name="model", source=str(layer_dir))

    out.mkdir(parents=True, exist_ok=True)
    (out / "textures").mkdir(exist_ok=True)

    # Real Cubism models use one shared atlas with remapped UVs, not per-part textures.
    atlas_img, uv_remap = build_atlas(rig, layer_dir)
    atlas_img.save(out / "textures" / "atlas.png")
    (out / "model.moc3").write_bytes(write_moc3(rig_to_moc3(rig, atlas_uv=uv_remap)))

    def wj(rel: str, doc: object) -> None:
        (out / rel).write_text(json.dumps(doc, indent=2))

    physics_file = None
    if rig.physics:
        physics_file = "model.physics3.json"
        wj(physics_file, _physics3.physics3(rig))
    cdi_file = "model.cdi3.json"
    wj(cdi_file, _cdi3.cdi3(rig))

    motions: dict[str, list[str]] = {}
    for anim in rig.animations:
        rel = f"model.{anim.name}.motion3.json"
        wj(rel, _motion3.motion3(anim))
        motions.setdefault(anim.name.capitalize(), []).append(rel)

    wj("model.model3.json", _model3.model3(
        rig, moc="model.moc3", textures=["textures/atlas.png"], physics=physics_file,
        display_info=cdi_file, motions=motions or None))

    print(f"wrote {out}/model.model3.json")
    print(f"  parts={len(rig.meshes)}  params={len(rig.parameters)}  physics={len(rig.physics)}")
    print(f"  motions={sorted(motions)}")
    print(f"  moc3={(out / 'model.moc3').stat().st_size} bytes (real binary — loads in Cubism Viewer)")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("layer_dir", type=Path, help="directory of {order}_{role}.png layers")
    ap.add_argument("out", type=Path, help="output bundle directory")
    args = ap.parse_args(argv)
    if not args.layer_dir.is_dir():
        print(f"error: {args.layer_dir} is not a directory", file=sys.stderr)
        return 2
    emit_bundle(args.layer_dir, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
