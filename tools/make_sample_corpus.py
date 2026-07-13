#!/usr/bin/env python3
"""Generate a small, reproducible *taste corpus* for calibrating the dynamics score.

Real professional ``.moc3`` / ``.physics3.json`` models are copyrighted and can't live in this repo,
so this writes a synthetic stand-in: a handful of characters whose per-part **silhouettes** are shaped
the way a real rigger's parts are (a ponytail hangs long and thin into void; a face is a compact blob
glued in the middle), each **labeled by professional convention** — hair strands / ribbons / skirt hems
/ capes get physics; faces / eyes / torsos / glued bodices / collars do not. Running the calibrator
against it answers: *does our geometric score, at its default thresholds, agree with a rigger's taste?*

    python tools/make_sample_corpus.py                    # writes ./corpus/ (gitignored)
    python tools/calibrate_dynamics.py --corpus corpus/manifest.json

To calibrate against your OWN real models instead, drop them in ``corpus/`` with a manifest of the same
shape (see tools/calibrate_dynamics.py) — a layer's ground-truth label is "has physics" when one of its
params appears in that model's real ``.physics3.json``. This generator is only the runnable demo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

CANVAS = 128


def _png(path: Path, shapes) -> None:
    """Draw opaque ``shapes`` (each an (x0,y0,x1,y1) rect in [0,1], y DOWN) onto a transparent canvas."""
    img = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for (x0, y0, x1, y1) in shapes:
        d.rectangle([x0 * CANVAS, y0 * CANVAS, x1 * CANVAS, y1 * CANVAS], fill=(200, 150, 160, 255))
    img.save(path)


# Each part: (id, role, params, pro_physics, shapes). ``shapes`` are rects in [0,1] with y DOWN (top=0).
# The silhouettes are deliberately convention-shaped so the geometry matches the label a rigger'd give.
_CHARACTERS = {
    "twintails_girl": [
        ("face", "face_base", ["ParamAngleX"], False, [(0.34, 0.10, 0.66, 0.48)]),
        ("eye_l", "eye_l", ["ParamEyeLOpen"], False, [(0.40, 0.28, 0.47, 0.34)]),
        ("bangs", "hair_front", ["ParamHairFront"], True, [(0.34, 0.06, 0.66, 0.22)]),
        ("twintail_l", "hair_side", ["ParamHairSide"], True, [(0.20, 0.14, 0.28, 0.66)]),
        ("twintail_r", "hair_side", ["ParamHairSide"], True, [(0.72, 0.14, 0.80, 0.66)]),
        ("torso", "torso", ["ParamBodyAngleX"], False, [(0.38, 0.48, 0.62, 0.74)]),
        ("skirt", "clothing", ["ParamSkirtC"], True, [(0.30, 0.70, 0.70, 0.92)]),
    ],
    "cape_boy": [
        ("face", "face_base", ["ParamAngleX"], False, [(0.36, 0.12, 0.64, 0.46)]),
        ("ponytail", "hair_back", ["ParamHairBack"], True, [(0.46, 0.10, 0.54, 0.60)]),
        ("collar", "clothing", ["ParamCollar"], False, [(0.42, 0.46, 0.58, 0.52)]),
        ("bodice", "clothing", ["ParamBody"], False, [(0.40, 0.50, 0.60, 0.74)]),
        # cape hangs wider and lower than the torso -> a big free edge into void
        ("cape", "clothing", ["ParamCape"], True, [(0.26, 0.48, 0.74, 0.88)]),
        ("torso", "torso", ["ParamBodyAngleX"], False, [(0.40, 0.48, 0.60, 0.74)]),
    ],
}


def build(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    models = []
    for name, parts in _CHARACTERS.items():
        phys_params = sorted({p for _, _, params, physics, _ in parts if physics for p in params})
        (out_dir / f"{name}.physics3.json").write_text(json.dumps({
            "PhysicsSettings": [{"Output": [{"Destination": {"Id": p}}]} for p in phys_params]
        }, indent=2))
        layers = []
        for i, (pid, role, params, _physics, shapes) in enumerate(parts):
            tex = f"{name}__{pid}.png"
            _png(out_dir / tex, shapes)
            layers.append({"id": pid, "role": role, "texture": tex, "params": params, "draw_order": i})
        models.append({"name": name, "physics3": f"{name}.physics3.json", "layers": layers})
    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps({"models": models}, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate a synthetic taste corpus for dynamics calibration.")
    ap.add_argument("--out", type=Path, default=Path("corpus"), help="output dir (default: ./corpus)")
    args = ap.parse_args(argv)
    manifest = build(args.out)
    n_parts = sum(len(p) for p in _CHARACTERS.values())
    print(f"wrote {len(_CHARACTERS)} characters, {n_parts} parts -> {manifest}")
    print(f"run: python tools/calibrate_dynamics.py --corpus {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
