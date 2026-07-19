"""Comprehensive per-character Live2D sweep: emit the .moc3, render every motion axis at its extreme
to two contact sheets (full body + head zoom), simulate the physics on the motion-driving clips, and
print the QA/coverage verdict. One command so a reviewer (human or agent) can triage a character.

    PYTHONPATH=src python tools/sweep_character.py <layer_dir> <out_dir>

Writes <out_dir>/sweep_full.png and <out_dir>/sweep_head.png; prints physics excitation + QA to stdout.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moc3_render import Renderer  # noqa: E402


# Each entry: label, param settings. Params clamp to their own range in the renderer, so a value past
# the max just saturates. These exercise every rig axis at both extremes.
POSES = [
    ("rest", {}),
    ("yaw+", {"ParamAngleX": 30}), ("yaw-", {"ParamAngleX": -30}),
    ("pitch+", {"ParamAngleY": 30}), ("pitch-", {"ParamAngleY": -30}),
    ("roll+", {"ParamAngleZ": 30}),
    ("body_sway", {"ParamBodyAngleX": 10, "ParamBodyAngleZ": 5}),
    ("body_bow", {"ParamBodyAngleY": 10}),
    ("arms_raise", {"ParamArmLA": 10, "ParamArmRA": 10, "ParamArmLB": 7, "ParamArmRB": 7}),
    ("arms_swing", {"ParamArmLA": 10, "ParamArmRA": -10}),
    ("legs_splay", {"ParamLegLA": 10, "ParamLegRA": 10, "ParamLegLB": 6, "ParamLegRB": 6}),
    ("hair_bounce", {"ParamHairFrontV": 1, "ParamHairSideV": 1, "ParamHairBackV": 1}),
]
FACE_POSES = [
    ("rest", {}),
    ("blink", {"ParamEyeLOpen": 0, "ParamEyeROpen": 0}),
    ("talk", {"ParamMouthOpenY": 1}),
    ("smile", {"ParamMouthForm": 1, "ParamEyeLOpen": 0.6, "ParamEyeROpen": 0.6}),
    ("sad", {"ParamMouthForm": -1}),
    ("surprise", {"ParamMouthOpenY": 0.7, "ParamBrowLY": 1, "ParamBrowRY": 1}),
    ("look_l", {"ParamEyeBallX": -1, "ParamEyeBallY": 1}),
    ("look_r", {"ParamEyeBallX": 1, "ParamEyeBallY": -1}),
    ("brows", {"ParamBrowLY": 1, "ParamBrowRY": 1}),
]


def _sheet(renderer, poses, size, cols, out_path, head=False):
    from PIL import Image, ImageDraw
    import numpy as np

    bounds = renderer.rest_bounds(margin=0.25)
    if head:  # crop the top of the figure for the face sheet
        renderer.set_many({})
        im = renderer.render(size=700, bounds=bounds)
        a = np.array(im.convert("RGB"))
        ys, xs = np.where(a.sum(2) < 720)
        if len(ys):
            x0, x1, y0 = xs.min(), xs.max(), ys.min()
            # map the head crop (top ~30%) back to model bounds
            bx0, by1 = bounds[0], bounds[3]
            bw, bh = bounds[2] - bounds[0], bounds[3] - bounds[1]
            fx0 = bx0 + bw * x0 / 700
            fx1 = bx0 + bw * x1 / 700
            fy1 = by1 - bh * y0 / 700
            bounds = (fx0, fy1 - 0.34 * bh, fx1, fy1)
    rows = (len(poses) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * size, rows * (size + 20)), (20, 22, 26))
    d = ImageDraw.Draw(sheet)
    for i, (label, settings) in enumerate(poses):
        renderer.set_many(settings)
        im = renderer.render(size=size, bounds=bounds).convert("RGB")
        x, y = (i % cols) * size, (i // cols) * (size + 20)
        sheet.paste(im, (x, y + 20))
        d.text((x + 5, y + 4), label, fill=(140, 255, 220))
    sheet.save(out_path)


def main() -> int:
    layer_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / "bundle"

    src = str(Path(__file__).resolve().parents[1] / "src")
    r = subprocess.run([sys.executable, "tools/emit_cubism_bundle.py", str(layer_dir), str(bundle)],
                       capture_output=True, text=True, env={"PYTHONPATH": src, "PATH": __import__("os").environ["PATH"]})
    print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "", flush=True)
    if r.returncode != 0:
        print("EMIT FAILED:\n" + r.stderr[-1500:]); return 1

    renderer = Renderer(bundle / "model.moc3", bundle / "textures" / "atlas.png")
    _sheet(renderer, POSES, 300, 6, out_dir / "sweep_full.png")
    _sheet(renderer, FACE_POSES, 340, 5, out_dir / "sweep_head.png", head=True)
    print(f"WROTE {out_dir/'sweep_full.png'} and {out_dir/'sweep_head.png'}", flush=True)

    # physics: simulate the motion-driving clips
    print("\n=== PHYSICS EXCITATION ===", flush=True)
    for clip in ("head_yaw", "head_pitch", "head_roll", "body_sway", "legs_swing"):
        p = subprocess.run([sys.executable, "tools/physics_excite.py", str(bundle), "--clip", clip],
                           capture_output=True, text=True,
                           env={"PYTHONPATH": src, "PATH": __import__("os").environ["PATH"]})
        for line in p.stdout.splitlines():
            if any(k in line for k in ("Hair", "Cloth", "Acc", "swings", "FROZEN", "output")):
                print(f"[{clip}] {line.strip()}", flush=True)

    # QA gate + coverage
    print("\n=== QA ===", flush=True)
    qa = subprocess.run([sys.executable, "-c",
        "from image2live2d.core import decompose;from image2live2d.pipeline import rig_from_stack;"
        "from image2live2d.core.qa.harness import evaluate;from pathlib import Path;"
        f"rig=rig_from_stack(decompose.from_layer_dir(Path('{layer_dir}')),name='c');"
        "rep=evaluate(rig,'c');"
        "print('parts',len(rig.parts),'params',len(rig.parameters),'physics',len(rig.physics));"
        "print('passed',rep.passed,'reasons',rep.reasons)"],
        capture_output=True, text=True, env={"PYTHONPATH": src, "PATH": __import__("os").environ["PATH"]})
    print(qa.stdout.strip() or qa.stderr[-800:], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
