"""Render every motion clip in a Cubism bundle to a contact sheet — a comprehensive visual check.

A motion "looks live" and still be wrong: parts that are independent layers pull apart at joint
extremes (a raised arm tears from the shoulder, a leaned torso opens the neck seam). Numbers can't see
that; only a render can. This drives the EMITTED ``.moc3`` through the native Cubism core — what Cubism
Viewer actually draws — samples each clip at its held extremes, and tiles the frames into one image per
clip, all at the same fixed framing so the deformation is comparable frame to frame.

Usage:
    PYTHONPATH=src python tools/render_motions.py <bundle_dir> [--out DIR] [--clip NAME] [--size N]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moc3_render import Renderer  # noqa: E402


def _curve(motion: dict, pid: str):
    for c in motion["Curves"]:
        if c["Id"] != pid:
            continue
        seg = c["Segments"]
        pts, i = [(seg[0], seg[1])], 2
        while i < len(seg):
            step = 7 if int(seg[i]) == 1 else 3
            pts.append((seg[i + step - 2], seg[i + step - 1]))
            i += step
        return pts
    return None


def _sample(pts, t: float) -> float:
    if t <= pts[0][0]:
        return pts[0][1]
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if t <= t1:
            return v0 + (t - t0) / max(t1 - t0, 1e-9) * (v1 - v0)
    return pts[-1][1]


def _pose_at(motion: dict, t: float) -> dict:
    """Every curve's value at time ``t`` — the full parameter pose the runtime would hold."""
    return {c["Id"]: _sample(_curve(motion, c["Id"]), t) for c in motion["Curves"]}


def _frame_times(motion: dict, n: int) -> list[float]:
    """``n`` sample times that land on the clip's *held* poses, not its transitions.

    A drive clip is snap -> hold -> snap -> hold -> settle; sampling on a uniform grid can catch a frame
    mid-snap, where nothing is at rest and the pose is meaningless. Instead pick the times where the
    total motion (summed |value - default-ish|) is locally largest and where it is smallest — the
    extremes and the neutral — which is exactly what we want to inspect."""
    dur = motion["Meta"]["Duration"]
    # coarse scan of overall deflection over the clip
    grid = [dur * k / 60 for k in range(61)]
    defl = []
    for t in grid:
        pose = _pose_at(motion, t)
        defl.append((t, sum(abs(v) for v in pose.values())))
    # always include t=0 (rest-ish) and the global max-deflection frame; fill the rest evenly
    times = {0.0, max(defl, key=lambda kv: kv[1])[0]}
    for k in range(1, n):
        times.add(dur * k / (n - 1))
    return sorted(times)[:n]


def _label(img, text: str):
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, len(text) * 7 + 8, 16], fill=(20, 20, 28))
    d.text((4, 3), text, fill=(240, 240, 245))
    return img


def _sheet(frames, cols: int):
    from PIL import Image
    if not frames:
        return None
    w, h = frames[0].size
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (w * cols, h * rows), (255, 255, 255))
    for i, fr in enumerate(frames):
        sheet.paste(fr, ((i % cols) * w, (i // cols) * h))
    return sheet


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    def opt(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default

    args = [a for a in argv if not a.startswith("--")
            and (not argv or argv[argv.index(a) - 1] not in ("--out", "--clip", "--size"))]
    if not args:
        print(__doc__)
        return 2
    bundle = Path(args[0])
    out = Path(opt("--out", str(bundle / "motion_sheets")))
    only = opt("--clip")
    size = int(opt("--size", "360"))
    frames_per = 5

    out.mkdir(parents=True, exist_ok=True)
    r = Renderer(str(bundle / "model.moc3"), str(bundle / "textures" / "atlas.png"))
    bounds = r.rest_bounds(margin=0.08)

    model3 = json.loads((bundle / "model.model3.json").read_text())
    clips = sorted({f["File"].split(".")[-3]
                    for g in model3["FileReferences"]["Motions"].values() for f in g})
    if only:
        clips = [only]

    for name in clips:
        motion = json.loads((bundle / f"model.{name}.motion3.json").read_text())
        frames = []
        for t in _frame_times(motion, frames_per):
            r.set_many(_pose_at(motion, t))
            fr = r.render(size=size, bounds=bounds)
            frames.append(_label(fr, f"{name}  t={t:.2f}s"))
        sheet = _sheet(frames, cols=frames_per)
        path = out / f"{name}.png"
        sheet.save(path)
        print(f"  {path}")
    print(f"\n{len(clips)} sheet(s) in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
