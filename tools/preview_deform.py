"""Offline deformation preview — render an IRR rig at parameter extremes to PNG contact sheets.

A self-QA aid: nijigenerate is the ground-truth visual gate, but this lets you *see* the rig deform
(blink / head-turn / mouth / lean) headlessly to catch gross errors — an eye that won't close, a head
that shears, an accessory that floats off the head. It is an **approximate** piecewise-affine (per
triangle) software warp, not nijigenerate's renderer.

Usage:
    PYTHONPATH=src python tools/preview_deform.py <char.psd | layer_dir> [out_dir]

Writes <out_dir>/<name>_poses.png (full body) and <name>_face_poses.png (head crop).
Coordinate conventions match core/mesh: model space y-UP normalized [0,1]; UVs v-DOWN; keyform
mesh_offsets are per-vertex Vec2 deltas in model space.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from image2live2d.core import decompose, mesh as meshmod
from image2live2d.core.rig import author_rig, select_template
from image2live2d.pipeline import _lift_occluded_accessories, _safe_landmarks

RES = 1024

POSES = {
    "rest": {},
    "blink": {"ParamEyeLOpen": 0.0, "ParamEyeROpen": 0.0},
    "look_L": {"ParamAngleX": -30.0, "ParamEyeBallX": -1.0},
    "look_R": {"ParamAngleX": 30.0, "ParamEyeBallX": 1.0},
    "mouth_open": {"ParamMouthOpenY": 1.0},
    "lean": {"ParamAngleZ": 30.0, "ParamBodyAngleZ": 10.0},
}
FACE_POSES = ["rest", "blink", "look_L", "look_R", "mouth_open"]


def _interp(param, value):
    kfs = sorted(param.keyforms, key=lambda k: k.value)
    if not kfs:
        return {}
    if value <= kfs[0].value:
        chosen = [(kfs[0], 1.0)]
    elif value >= kfs[-1].value:
        chosen = [(kfs[-1], 1.0)]
    else:
        chosen = [(kfs[0], 1.0)]
        for a, b in zip(kfs, kfs[1:]):
            if a.value <= value <= b.value:
                t = (value - a.value) / (b.value - a.value) if b.value != a.value else 0.0
                chosen = [(a, 1 - t), (b, t)]
                break
    out: dict[str, np.ndarray] = {}
    for kf, wt in chosen:
        for pid, offs in kf.mesh_offsets.items():
            out[pid] = out.get(pid, 0) + np.array(offs, dtype=float) * wt
    return out


def _offsets(params, settings):
    acc: dict[str, np.ndarray] = {}
    for p in params:
        v = settings.get(p.id, 0.0)
        if v == 0.0:
            continue
        for pid, arr in _interp(p, v).items():
            acc[pid] = acc.get(pid, 0) + arr
    return acc


def _affine(dst, src):
    """(a,b,c,d,e,f) for PIL AFFINE mapping output(dst)->input(src): A·coeffs = src."""
    A = np.array([[dst[0][0], dst[0][1], 1], [dst[1][0], dst[1][1], 1], [dst[2][0], dst[2][1], 1]],
                 dtype=float)
    try:
        inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return None
    sx = inv @ np.array([src[0][0], src[1][0], src[2][0]])
    sy = inv @ np.array([src[0][1], src[1][1], src[2][1]])
    return (*sx, *sy)


def render(stack, meshes, params, settings):
    mbp = {m.part_id: m for m in meshes}
    offs = _offsets(params, settings)
    canvas = Image.new("RGBA", (RES, RES), (0, 0, 0, 0))
    for layer in sorted(stack.layers, key=lambda L: L.draw_order):  # bottom -> top
        m = mbp.get(layer.id)
        if m is None:
            continue
        tex = Image.open(layer.texture_path).convert("RGBA")
        tw, th = tex.size
        verts = np.array(m.vertices, dtype=float)
        d = offs.get(layer.id)
        dv = verts + d if d is not None and len(d) == len(verts) else verts
        dst_px = np.column_stack([dv[:, 0] * RES, (1 - dv[:, 1]) * RES])
        src_px = np.array(m.uvs, dtype=float) * [tw, th]
        for tri in m.triangles:
            dst = [tuple(dst_px[i]) for i in tri]
            src = [tuple(src_px[i]) for i in tri]
            xs = [p[0] for p in dst]; ys = [p[1] for p in dst]
            bx0, by0 = max(0, int(np.floor(min(xs)))), max(0, int(np.floor(min(ys))))
            bx1, by1 = min(RES, int(np.ceil(max(xs)))), min(RES, int(np.ceil(max(ys))))
            if bx1 - bx0 < 1 or by1 - by0 < 1:
                continue
            local = [(x - bx0, y - by0) for x, y in dst]
            coeffs = _affine(local, src)
            if coeffs is None:
                continue
            patch = tex.transform((bx1 - bx0, by1 - by0), Image.AFFINE, coeffs, resample=Image.BILINEAR)
            mask = Image.new("L", (bx1 - bx0, by1 - by0), 0)
            md = ImageDraw.Draw(mask)
            md.polygon(local, fill=255)
            md.line(local + [local[0]], fill=255, width=2)  # dilate edges to close seams
            patch.putalpha(Image.composite(patch.getchannel("A"), mask, mask))
            region = Image.alpha_composite(canvas.crop((bx0, by0, bx1, by1)), patch)
            canvas.paste(region, (bx0, by0))
    return canvas


def _head_crop(meshes, layers):
    pts = []
    for L in layers:
        if L.semantic_role.value.startswith(
            ("face", "eye", "eyebrow", "pupil", "mouth", "nose", "hair", "ear", "accessory")
        ):
            m = next((m for m in meshes if m.part_id == L.id), None)
            if m:
                pts.extend(m.vertices)
    if not pts:
        return None
    a = np.array(pts)
    return (max(0, int((a[:, 0].min() - 0.03) * RES)), max(0, int((1 - (a[:, 1].max() + 0.03)) * RES)),
            min(RES, int((a[:, 0].max() + 0.03) * RES)), min(RES, int((1 - (a[:, 1].min() - 0.03)) * RES)))


def _sheet(tiles, path, tile=256):
    sheet = Image.new("RGBA", (tile * len(tiles), tile + 30), (245, 245, 245, 255))
    dr = ImageDraw.Draw(sheet)
    for i, (name, img) in enumerate(tiles):
        sheet.paste(img, (i * tile, 30), img)
        dr.text((i * tile + 6, 8), name, fill=(0, 0, 0, 255))
    sheet.convert("RGB").save(path)
    print("wrote", path)


def _extreme(p):
    """The signed value with the largest magnitude away from default (what 'moves it most')."""
    lo, hi = p.min - p.default, p.max - p.default
    return p.max if abs(hi) >= abs(lo) else p.min


def _max_disp(param, value):
    """Largest per-vertex displacement magnitude (normalized canvas units) and the parts that move."""
    moved = {}
    for pid, arr in _interp(param, value).items():
        mx = max((abs(complex(dx, dy)) for dx, dy in arr), default=0.0)
        if mx > 1e-5:
            moved[pid] = mx
    return moved


# facial params crop to the head; everything else shows the full body
_FACE_PREFIX = ("ParamEye", "ParamMouth", "ParamBrow", "ParamAngle")


def audit_all(stack, meshes, params, out_dir, name):
    crop = _head_crop(meshes, stack.layers)
    rows, lines = [], []
    RUNAWAY = 0.30   # >30% of the canvas in one vertex = almost certainly flung
    for p in sorted(params, key=lambda p: p.id):
        v = _extreme(p)
        moved = _max_disp(p, v)
        worst = max(moved.values(), default=0.0)
        flag = "DEAD " if not moved else ("RUNAWAY" if worst > RUNAWAY else "ok")
        lines.append(f"{flag:8s} {p.id:18s} ->{v:6.1f}  parts={len(moved):2d}  max|disp|={worst:.3f}"
                     + (f"  {max(moved, key=moved.get)}" if moved else ""))
        img = render(stack, meshes, params, {p.id: v})
        face = p.id.startswith(_FACE_PREFIX) and crop
        rows.append((p.id, (img.crop(crop) if face else img).resize((200, 200))))
    # tile 5 per row
    per = 5
    import math as _m
    nrow = _m.ceil(len(rows) / per)
    sheet = Image.new("RGBA", (200 * per, (200 + 22) * nrow), (245, 245, 245, 255))
    dr = ImageDraw.Draw(sheet)
    for i, (label, img) in enumerate(rows):
        x, y = (i % per) * 200, (i // per) * (200 + 22)
        sheet.paste(img, (x, y + 22), img)
        dr.text((x + 4, y + 6), label, fill=(0, 0, 0, 255))
    path = out_dir / f"{name}_all_params.png"
    sheet.convert("RGB").save(path)
    print("\n".join(lines))
    print(f"\n{sum(1 for L in lines if L.startswith('DEAD'))} dead, "
          f"{sum(1 for L in lines if L.startswith('RUNAWAY'))} runaway, of {len(lines)} params")
    print("wrote", path)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    argv = [a for a in sys.argv[1:] if a != "--all"]
    all_mode = "--all" in sys.argv
    src = Path(argv[0])
    out_dir = Path(argv[1]) if len(argv) > 1 else Path("out/preview")
    out_dir.mkdir(parents=True, exist_ok=True)
    name = src.stem if src.suffix == ".psd" else src.name.replace("_layers", "").replace("_relayers", "")
    work = out_dir / f"{name}_work"

    if src.suffix == ".psd":
        stack = decompose.from_psd(src, work)
    else:
        stack = decompose.from_layer_dir(src)
    meshes = meshmod.build_meshes(stack)
    _lift_occluded_accessories(stack, meshes)
    auth = author_rig(stack, meshes, select_template(stack), landmarks=_safe_landmarks(stack))
    params = auth.parameters

    if all_mode:                       # exercise EVERY movable part (one render per parameter)
        audit_all(stack, meshes, params, out_dir, name)
        return

    pr = {p.id: (p.min, p.max) for p in params}
    crop = _head_crop(meshes, stack.layers)

    full, face = [], []
    for nm, raw in POSES.items():
        s = {k: max(pr.get(k, (v, v))[0], min(pr.get(k, (v, v))[1], v)) for k, v in raw.items()}
        img = render(stack, meshes, params, s)
        full.append((nm, img.resize((256, 256))))
        if crop and nm in FACE_POSES:
            face.append((nm, img.crop(crop).resize((256, 256))))
    _sheet(full, str(out_dir / f"{name}_poses.png"))
    if face:
        _sheet(face, str(out_dir / f"{name}_face_poses.png"))


if __name__ == "__main__":
    main()
