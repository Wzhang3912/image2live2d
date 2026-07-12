"""Offline deformation renderer — rasterize an IRR rig at given parameter values to a PIL image.

An **approximate** piecewise-affine (per-triangle) software warp: it applies each parameter's
per-vertex keyform offsets to the part meshes and composites the textures in draw order. Not a
substitute for nijigenerate's renderer, but good enough to *see* the rig deform (blink, head-turn,
mouth, lean) headlessly — used by the web app's live preview and ``tools/preview_deform.py``.

Coordinate conventions match ``core/mesh``: model space y-UP normalized [0, 1]; UVs v-DOWN; keyform
``mesh_offsets`` are per-vertex Vec2 deltas in model space. Needs numpy + Pillow.
"""
from __future__ import annotations

from typing import Iterable

# Facial params crop to the head in head-zoom views; everything else shows the full body.
FACE_PREFIX = ("ParamEye", "ParamMouth", "ParamBrow", "ParamAngle")


def _interp(param, value: float) -> dict:
    """Per-vertex {part_id: ndarray(N,2)} offset for ``param`` at ``value`` (linear between keyforms)."""
    import numpy as np

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
    out: dict = {}
    for kf, wt in chosen:
        for pid, offs in kf.mesh_offsets.items():
            out[pid] = out.get(pid, 0) + np.array(offs, dtype=float) * wt
    return out


def _offsets(params, settings: dict) -> dict:
    """Sum per-vertex offsets across all params at their ``settings`` value (default 0 = rest)."""
    acc: dict = {}
    for p in params:
        v = settings.get(p.id, 0.0)
        if v == 0.0:
            continue
        for pid, arr in _interp(p, v).items():
            acc[pid] = acc.get(pid, 0) + arr
    return acc


def _affine(dst, src):
    """(a,b,c,d,e,f) for PIL AFFINE mapping output(dst)->input(src): A·coeffs = src."""
    import numpy as np

    A = np.array([[dst[0][0], dst[0][1], 1], [dst[1][0], dst[1][1], 1], [dst[2][0], dst[2][1], 1]],
                 dtype=float)
    try:
        inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return None
    sx = inv @ np.array([src[0][0], src[1][0], src[2][0]])
    sy = inv @ np.array([src[0][1], src[1][1], src[2][1]])
    return (*sx, *sy)


def _rot_about(V, pivot, roll, yaw, pitch):
    """Apply a group node rotation (matching the emitter): yaw=x-squash, pitch=y-squash, roll=in-plane
    rotation, all about ``pivot`` in model space (y-up). ``V`` is an (N,2) ndarray."""
    import math
    import numpy as np

    if roll == 0.0 and yaw == 0.0 and pitch == 0.0:
        return V
    px, py = pivot
    d = V - np.array([px, py])
    d = d * np.array([math.cos(yaw), math.cos(pitch)])  # pseudo-3D squash
    c, s = math.cos(roll), math.sin(roll)
    out = np.column_stack([d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c])
    return out + np.array([px, py])


_HEAD_TURN = {"ParamAngleX", "ParamAngleY", "ParamAngleZ"}
_BODY_TURN = {"ParamBodyAngleX", "ParamBodyAngleY", "ParamBodyAngleZ"}


def _head_ids(drawn):
    from .backends.nijilive.puppet import head_group_ids
    return head_group_ids([(L, m) for L, m in drawn])


def _grouped_offsets(params, settings, head_ids, body_ids):
    """Sum keyform offsets, but skip head-turn params on head parts and body-turn params on body parts
    (those move via the group rotation instead — matching the emitter). Others (blink/mouth/hair/skirt/
    breath, and the neck follow-through) stay as per-vertex deform."""
    acc: dict = {}
    for p in params:
        v = settings.get(p.id, 0.0)
        if v == 0.0:
            continue
        skip = head_ids if p.id in _HEAD_TURN else (body_ids if p.id in _BODY_TURN else set())
        for pid, arr in _interp(p, v).items():
            if pid in skip:
                continue
            acc[pid] = acc.get(pid, 0) + arr
    return acc


def _angles(params, settings, rot_map):
    """(roll, yaw, pitch) radians for a group from its ParamAngle*/ParamBodyAngle* settings."""
    pr = {p.id: p for p in params}
    out = {"transform.r.z": 0.0, "transform.r.y": 0.0, "transform.r.x": 0.0}
    for pid, (channel, max_rad) in rot_map.items():
        p = pr.get(pid)
        v = settings.get(pid, 0.0)
        if p is not None and v:
            denom = max(abs(p.min), abs(p.max), 1e-6)
            out[channel] = (v / denom) * max_rad
    return out["transform.r.z"], out["transform.r.y"], out["transform.r.x"]


def render_pose(stack, meshes, params, settings: dict | None = None, *, res: int = 512):
    """Render the rig at ``settings`` (``{param_id: value}``) to an RGBA ``res``x``res`` PIL image.

    Mirrors the emitter's node hierarchy: head-turn (ParamAngle*) and body-sway (ParamBodyAngle*) move
    their group as ONE rigid unit (rotation about a pivot) instead of per-part keyform warps — so the
    preview matches nijigenerate's connected-body motion. Local params (blink/mouth/hair/skirt) stay
    per-vertex."""
    import numpy as np
    from PIL import Image, ImageDraw
    from .backends.nijilive.puppet import _HEAD_ROT, _BODY_ROT

    settings = settings or {}
    mbp = {m.part_id: m for m in meshes}

    # group hierarchy state (ids, pivots, angles)
    drawn = [(L, mbp[L.id]) for L in stack.layers if L.id in mbp]
    head_ids = _head_ids(drawn)
    body_ids = {L.id for L, _ in drawn if L.id not in head_ids}
    all_pts = np.array([v for _, m in drawn for v in m.vertices], dtype=float) if drawn else np.zeros((1, 2))
    feet_pivot = ((all_pts[:, 0].min() + all_pts[:, 0].max()) / 2.0, all_pts[:, 1].min())
    hpts = np.array([v for L, m in drawn if L.id in head_ids for v in m.vertices], dtype=float)
    head_pivot = ((hpts[:, 0].min() + hpts[:, 0].max()) / 2.0, hpts[:, 1].min()) if len(hpts) else feet_pivot
    h_roll, h_yaw, h_pitch = _angles(params, settings, _HEAD_ROT)
    b_roll, b_yaw, b_pitch = _angles(params, settings, _BODY_ROT)

    # offsets: skip group-turn params for their own group's parts (the group transform replaces them)
    offs = _grouped_offsets(params, settings, head_ids, body_ids)

    canvas = Image.new("RGBA", (res, res), (0, 0, 0, 0))
    for layer in sorted(stack.layers, key=lambda L: L.draw_order):  # bottom -> top
        m = mbp.get(layer.id)
        if m is None:
            continue
        tex = Image.open(layer.texture_path).convert("RGBA")
        tw, th = tex.size
        verts = np.array(m.vertices, dtype=float)
        d = offs.get(layer.id)
        dv = verts + d if d is not None and len(d) == len(verts) else verts
        if layer.id in head_ids:                      # head group: head rot, then body rot (nested)
            dv = _rot_about(dv, head_pivot, h_roll, h_yaw, h_pitch)
            dv = _rot_about(dv, feet_pivot, b_roll, b_yaw, b_pitch)
        elif layer.id in body_ids:                    # body group: body rot about the feet
            dv = _rot_about(dv, feet_pivot, b_roll, b_yaw, b_pitch)
        dst_px = np.column_stack([dv[:, 0] * res, (1 - dv[:, 1]) * res])
        src_px = np.array(m.uvs, dtype=float) * [tw, th]
        for tri in m.triangles:
            dst = [tuple(dst_px[i]) for i in tri]
            src = [tuple(src_px[i]) for i in tri]
            xs = [p[0] for p in dst]
            ys = [p[1] for p in dst]
            bx0, by0 = max(0, int(np.floor(min(xs)))), max(0, int(np.floor(min(ys))))
            bx1, by1 = min(res, int(np.ceil(max(xs)))), min(res, int(np.ceil(max(ys))))
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


def max_displacement(param, value: float) -> dict:
    """{part_id: max per-vertex |displacement|} for ``param`` at ``value`` (audit helper)."""
    moved = {}
    for pid, arr in _interp(param, value).items():
        mx = max((float((dx * dx + dy * dy) ** 0.5) for dx, dy in arr), default=0.0)
        if mx > 1e-5:
            moved[pid] = mx
    return moved


def extreme_value(p):
    """Signed value with the largest magnitude from default (what moves the param most)."""
    return p.max if abs(p.max - p.default) >= abs(p.min - p.default) else p.min


def head_crop(meshes, layers, res: int, roles: Iterable[str] = (
    "face", "eye", "eyebrow", "pupil", "mouth", "nose", "hair", "ear", "accessory")):
    """Pixel crop box (in render space, y-down) around the head parts, or None."""
    import numpy as np

    pts = []
    rt = tuple(roles)
    for L in layers:
        if L.semantic_role.value.startswith(rt):
            m = next((m for m in meshes if m.part_id == L.id), None)
            if m:
                pts.extend(m.vertices)
    if not pts:
        return None
    a = np.array(pts)
    return (max(0, int((a[:, 0].min() - 0.03) * res)), max(0, int((1 - (a[:, 1].max() + 0.03)) * res)),
            min(res, int((a[:, 0].max() + 0.03) * res)), min(res, int((1 - (a[:, 1].min() - 0.03)) * res)))
