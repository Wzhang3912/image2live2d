"""Render a real ``.moc3`` through the native Cubism core, at arbitrary parameter poses.

Ground-truth renderer: binds ``csmGetDrawableVertexUvs`` / ``RenderOrders`` / ``Opacities`` on top of
the project's ``cubism_core`` bridge, then rasterizes textured triangles against the model's atlas.
This is what Cubism Viewer draws, so it is the right oracle for "what does the user actually see".

It renders the parameter deformation — the input-param poses (arm/leg/body/head, blink, mouth). It does
NOT run the SimplePhysics hair/cloth swing: that is a separate pass the SDK framework applies on top of
the core, and the core alone does not simulate it. That split is exactly right for the thing we use this
for — a seam that opens when a joint bends is a *deformation* defect, visible here; the hair swing is
secondary motion measured separately (see ``tools/physics_excite.py``).
"""
from __future__ import annotations

import ctypes as C
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cubism_core import _Vec2, Model  # noqa: E402


def _bind_extra(lib) -> None:
    lib.csmGetDrawableVertexUvs.restype = C.POINTER(C.POINTER(_Vec2))
    lib.csmGetDrawableVertexUvs.argtypes = [C.c_void_p]
    lib.csmGetDrawableRenderOrders.restype = C.POINTER(C.c_int)
    lib.csmGetDrawableRenderOrders.argtypes = [C.c_void_p]
    lib.csmGetDrawableOpacities.restype = C.POINTER(C.c_float)
    lib.csmGetDrawableOpacities.argtypes = [C.c_void_p]


class Renderer:
    def __init__(self, moc3: str, atlas: str):
        self.m = Model(moc3)
        _bind_extra(self.m._lib)
        self.atlas = np.asarray(Image.open(atlas).convert("RGBA"), dtype=np.float32) / 255.0
        self.ah, self.aw = self.atlas.shape[:2]

    def set(self, **params) -> None:
        self.m.reset()
        for k, v in params.items():
            self.m.set_param(k, v)
        self.m.update()

    def set_many(self, params: dict) -> None:
        self.m.reset()
        for k, v in params.items():
            self.m.set_param(k, v)
        self.m.update()

    def _uvs(self, d):
        uv = self.m._lib.csmGetDrawableVertexUvs(self.m._model)
        n = self.m._vcounts[d]
        return np.array([(uv[d][i].x, uv[d][i].y) for i in range(n)], dtype=np.float32)

    def _verts(self, d):
        pos = self.m._lib.csmGetDrawableVertexPositions(self.m._model)
        n = self.m._vcounts[d]
        return np.array([(pos[d][i].x, pos[d][i].y) for i in range(n)], dtype=np.float32)

    def _tris(self, d):
        idx = self.m._lib.csmGetDrawableIndices(self.m._model)
        n = self.m._icounts[d]
        return np.array([idx[d][i] for i in range(n)], dtype=np.int32).reshape(-1, 3)

    def rest_bounds(self, margin: float = 0.08):
        self.m.reset()
        self.m.update()
        allv = np.vstack([self._verts(d) for d in range(self.m._n_draw)])
        minx, miny = allv.min(0)
        maxx, maxy = allv.max(0)
        pad = margin * max(maxx - minx, maxy - miny)
        return (minx - pad, miny - pad, maxx + pad, maxy + pad)

    def render(self, size: int = 700, bounds=None) -> Image.Image:
        """Rasterize all drawables in render order. ``bounds=(minx,miny,maxx,maxy)`` in model space
        keeps the framing fixed across poses so deformations are comparable."""
        lib, mdl = self.m._lib, self.m._model
        orders = lib.csmGetDrawableRenderOrders(mdl)
        opac = lib.csmGetDrawableOpacities(mdl)
        n = self.m._n_draw

        if bounds is None:
            allv = np.vstack([self._verts(d) for d in range(n)])
            minx, miny = allv.min(0)
            maxx, maxy = allv.max(0)
            pad = 0.05 * max(maxx - minx, maxy - miny)
            bounds = (minx - pad, miny - pad, maxx + pad, maxy + pad)
        minx, miny, maxx, maxy = bounds
        scale = size / max(maxx - minx, maxy - miny)
        W = H = size
        canvas = np.ones((H, W, 3), dtype=np.float32)

        def to_px(v):
            x = (v[:, 0] - minx) * scale
            y = (maxy - v[:, 1]) * scale     # y-up model -> y-down raster
            return np.stack([x, y], 1)

        for d in sorted(range(n), key=lambda i: orders[i]):
            if opac[d] <= 0.01:
                continue
            P = to_px(self._verts(d))
            UV = self._uvs(d)
            UV = np.stack([UV[:, 0] * (self.aw - 1), (1.0 - UV[:, 1]) * (self.ah - 1)], 1)
            for tri in self._tris(d):
                p, uv = P[tri], UV[tri]
                x0, y0 = np.floor(p.min(0)).astype(int)
                x1, y1 = np.ceil(p.max(0)).astype(int)
                x0, y0 = max(x0, 0), max(y0, 0)
                x1, y1 = min(x1, W - 1), min(y1, H - 1)
                if x1 <= x0 or y1 <= y0:
                    continue
                xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
                px = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], 1)
                v0, v1, v2 = p[0], p[1], p[2]
                den = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
                if abs(den) < 1e-9:
                    continue
                w0 = ((v1[1] - v2[1]) * (px[:, 0] - v2[0]) + (v2[0] - v1[0]) * (px[:, 1] - v2[1])) / den
                w1 = ((v2[1] - v0[1]) * (px[:, 0] - v2[0]) + (v0[0] - v2[0]) * (px[:, 1] - v2[1])) / den
                w2 = 1.0 - w0 - w1
                inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
                if not inside.any():
                    continue
                w0, w1, w2 = w0[inside], w1[inside], w2[inside]
                su = np.clip(w0 * uv[0, 0] + w1 * uv[1, 0] + w2 * uv[2, 0], 0, self.aw - 1).astype(int)
                sv = np.clip(w0 * uv[0, 1] + w1 * uv[1, 1] + w2 * uv[2, 1], 0, self.ah - 1).astype(int)
                texel = self.atlas[sv, su]
                a = (texel[:, 3] * opac[d])[:, None]
                tx = px[inside, 0].astype(int)
                ty = px[inside, 1].astype(int)
                canvas[ty, tx] = texel[:, :3] * a + canvas[ty, tx] * (1 - a)

        return Image.fromarray((canvas * 255).astype(np.uint8))
