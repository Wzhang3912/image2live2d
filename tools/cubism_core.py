#!/usr/bin/env python3
"""Minimal ctypes binding to the official **Live2D Cubism Core** — the ground-truth runtime.

Loads a real ``.moc3`` and exposes each drawable's rest-pose mesh (vertices + triangles + id) and the
parameter table, plus a helper that finds which drawables a set of parameters actually deforms (by
perturbing the parameter and watching the vertices move). That is exactly what the dynamics calibrator
needs to compare our geometric "needs physics?" verdict against a pro rigger's real ``.physics3.json``.

The Cubism Core is proprietary and NOT shipped here — this only *binds* a copy already installed on the
machine (e.g. VTube Studio's ``Live2DCubismCore.bundle``). Point it at yours with ``--core`` or the
``CUBISM_CORE`` env var; the common VTS location is auto-discovered. Nothing from a model is committed.
"""

from __future__ import annotations

import ctypes as C
import glob
import os
from dataclasses import dataclass
from pathlib import Path

_ALIGN_MOC = 64      # csmAlignofMoc
_ALIGN_MODEL = 16    # csmAlignofModel

# Common places a Cubism Core lives on macOS/Linux/Windows (VTube Studio bundles one).
_CORE_GLOBS = [
    "/Applications/VTube Studio.app/**/Live2DCubismCore.bundle",
    os.path.expanduser("~/Library/Application Support/Steam/steamapps/common/VTube Studio/"
                        "**/Live2DCubismCore.bundle"),
    os.path.expanduser("~/**/Live2DCubismCore.bundle"),
    os.path.expanduser("~/**/libLive2DCubismCore.*"),
]


class _Vec2(C.Structure):
    _fields_ = [("x", C.c_float), ("y", C.c_float)]


def find_core() -> str:
    """Locate a Cubism Core library (``CUBISM_CORE`` env var wins, else the known VTS locations)."""
    env = os.environ.get("CUBISM_CORE")
    if env and Path(env).exists():
        return env
    for pattern in _CORE_GLOBS:
        hits = glob.glob(pattern, recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(
        "Live2DCubismCore not found — set CUBISM_CORE=/path/to/Live2DCubismCore.bundle "
        "(e.g. from a VTube Studio install). The core is proprietary and not shipped with this repo."
    )


def _bind(lib: C.CDLL) -> None:
    p = C.c_void_p
    lib.csmGetLatestMocVersion.restype = C.c_uint
    lib.csmGetMocVersion.restype = C.c_uint
    lib.csmGetMocVersion.argtypes = [p, C.c_uint]
    lib.csmHasMocConsistency.restype = C.c_int
    lib.csmHasMocConsistency.argtypes = [p, C.c_uint]
    lib.csmReviveMocInPlace.restype = p
    lib.csmReviveMocInPlace.argtypes = [p, C.c_uint]
    lib.csmGetSizeofModel.restype = C.c_uint
    lib.csmGetSizeofModel.argtypes = [p]
    lib.csmInitializeModelInPlace.restype = p
    lib.csmInitializeModelInPlace.argtypes = [p, p, C.c_uint]
    lib.csmUpdateModel.argtypes = [p]
    lib.csmReadCanvasInfo.argtypes = [p, C.POINTER(_Vec2), C.POINTER(_Vec2), C.POINTER(C.c_float)]
    for fn in ("csmGetParameterCount", "csmGetDrawableCount"):
        getattr(lib, fn).restype = C.c_int
        getattr(lib, fn).argtypes = [p]
    lib.csmGetParameterIds.restype = C.POINTER(C.c_char_p)
    lib.csmGetDrawableIds.restype = C.POINTER(C.c_char_p)
    for fn in ("csmGetParameterDefaultValues", "csmGetParameterMaximumValues",
               "csmGetParameterMinimumValues", "csmGetParameterValues"):
        getattr(lib, fn).restype = C.POINTER(C.c_float)
        getattr(lib, fn).argtypes = [p]
    lib.csmGetDrawableVertexCounts.restype = C.POINTER(C.c_int)
    lib.csmGetDrawableIndexCounts.restype = C.POINTER(C.c_int)
    lib.csmGetDrawableVertexPositions.restype = C.POINTER(C.POINTER(_Vec2))
    lib.csmGetDrawableIndices.restype = C.POINTER(C.POINTER(C.c_ushort))
    lib.csmGetDrawableOpacities.restype = C.POINTER(C.c_float)
    for fn in ("csmGetParameterIds", "csmGetDrawableIds", "csmGetDrawableVertexCounts",
               "csmGetDrawableIndexCounts", "csmGetDrawableVertexPositions", "csmGetDrawableIndices",
               "csmGetDrawableOpacities"):
        getattr(lib, fn).argtypes = [p]


def _aligned(size: int, align: int):
    """A zeroed buffer plus a pointer into it aligned to ``align`` (kept together so the buffer lives)."""
    raw = C.create_string_buffer(size + align)
    addr = C.addressof(raw)
    ptr = C.c_void_p(addr + (-addr % align))
    return raw, ptr


@dataclass
class Drawable:
    id: str
    vertices: list[tuple[float, float]]     # rest pose, model space (y up), as delivered by the core
    triangles: list[tuple[int, int, int]]


class Model:
    """A loaded Cubism model: drawables (rest-pose meshes) + a writable parameter table."""

    def __init__(self, moc3_path: str, core_path: str | None = None):
        self._lib = C.CDLL(core_path or find_core())
        _bind(self._lib)
        data = Path(moc3_path).read_bytes()
        self._moc_raw, moc_ptr = _aligned(len(data), _ALIGN_MOC)
        C.memmove(moc_ptr, data, len(data))
        if not self._lib.csmHasMocConsistency(moc_ptr, len(data)):
            raise ValueError(f"{moc3_path}: failed Cubism moc consistency check")
        moc = self._lib.csmReviveMocInPlace(moc_ptr, len(data))
        model_size = self._lib.csmGetSizeofModel(moc)
        self._model_raw, model_ptr = _aligned(model_size, _ALIGN_MODEL)
        self._model = self._lib.csmInitializeModelInPlace(moc, model_ptr, model_size)
        self._read_static()

    def _read_static(self) -> None:
        lib, m = self._lib, self._model
        self.param_ids = [lib.csmGetParameterIds(m)[i].decode()
                          for i in range(lib.csmGetParameterCount(m))]
        self._defaults = lib.csmGetParameterDefaultValues(m)
        self._maxes = lib.csmGetParameterMaximumValues(m)
        self._mins = lib.csmGetParameterMinimumValues(m)
        self._values = lib.csmGetParameterValues(m)          # writable in-place
        self._param_ix = {pid: i for i, pid in enumerate(self.param_ids)}
        self._n_draw = lib.csmGetDrawableCount(m)
        self._draw_ids = [lib.csmGetDrawableIds(m)[i].decode() for i in range(self._n_draw)]
        self._vcounts = lib.csmGetDrawableVertexCounts(m)
        self._icounts = lib.csmGetDrawableIndexCounts(m)

    def reset(self) -> None:
        for i in range(len(self.param_ids)):
            self._values[i] = self._defaults[i]
        self.update()

    def update(self) -> None:
        self._lib.csmUpdateModel(self._model)

    def set_param(self, pid: str, value: float) -> bool:
        i = self._param_ix.get(pid)
        if i is None:
            return False
        self._values[i] = value
        return True

    def opacity_of(self, drawable_id: str) -> float:
        """The drawable's current opacity (0..1) as the core computes it from the live parameter table —
        the runtime-truth read for opacity keyforms, the analogue of ``_positions`` for geometry. Call
        ``update()`` (or ``set_param`` then ``update``) first so the value reflects the current pose."""
        ops = self._lib.csmGetDrawableOpacities(self._model)
        return float(ops[self._draw_ids.index(drawable_id)])

    def _positions(self) -> list[list[tuple[float, float]]]:
        pos = self._lib.csmGetDrawableVertexPositions(self._model)
        out = []
        for d in range(self._n_draw):
            vp, vc = pos[d], self._vcounts[d]
            out.append([(vp[v].x, vp[v].y) for v in range(vc)])
        return out

    def drawables(self) -> list[Drawable]:
        """Rest-pose drawables (parameters reset to defaults)."""
        self.reset()
        idx = self._lib.csmGetDrawableIndices(self._model)
        verts = self._positions()
        out = []
        for d in range(self._n_draw):
            ip, ic = idx[d], self._icounts[d]
            tris = [(ip[k], ip[k + 1], ip[k + 2]) for k in range(0, ic, 3)]
            out.append(Drawable(self._draw_ids[d], verts[d], tris))
        return out

    def drawables_moved_by(self, param_ids, *, eps: float = 1e-3) -> set[str]:
        """The set of drawable ids that visibly move when any of ``param_ids`` is driven to an extreme
        (max, then min) with all others at default — i.e. the drawables those parameters deform."""
        present = [p for p in param_ids if p in self._param_ix]
        self.reset()
        base = self._positions()
        moved: set[str] = set()
        for pid in present:
            i = self._param_ix[pid]
            for target in (self._maxes[i], self._mins[i]):
                self.reset()
                self._values[i] = target
                self.update()
                now = self._positions()
                for d in range(self._n_draw):
                    if self._draw_ids[d] in moved:
                        continue
                    if any(abs(a[0] - b[0]) > eps or abs(a[1] - b[1]) > eps
                           for a, b in zip(now[d], base[d])):
                        moved.add(self._draw_ids[d])
        self.reset()
        return moved


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Dump a .moc3's drawables + parameters via the Cubism core.")
    ap.add_argument("moc3", type=Path)
    ap.add_argument("--core", default=None, help="path to Live2DCubismCore (else auto/CUBISM_CORE)")
    args = ap.parse_args(argv)
    model = Model(str(args.moc3), args.core)
    draws = model.drawables()
    print(f"{len(draws)} drawables, {len(model.param_ids)} parameters")
    print("first drawables:", [d.id for d in draws[:12]])
    print("first params:   ", model.param_ids[:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
