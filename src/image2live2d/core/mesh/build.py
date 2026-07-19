"""Stage 3 — Mesh. Build a deformation mesh per layer.

A layer is a (typically full-canvas) inpainted RGBA PNG with the part painted in place and the rest
transparent. We lay a ``grid`` x ``grid`` lattice of quad cells over the part, drop cells whose
footprint is fully transparent, triangulate the survivors, and emit per-vertex UVs. Grid meshes are
the convention Live2D and nijilive expect; dropping empty cells keeps the deform grid tight around
the art so warps don't drag dead space.

Because See-through layers are full-canvas, ``build_mesh`` first finds the part's **alpha bounding
box** and lays the grid over just that sub-region — otherwise a small mouth on a 1024px canvas would
get a single cell. Vertices stay in correct canvas/model coordinates; only the grid's *extent* is
tightened.

Coordinate conventions (consistent with ``irr.example`` and the nijilive emitter):
* Model space has **y up**.
* UVs have **v down** (texture row order): ``v = 0`` at the top of the texture.
* A layer bbox is ``(x0, y0, x1, y1)`` in normalized canvas coords with **y down** (image
  convention). ``build_mesh`` flips it into a model-space rect (y up).

The pure ``grid_mesh`` core takes an alpha *sampler* (``(u, v) -> 0..255``) so it is fully testable
without Pillow or any ML extras; ``build_mesh`` is the thin Pillow-backed wrapper.
"""

from __future__ import annotations

from typing import Callable

from ..types import Layer, LayerStack
from ...irr.schema import Mesh, Vec2

# (u, v) in [0, 1], v down -> alpha 0..255.
AlphaSampler = Callable[[float, float], int]

DEFAULT_GRID = 14  # finer lattice -> smoother deformation + tighter coverage of thin features
#                    (hair strands, ankles), fewer seams under motion than the old 10x10
DEFAULT_ALPHA_THRESHOLD = 8  # below this a texel counts as transparent
_BBOX_MASS_FRAC = 0.02       # a row/col with < this fraction of the peak line's solid-alpha mass is
#                              scatter, not content, and is trimmed from the bbox ends (see alpha_bbox)
_BBOX_SOLID_ALPHA = 64       # only texels this opaque count toward the bbox mass — the faint decomposer
#                              halo (alpha ~8-63) is excluded so it can't inflate the box
DEFAULT_CELL_SAMPLES = 3  # NxN probe points per cell when testing coverage

# uv rect = (u0, v0, u1, v1): u0/u1 = left/right, v0/v1 = top/bottom (v down).
UvRect = tuple[float, float, float, float]
FULL_UV: UvRect = (0.0, 0.0, 1.0, 1.0)


def grid_mesh(
    part_id: str,
    rect: tuple[float, float, float, float],
    alpha_at: AlphaSampler,
    *,
    grid: int = DEFAULT_GRID,
    uv_rect: UvRect = FULL_UV,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
    cell_samples: int = DEFAULT_CELL_SAMPLES,
) -> Mesh:
    """Build a grid ``Mesh`` over a model-space ``rect``, keeping only alpha-covered cells.

    ``rect`` is ``(x_min, y_min, x_max, y_max)`` in model space (y up). ``uv_rect`` is the texture
    region that maps onto ``rect`` (defaults to the whole texture). ``alpha_at(u, v)`` samples the
    layer texture with ``u`` left->right and ``v`` top->bottom over [0, 1].

    Cells are kept when any of their ``cell_samples`` x ``cell_samples`` interior probes is at least
    ``alpha_threshold``; unused lattice nodes are dropped and indices remapped. If *no* cell is
    covered (a fully transparent region — a bug upstream) the full grid is kept so we never emit a
    degenerate mesh; ``validate.lint`` flags that case.
    """
    if grid < 1:
        raise ValueError(f"grid must be >= 1, got {grid}")
    x_min, y_min, x_max, y_max = rect
    w = x_max - x_min
    h = y_max - y_min
    u0, v0, u1, v1 = uv_rect

    def node_pos(i: int, j: int) -> Vec2:
        # i across x (0..grid), j up the y axis (0..grid)
        return (x_min + (i / grid) * w, y_min + (j / grid) * h)

    def node_uv(i: int, j: int) -> Vec2:
        u = u0 + (i / grid) * (u1 - u0)
        v = v1 - (j / grid) * (v1 - v0)  # j up -> v down: top row (j==grid) is v0
        return (u, v)

    def probe_uv(uf: float, jf: float) -> tuple[float, float]:
        u = u0 + (uf / grid) * (u1 - u0)
        v = v1 - (jf / grid) * (v1 - v0)
        return u, v

    def cell_covered(ci: int, cj: int) -> bool:
        for s in range(cell_samples):
            uf = ci + (s + 0.5) / cell_samples
            for t in range(cell_samples):
                jf = cj + (t + 0.5) / cell_samples
                u, v = probe_uv(uf, jf)
                if alpha_at(u, v) >= alpha_threshold:
                    return True
        return False

    covered = [
        (ci, cj) for ci in range(grid) for cj in range(grid) if cell_covered(ci, cj)
    ]
    if not covered:
        covered = [(ci, cj) for ci in range(grid) for cj in range(grid)]

    index_of: dict[tuple[int, int], int] = {}
    vertices: list[Vec2] = []
    uvs: list[Vec2] = []

    def use(i: int, j: int) -> int:
        key = (i, j)
        idx = index_of.get(key)
        if idx is None:
            idx = len(vertices)
            index_of[key] = idx
            vertices.append(node_pos(i, j))
            uvs.append(node_uv(i, j))
        return idx

    triangles: list[tuple[int, int, int]] = []
    for ci, cj in covered:
        bl = use(ci, cj)
        br = use(ci + 1, cj)
        tr = use(ci + 1, cj + 1)
        tl = use(ci, cj + 1)
        # winding matches irr.example: (bl, br, tr), (bl, tr, tl)
        triangles.append((bl, br, tr))
        triangles.append((bl, tr, tl))

    return Mesh(part_id=part_id, vertices=vertices, uvs=uvs, triangles=triangles)


def alpha_bbox(
    sample: Callable[[int, int], int], width: int, height: int, threshold: int
) -> tuple[int, int, int, int] | None:
    """Pixel bounding box ``(px0, py0, px1, py1)`` (inclusive) of texels >= ``threshold``.

    Robust to **sparse faint scatter**: some decomposer (See-through) layers carry a thin halo of
    near-transparent pixels flung to the canvas corners. A raw min/max bbox then spans the whole canvas
    even though the real content is a tight blob — which, for a *face* layer, wrecks the head-turn pivot
    (the head detaches and floats off on a turn, measured on two of eight test characters). So a row or
    column is only counted toward the box if its total alpha *mass* is a real fraction of the median
    row/column mass; the scatter's is a rounding error, so it is trimmed from each end. A clean part's
    edges carry substantial mass, so its box is unchanged. Returns ``None`` if every texel is transparent.
    """
    # Mass counts only reasonably-SOLID texels. The scatter halo is near-transparent (measured alpha
    # 8-63 on a See-through mouth layer), while real content — even a thin lip stroke — is opaque
    # (>=~128). Weighting the box by solid mass makes the faint sprinkle contribute a rounding error no
    # matter how many lines it dusts, so it trims cleanly; a mass floor alone missed a scatter band
    # dense enough to clear it. Cells still include everything >= `threshold`, so soft edges stay in the
    # mesh SHAPE — only the box extent ignores the faint stuff.
    solid = max(threshold, _BBOX_SOLID_ALPHA)
    col_mass = [0] * width
    row_mass = [0] * height
    solid_total = 0
    any_texel = False
    for py in range(height):
        for px in range(width):
            a = sample(px, py)
            if a >= threshold:
                any_texel = True
            if a >= solid:
                col_mass[px] += a
                row_mass[py] += a
                solid_total += a
    if not any_texel:
        return None
    if solid_total == 0:                          # a genuinely faint part (soft glow) — keep raw extent
        col_mass = [1 if any(sample(px, py) >= threshold for py in range(height)) else 0
                    for px in range(width)]
        row_mass = [1 if any(sample(px, py) >= threshold for px in range(width)) else 0
                    for py in range(height)]

    def _span(mass: list[int], n: int) -> tuple[int, int]:
        # Reference the PEAK line, not the median: when the scatter dusts more lines than the content
        # occupies (a full-canvas sprinkle vs a small face), the median line is itself scatter, so a
        # median-relative floor trims nothing. The content's peak line always dwarfs the scatter.
        floor = max(mass) * _BBOX_MASS_FRAC       # below this a line is scatter, not content
        lo = 0
        while lo < n and mass[lo] < floor:
            lo += 1
        hi = n - 1
        while hi > lo and mass[hi] < floor:
            hi -= 1
        return lo, hi

    px0, px1 = _span(col_mass, width)
    py0, py1 = _span(row_mass, height)
    return px0, py0, px1, py1


def build_mesh(
    layer: Layer,
    *,
    grid: int = DEFAULT_GRID,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
    tighten: bool = True,
) -> Mesh:
    """Build a grid mesh for ``layer``, clipped (and optionally tightened) to its PNG alpha.

    Reads ``layer.texture_path`` with Pillow. The layer's image-space bbox (y down) is flipped into
    a model-space rect (y up). When ``tighten`` is set, the lattice is laid over the part's alpha
    bounding box rather than the whole layer, so small parts still get adequate resolution.
    """
    from PIL import Image  # local import: keep the core contract importable without Pillow

    with Image.open(layer.texture_path) as img:
        rgba = img.convert("RGBA")
        width, height = rgba.size
        alpha_px = rgba.getchannel("A").load()

    def alpha_at(u: float, v: float) -> int:
        px = min(width - 1, max(0, int(u * width)))
        py = min(height - 1, max(0, int(v * height)))
        return alpha_px[px, py]

    x0, y0, x1, y1 = layer.bbox  # image space, y down
    rect = (x0, 1.0 - y1, x1, 1.0 - y0)  # model space, y up
    uv_rect: UvRect = FULL_UV

    if tighten:
        box = alpha_bbox(lambda px, py: alpha_px[px, py], width, height, alpha_threshold)
        if box is not None:
            px0, py0, px1, py1 = box
            uv_rect = (px0 / width, py0 / height, (px1 + 1) / width, (py1 + 1) / height)
            rect = _subrect(rect, uv_rect)

    return grid_mesh(
        layer.id, rect, alpha_at, grid=grid, uv_rect=uv_rect, alpha_threshold=alpha_threshold
    )


def _subrect(rect: tuple[float, float, float, float], uv: UvRect) -> tuple[float, float, float, float]:
    """Map a uv sub-rect (v down) into the portion of a model rect (y up) it covers."""
    x_min, y_min, x_max, y_max = rect
    u0, v0, u1, v1 = uv
    xw, yh = x_max - x_min, y_max - y_min
    sx0 = x_min + u0 * xw
    sx1 = x_min + u1 * xw
    sy_min = y_max - v1 * yh  # v1 = bottom
    sy_max = y_max - v0 * yh  # v0 = top
    return (sx0, sy_min, sx1, sy_max)


def build_meshes(stack: LayerStack, *, grid: int = DEFAULT_GRID) -> list[Mesh]:
    """Build meshes for every layer in the stack."""
    return [build_mesh(layer, grid=grid) for layer in stack.layers]
