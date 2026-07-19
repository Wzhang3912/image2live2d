"""Stage 3 (mesh) tests.

The pure ``grid_mesh`` core is exercised with synthetic alpha samplers (no Pillow / ML deps). One
``build_mesh`` test is guarded by Pillow availability so the suite stays green on a bare install.
"""

from __future__ import annotations

import importlib.util

import pytest

from image2live2d.core.mesh import build_mesh, grid_mesh
from image2live2d.core.types import Layer
from image2live2d.irr.schema import Mesh, SemanticRole

UNIT = (0.0, 0.0, 1.0, 1.0)


def _opaque(_u: float, _v: float) -> int:
    return 255


def test_full_coverage_grid_shape():
    m = grid_mesh("p", UNIT, _opaque, grid=4)
    assert isinstance(m, Mesh)  # constructs => passes IRR geometry validator
    # fully covered 4x4 grid: (grid+1)^2 vertices, 2 triangles per cell
    assert len(m.vertices) == 25
    assert len(m.uvs) == 25
    assert len(m.triangles) == 4 * 4 * 2


def test_uv_corners_and_orientation():
    m = grid_mesh("p", UNIT, _opaque, grid=1)  # single quad
    # one cell -> 4 verts in use order bl, br, tr, tl
    assert m.vertices == [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    # v is down: bottom row (model y=0) is v=1, top row (model y=1) is v=0
    assert m.uvs == [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
    assert m.triangles == [(0, 1, 2), (0, 2, 3)]


def test_transparent_cells_are_dropped():
    # opaque only on the left half (u < 0.5)
    def left_half(u: float, _v: float) -> int:
        return 255 if u < 0.5 else 0

    m = grid_mesh("p", UNIT, left_half, grid=4)
    # 4x4 grid, left 2 columns kept -> 2*4 = 8 cells, 16 tris
    assert len(m.triangles) == 8 * 2
    # every kept vertex sits in the left portion (x <= 0.5)
    assert all(x <= 0.5 + 1e-9 for x, _ in m.vertices)
    # and fewer than the full lattice
    assert len(m.vertices) < 25


def test_indices_in_range_after_remap():
    def diagonal(u: float, v: float) -> int:
        # keep a diagonal band -> forces non-contiguous vertex reuse / remapping
        return 255 if abs(u - v) < 0.2 else 0

    m = grid_mesh("p", UNIT, diagonal, grid=6)
    n = len(m.vertices)
    assert n >= 3
    for tri in m.triangles:
        assert all(0 <= idx < n for idx in tri)


def test_empty_layer_falls_back_to_full_grid():
    def empty(_u: float, _v: float) -> int:
        return 0

    m = grid_mesh("p", UNIT, empty, grid=3)
    # no degenerate mesh: full grid retained
    assert len(m.triangles) == 3 * 3 * 2


def test_bbox_maps_into_model_rect():
    # a sub-region bbox should place verts inside that rect (model y flipped from image y)
    m = grid_mesh("p", (0.25, 0.5, 0.75, 1.0), _opaque, grid=1)
    xs = [x for x, _ in m.vertices]
    ys = [y for _, y in m.vertices]
    assert min(xs) == 0.25 and max(xs) == 0.75
    assert min(ys) == 0.5 and max(ys) == 1.0


_HAS_PIL = importlib.util.find_spec("PIL") is not None


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not installed")
def test_build_mesh_reads_alpha_from_png(tmp_path):
    from PIL import Image

    # left half opaque, right half transparent
    img = Image.new("RGBA", (16, 16), (255, 0, 0, 0))
    for x in range(8):
        for y in range(16):
            img.putpixel((x, y), (255, 0, 0, 255))
    p = tmp_path / "layer.png"
    img.save(p)

    layer = Layer(
        id="face_base",
        semantic_role=SemanticRole.face_base,
        texture_path=p,
        draw_order=0,
        width=16,
        height=16,
    )
    m = build_mesh(layer, grid=4)
    # tighten lays the full grid over the part's alpha bbox (left half) -> all verts on the left,
    # full 4x4 resolution concentrated there
    assert all(x <= 0.5 + 1e-9 for x, _ in m.vertices)
    assert len(m.triangles) == 4 * 4 * 2


# --- alpha_bbox robustness to faint decomposer scatter --------------------------------------------
from image2live2d.core.mesh.build import alpha_bbox  # noqa: E402


def test_alpha_bbox_ignores_faint_corner_scatter():
    """See-through layers can carry a near-transparent halo dusted across the whole canvas. A raw
    min/max box then spans everything; for a face layer that wrecks the head-turn pivot (the head
    detaches and floats off on a turn, seen on 2 of 8 test characters). The box must tighten to the
    SOLID content and ignore the faint sprinkle."""
    solid = {(x, y) for x in range(40, 60) for y in range(40, 60)}   # opaque blob
    scatter = {(0, 0), (99, 0), (0, 99), (99, 99), (3, 80), (95, 8)}  # faint dust in the corners

    def sample(px, py):
        if (px, py) in solid:
            return 255
        if (px, py) in scatter:
            return 30          # below the solid cutoff -> must not count toward the box
        return 0

    assert alpha_bbox(sample, 100, 100, threshold=8) == (40, 40, 59, 59)


def test_alpha_bbox_survives_a_dense_scatter_band():
    """The scatter isn't always a few corner pixels — one real mouth layer had a faint band touching
    more rows than the content occupied. A peak-relative mass floor alone missed it; weighting only
    SOLID texels is what fixes it, so a thin opaque stroke still wins over a wide faint band."""
    stroke = {(x, 50) for x in range(30, 70)}                        # thin opaque horizontal stroke
    band = {(x, y) for x in range(0, 100) for y in (10, 90)}         # wide faint band, 2 full rows

    def sample(px, py):
        if (px, py) in stroke:
            return 255
        if (px, py) in band:
            return 40          # faint -> excluded from the box
        return 0

    x0, y0, x1, y1 = alpha_bbox(sample, 100, 100, threshold=8)
    assert (y0, y1) == (50, 50)                                       # box is the stroke, not the band
    assert (x0, x1) == (30, 69)


def test_alpha_bbox_keeps_a_genuinely_faint_part():
    """A part that is faint EVERYWHERE (a soft glow, no solid core) must not vanish — fall back to the
    plain threshold extent rather than returning nothing."""
    glow = {(x, y) for x in range(20, 40) for y in range(20, 40)}

    def sample(px, py):
        return 30 if (px, py) in glow else 0

    assert alpha_bbox(sample, 100, 100, threshold=8) == (20, 20, 39, 39)
