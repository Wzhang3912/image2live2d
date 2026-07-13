"""Unit coverage for the representation-alignment geometry in tools/align_pro_model.py — rasterising a
mesh to a footprint and merging near-coincident footprints (depth layers). Pure geometry, no Cubism
core, so it runs in CI; it guards the IoU merge rule that keeps alignment from either under-merging
(depth layers stay split) or chaining the whole character into one blob.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from align_pro_model import merge_overlapping, rasterize_cells  # noqa: E402

_SQUARE = ([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], [(0, 1, 2), (0, 2, 3)])


def test_rasterize_covers_the_footprint():
    verts, tris = _SQUARE
    assert len(rasterize_cells(verts, tris, 4)) == 16          # full unit square -> every cell
    # a quarter square in the corner covers only the low cells
    quarter = ([(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)], [(0, 1, 2), (0, 2, 3)])
    assert len(rasterize_cells(*quarter, 4)) == 4              # 2x2 block of cells


def _cells(*ij):
    return set(ij)


def test_coincident_footprints_merge_but_contained_ones_dont():
    big = {(i, j) for i in range(4) for j in range(4)}         # 16 cells
    base = {(i, j) for i in range(4) for j in range(4)}        # coincident with `big` (IoU 1.0)
    clip = {(1, 1)}                                            # sits INSIDE big, IoU 1/16 -> separate
    far = {(9, 9)}                                             # disjoint
    groups = merge_overlapping({"big": big, "base": base, "clip": clip, "far": far})
    by = {frozenset(g) for g in groups}
    assert frozenset({"big", "base"}) in by                   # depth layers fuse
    assert frozenset({"clip"}) in by                          # a clip over a big region stays its own
    assert frozenset({"far"}) in by


def test_partial_overlap_below_threshold_stays_split():
    a = {(0, 0), (0, 1)}
    b = {(0, 1), (0, 2)}                                       # IoU = 1/3 < 0.6
    groups = merge_overlapping({"a": a, "b": b})
    assert {frozenset(g) for g in groups} == {frozenset({"a"}), frozenset({"b"})}


def test_merge_is_transitive():
    a = {(i, j) for i in range(4) for j in range(4)}
    b = {(i, j) for i in range(4) for j in range(4)} | {(4, 0)}   # ~coincident with a
    c = b | {(4, 1)}                                              # ~coincident with b
    groups = merge_overlapping({"a": a, "b": b, "c": c}, thresh=0.6)
    assert len(groups) == 1 and set(groups[0]) == {"a", "b", "c"}
