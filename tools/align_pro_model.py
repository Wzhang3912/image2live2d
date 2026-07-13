"""Representation alignment — collapse a pro rig's depth-layered drawables into flat spatial parts.

Our dynamics score reads SPARSE, non-overlapping parts (a flat image split into transparent layers, as
our decomposer emits). A hand-built Live2D rig is the opposite: it stacks many co-located meshes at
different depths — a hair strand plus its shade, highlight and back-hair layers all occupy the same
region. Scored directly, every part is "backed" by another, so the free-edge "opens into void" cue
(the score's decisive signal) never fires and everything reads rigid (see tools/calibrate_moc3.py).

This module realigns the representation: it rasterises each part to an occupancy footprint and unions
parts that heavily overlap (the smaller mostly inside the larger — the signature of a shade/back layer
sitting on its base). Merging those depth layers back into one spatial region restores an outer
boundary that opens to void, so the free-edge cue — and calibration against a real rig — works again.

Pure geometry (no Cubism core, no Pillow), so the merge logic is unit-testable on synthetic meshes.
"""

from __future__ import annotations


def _in_triangle(px: float, py: float, a, b, c) -> bool:
    (ax, ay), (bx, by), (cx, cy) = a, b, c
    d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by)
    d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy)
    d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay)
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))


def rasterize_cells(vertices, triangles, res: int) -> set[tuple[int, int]]:
    """The set of ``res``×``res`` grid cells (over the [0,1] frame) a mesh covers — its footprint."""
    cells: set[tuple[int, int]] = set()
    for ia, ib, ic in triangles:
        a, b, c = vertices[ia], vertices[ib], vertices[ic]
        xs = (a[0], b[0], c[0])
        ys = (a[1], b[1], c[1])
        i0, i1 = max(0, int(min(xs) * res)), min(res - 1, int(max(xs) * res))
        j0, j1 = max(0, int(min(ys) * res)), min(res - 1, int(max(ys) * res))
        for j in range(j0, j1 + 1):
            py = (j + 0.5) / res
            for i in range(i0, i1 + 1):
                if (i, j) not in cells and _in_triangle((i + 0.5) / res, py, a, b, c):
                    cells.add((i, j))
    return cells


def merge_overlapping(cell_sets: dict, *, thresh: float = 0.6) -> list[list]:
    """Group part ids whose footprints are nearly **coincident**, transitively. Two parts merge when
    their intersection-over-union (IoU) is at least ``thresh`` — i.e. they cover essentially the same
    region, the signature of a shade / highlight / outline layer stacked on its base. IoU (not
    containment) is deliberate: a small part that merely sits *inside* a much larger one (a hair clip
    over the head) has low IoU and stays separate, so the merge collapses depth layers without chaining
    the whole character into one blob. Returns id-groups; a part coincident with nothing is a singleton."""
    ids = list(cell_sets)
    parent = {k: k for k in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(len(ids)):
        a = ids[i]
        ca = cell_sets[a]
        if not ca:
            continue
        for j in range(i + 1, len(ids)):
            b = ids[j]
            cb = cell_sets[b]
            if not cb:
                continue
            inter = len(ca & cb)
            if inter and inter / (len(ca) + len(cb) - inter) >= thresh:   # IoU
                union(a, b)

    groups: dict = {}
    for k in ids:
        groups.setdefault(find(k), []).append(k)
    return list(groups.values())
