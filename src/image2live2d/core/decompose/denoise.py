"""Speck removal for decomposer layers — drop the stray alpha blobs See-through flings across a layer.

See-through's layers are not clean cutouts. Besides the faint halo that ``_clean_alpha`` clips, every
layer carries a handful of small, *fully opaque* blobs scattered far from the real content — measured
on 8 real characters: an ``eyebrow`` layer whose two brows sit at x 579-688 also had specks at x 0 and
x 1265-1279, i.e. at the opposite corner of a 1280px canvas.

Downstream we already defend the *bounding box* against this (``mesh.build.alpha_bbox`` weights rows
and columns by solid alpha mass and trims scatter off the ends), so the specks were invisible in the
finished mesh. What that 1-D defence cannot protect is anything that reads the layer's alpha
**structure** rather than its extent — and the L/R splitters do exactly that. ``_split_lr`` accepts a
layer only when its alpha forms *exactly two* horizontal blobs; a 9px-wide speck column is a third run,
so the split is refused and the character is emitted one-eyed / one-browed. Measured across the 8 real
PSDs: **10 of 32** combined facial layers failed to split for this reason alone.

The rule is **relative, not absolute**. The rival this was harvested from (Anime2.5DRig) uses a fixed
40px threshold, which is calibrated to their canvas and does not survive ours — a real ``nose`` layer's
entire content is only 84px of alpha on a 1280px canvas, so an absolute floor in that range deletes the
nose outright.

Size alone is not enough either, because a layer's *own* content is multi-blob at very different
scales: measured over the 8 characters, every ``eyelash`` layer carries a genuine mirrored pair of
lower lashes at 8-12% of the main lash, while a ``lavendergown`` eyebrow's scatter reaches 15% of a
brow. Those bands overlap, so no size ratio can separate them.

What does separate them is **position**. Real secondary art sits on top of the content; scatter is
flung away from it. Taking the ``core`` components (those within 4x of the largest) as the content and
measuring each remaining component's distance to that hull in units of the hull's own span:

    genuine secondary art   0.00 - 0.39 spans      (lower lashes, hat trim, collar)
    scatter                 9.49 - 106  spans      (every speck, on every character)

so ``FAR_FRAC = 1.0`` sits in a 24x-wide empty gap. Size still carries the easy cases (anything under
``MIN_AREA_FRAC`` of the largest component is scatter wherever it lies), and core components are never
dropped however far out they sit — two twintails are content, not noise.

Connectivity is **8-way** on purpose: a thin anti-aliased diagonal (an eyelash, a hair strand) breaks
into disconnected pieces under 4-connectivity, and those pieces would then look like specks next to the
part's main blob.

Requires numpy (decompose extra).
"""

from __future__ import annotations

# A component at least this fraction of the layer's largest one is *content* — it defines the hull the
# rest are judged against, and is never dropped. Every paired facial layer measured (two brows, two
# irides, two lashes) has its twin at 0.87-1.00, so the pair always lands in the core together.
CORE_FRAC = 0.25
# A component smaller than this fraction of the largest is scatter wherever it sits.
MIN_AREA_FRAC = 0.02
# ...and any non-core component further than this many hull-spans from the content is scatter too.
# Measured separation on 8 real characters: content <= 0.39 spans, scatter >= 9.49 spans.
FAR_FRAC = 1.0

_ALPHA_THRESHOLD = 8  # matches mesh.build.DEFAULT_ALPHA_THRESHOLD — below this a texel is transparent


def components(mask):
    """Label the 8-connected components of a 2-D bool array.

    Returns ``(labels, areas)``: ``labels`` is an int32 array with 0 for background and 1..n for the
    components, and ``areas`` maps label -> pixel count. Implemented as run-length labelling with
    union-find (one pass over horizontal runs, not over pixels) so a full-canvas layer stays cheap.
    """
    import numpy as np

    height, width = mask.shape
    parent: list[int] = []

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> int:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
        return min(ra, rb)

    runs: list[tuple[int, int, int, int]] = []  # (y, x0, x1, label)
    prev: list[tuple[int, int, int]] = []       # previous row's (x0, x1, label)
    for y in range(height):
        row = mask[y]
        if not row.any():
            prev = []
            continue
        # run starts/ends via a padded diff, so this is numpy work rather than a per-pixel loop
        edges = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1) - 1
        cur: list[tuple[int, int, int]] = []
        for x0, x1 in zip(starts.tolist(), ends.tolist()):
            label = -1
            for px0, px1, plabel in prev:
                if px0 <= x1 + 1 and x0 <= px1 + 1:   # +1 -> 8-connectivity (diagonal touch counts)
                    label = find(plabel) if label < 0 else union(label, plabel)
            if label < 0:
                label = len(parent)
                parent.append(label)
            cur.append((x0, x1, label))
            runs.append((y, x0, x1, label))
        prev = cur

    labels = np.zeros(mask.shape, dtype=np.int32)
    areas: dict[int, int] = {}
    remap: dict[int, int] = {}
    for y, x0, x1, label in runs:
        root = find(label)
        out = remap.get(root)
        if out is None:
            out = remap[root] = len(remap) + 1
        labels[y, x0:x1 + 1] = out
        areas[out] = areas.get(out, 0) + (x1 - x0 + 1)
    return labels, areas


def speck_labels(labels, areas) -> list[int]:
    """Which component labels are scatter: too small to be content, or too far from it to belong.

    Split out from :func:`drop_specks` so the decision is testable on plain arrays, with no image.
    """
    import numpy as np

    biggest = max(areas.values())
    core = [label for label, area in areas.items() if area >= CORE_FRAC * biggest]
    boxes = {}
    for label in areas:
        ys, xs = np.nonzero(labels == label)
        boxes[label] = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    hx0 = min(boxes[c][0] for c in core)
    hy0 = min(boxes[c][1] for c in core)
    hx1 = max(boxes[c][2] for c in core)
    hy1 = max(boxes[c][3] for c in core)
    span = max(hx1 - hx0, hy1 - hy0, 1)

    out: list[int] = []
    for label, area in areas.items():
        if label in core:
            continue
        if area < MIN_AREA_FRAC * biggest:
            out.append(label)
            continue
        x0, y0, x1, y1 = boxes[label]
        dx = max(hx0 - x1, 0, x0 - hx1)          # 0 when the boxes overlap on that axis
        dy = max(hy0 - y1, 0, y0 - hy1)
        if (dx * dx + dy * dy) ** 0.5 > FAR_FRAC * span:
            out.append(label)
    return out


def drop_specks(image, *, threshold: int = _ALPHA_THRESHOLD):
    """Return ``image`` (RGBA) with its scatter components cleared to fully transparent.

    A layer with one component, or none, is returned unchanged; otherwise see :func:`speck_labels`.
    """
    import numpy as np
    from PIL import Image

    arr = np.array(image.convert("RGBA"))
    mask = arr[:, :, 3] >= threshold
    if not mask.any():
        return image
    labels, areas = components(mask)
    if len(areas) <= 1:
        return image
    specks = speck_labels(labels, areas)
    if not specks:
        return image
    arr[np.isin(labels, specks)] = 0
    return Image.fromarray(arr, "RGBA")
