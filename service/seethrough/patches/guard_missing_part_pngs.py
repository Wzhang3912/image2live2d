#!/usr/bin/env python3
"""Guard See-through's ``load_parts`` against a missing tag PNG (image2live2d vendored patch).

``further_extr`` -> ``load_parts`` (``common/utils/io_utils.py``) iterates the tags listed in
``info.json`` and calls ``load_part(osp.join(srcp, tag + '.png'))`` for each. ``load_part`` does a bare
``Image.open()`` with no existence check, and ``load_parts`` guards only the *return* value
(``if p is not None``) — so any tag whose PNG is absent raises ``FileNotFoundError`` and fails the whole
decompose. That happens two ways we hit in practice:
  1. the empty-head-crop guard skipped head sub-part extraction, so ``headwear.png`` etc. never exist;
  2. a tag came out empty this run (GPU/precision nondeterminism) and wasn't written (e.g. ``eyes.png``).

This inserts an existence check so an absent tag PNG is skipped (that part is simply dropped) instead of
crashing — turning both cases into graceful degradation (a coarser rig) instead of zero layers.

Content-anchored, idempotent, non-fatal, and ``ast``-verified — same contract as
``guard_empty_head_crop.py``.
"""
from __future__ import annotations

import ast
import sys

MARKER = "image2live2d: skip absent tag PNG"
# The per-tag load call inside load_parts' loop (matched after whitespace strip).
ANCHOR = (
    "p = load_part(osp.join(srcp, tag + '.png'), "
    "rotate=rotate, pad=pad, min_width=min_width, min_sz=min_sz)"
)


def patch_text(text: str) -> tuple[str, str]:
    """Return ``(new_text, status)``. status is patched | already-patched | anchor-not-found."""
    if MARKER in text:
        return text, "already-patched"

    lines = text.splitlines(keepends=True)
    anchor_i = next((i for i, ln in enumerate(lines) if ln.strip() == ANCHOR), None)
    if anchor_i is None:
        return text, "anchor-not-found"

    indent = " " * (len(lines[anchor_i]) - len(lines[anchor_i].lstrip()))
    newline = "\r\n" if lines[anchor_i].endswith("\r\n") else "\n"
    guard = [
        f"{indent}if not osp.exists(osp.join(srcp, tag + '.png')):{newline}",
        f"{indent}    continue  # {MARKER}: head parts absent when refinement is skipped, or a "
        f"tag that came out empty this run{newline}",
    ]
    out = "".join(lines[:anchor_i] + guard + lines[anchor_i:])
    ast.parse(out)
    return out, "patched"


def main(path: str) -> int:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    out, status = patch_text(text)
    if status == "patched":
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(out)
        print(f"guard_missing_part_pngs: patched {path}")
        return 0
    if status == "already-patched":
        print(f"guard_missing_part_pngs: {path} already patched (no-op)")
        return 0
    print(
        f"guard_missing_part_pngs: {status} in {path} — leaving unpatched (See-through may have "
        "changed shape; the missing-tag-PNG crash guard is NOT active)",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: guard_missing_part_pngs.py <io_utils.py>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
