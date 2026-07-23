#!/usr/bin/env python3
"""Guard See-through's ``apply_layerdiff`` against an empty head crop (image2live2d vendored patch).

See-through's v3 layerdiff derives the head region from the ``'head'`` body segment's alpha. For some
non-human faces (e.g. the KDE dragon mascots) that segment comes out empty, so the head crop is
zero-size and ``center_square_pad_resize`` -> ``cv2.resize`` dies with
``cv2.error (-215) !ssize.empty()`` — failing the WHOLE decompose and returning zero layers.

This rewrites the head sub-part refinement block so it runs only when the crop is non-degenerate.
Otherwise it skips head sub-part extraction and lets ``apply_marigold`` + ``further_extr`` still
assemble the coarse body layers (front/back hair, head, neck, topwear, tail, wings, ...) into a PSD —
a basic rig instead of nothing. ``further_extr`` already guards every head sub-part with
``if tag in tag2pinfo``, so the missing eyes/mouth/nose degrade gracefully.

Design notes:
- **Content-anchored**, not line-numbered: tolerates upstream line drift.
- **Idempotent**: a second run is a no-op (detects the marker).
- **Non-fatal**: if the expected code isn't found (upstream changed shape), it leaves the file
  untouched and exits non-zero so the deploy script can warn and continue serving unpatched (the 3/5
  inputs that already work keep working) rather than aborting the whole boot.
- **Self-verifying**: the rewritten source is ``ast.parse``-d before being written back.

Usage: ``python guard_empty_head_crop.py /opt/see-through/common/utils/inference_utils.py``
"""
from __future__ import annotations

import ast
import sys

MARKER = "image2live2d: empty head crop"
# The exact statement that crashes on a zero-size crop (matched after whitespace strip).
ANCHOR = (
    "input_head, pad_size, pad_pos = center_square_pad_resize("
    "input_head, resolution, return_pad_info=True)"
)


def patch_text(text: str) -> tuple[str, str]:
    """Return ``(new_text, status)``. status is patched | already-patched | anchor-not-found."""
    if MARKER in text:
        return text, "already-patched"

    lines = text.splitlines(keepends=True)
    anchor_i = next((i for i, ln in enumerate(lines) if ln.strip() == ANCHOR), None)
    if anchor_i is None:
        return text, "anchor-not-found"

    base_indent = len(lines[anchor_i]) - len(lines[anchor_i].lstrip())

    # The head-refine block runs from the anchor down to the next non-blank line that dedents below
    # the anchor's indent (the enclosing ``elif``/``else`` or function end).
    end_i = len(lines)
    for j in range(anchor_i + 1, len(lines)):
        if not lines[j].strip():
            continue
        if (len(lines[j]) - len(lines[j].lstrip())) < base_indent:
            end_i = j
            break

    pad = " " * base_indent
    guard = [
        f"{pad}if ih == 0 or iw == 0:\n",
        f"{pad}    # {MARKER}: the 'head' body segment came out empty (e.g. a non-human face),\n",
        f"{pad}    # so the head crop is degenerate. Skip head sub-part refinement -> marigold and\n",
        f"{pad}    # further_extr still assemble the coarse body layers into a PSD instead of\n",
        f"{pad}    # crashing in cv2.resize on a zero-size image.\n",
        f"{pad}    print('{MARKER} -> skipping head sub-part refinement')\n",
        f"{pad}else:\n",
    ]
    # Re-indent the guarded block one level deeper, leaving blank lines blank.
    body = [ln if not ln.strip() else ("    " + ln) for ln in lines[anchor_i:end_i]]

    out = "".join(lines[:anchor_i] + guard + body + lines[end_i:])
    ast.parse(out)  # fail loudly rather than write syntactically broken source
    return out, "patched"


def main(path: str) -> int:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    out, status = patch_text(text)
    if status == "patched":
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(out)
        print(f"guard_empty_head_crop: patched {path}")
        return 0
    if status == "already-patched":
        print(f"guard_empty_head_crop: {path} already patched (no-op)")
        return 0
    print(
        f"guard_empty_head_crop: {status} in {path} — leaving unpatched (See-through may have "
        "changed shape; the empty-head-crop crash guard is NOT active)",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: guard_empty_head_crop.py <inference_utils.py>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
