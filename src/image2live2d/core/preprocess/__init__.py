"""Stage 1 — Preprocess. Background removal, normalization, character crop.

Tier-2 seam feeding the (gated) decomposer. Background removal uses ``rembg`` (behind the
``preprocess`` extra); cropping/normalization use Pillow. Pass ``remove_bg=False`` to skip rembg
(useful when the input is already a clean cutout, and to keep this testable without the heavy extra).
"""

from __future__ import annotations

import io
from pathlib import Path

from ..types import ImageAsset, PreparedImage


def prepare(
    image: ImageAsset,
    *,
    work_dir: str | Path | None = None,
    remove_bg: bool = True,
) -> PreparedImage:
    """Produce a clean, character-cropped RGBA cutout (+ alpha matte) from a source image.

    Removes the background (rembg) unless ``remove_bg=False``, crops to the subject's alpha bounding
    box, and writes ``<stem>_prepared.png`` / ``<stem>_alpha.png`` into ``work_dir`` (defaults next to
    the source). Requires Pillow; background removal additionally requires the ``preprocess`` extra.
    """
    from PIL import Image  # local import: keep the contract importable without Pillow

    work = Path(work_dir) if work_dir else image.path.parent
    work.mkdir(parents=True, exist_ok=True)

    with Image.open(image.path) as img:
        rgba = img.convert("RGBA")

    if remove_bg:
        cut = _remove_background(rgba)
    else:
        cut = rgba

    bbox = cut.getbbox()  # crop away fully-transparent margins
    if bbox is not None:
        cut = cut.crop(bbox)

    prepared_path = work / f"{image.path.stem}_prepared.png"
    alpha_path = work / f"{image.path.stem}_alpha.png"
    cut.save(prepared_path)
    cut.getchannel("A").save(alpha_path)

    return PreparedImage(
        path=prepared_path, width=cut.width, height=cut.height, alpha_path=alpha_path
    )


def _remove_background(rgba):
    """Run rembg on an RGBA PIL image, returning an RGBA cutout. Raises a clear error if rembg
    (the ``preprocess`` extra) is not installed."""
    try:
        from rembg import remove
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "preprocess.prepare(remove_bg=True) needs the 'preprocess' extra "
            "(pip install 'image2live2d[preprocess]')"
        ) from exc
    from PIL import Image

    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    out = remove(buf.getvalue())
    return Image.open(io.BytesIO(out)).convert("RGBA")
