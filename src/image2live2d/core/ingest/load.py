"""Stage 0 — Ingest. Acquire a source image (user upload, URL, or generation).

Tier-2 seam: this feeds the (gated) decomposer. Loading a file/bytes/URL is implemented here; the
GPT-image generation front door stays gated until Phase 6 (needs an API key + hosting).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ..types import ImageAsset


def load_image(source: str | Path | bytes, *, work_dir: str | Path | None = None) -> ImageAsset:
    """Load an image from a file path, raw bytes, or an http(s) URL into an ``ImageAsset``.

    Bytes/URL inputs are persisted into ``work_dir`` (a temp dir if omitted) so downstream stages can
    read them from disk. Requires Pillow (to read the dimensions)."""
    from PIL import Image  # local import: keep the contract importable without Pillow

    if isinstance(source, bytes):
        path = _persist(source, work_dir, "upload.png")
    elif isinstance(source, str) and source.startswith(("http://", "https://")):
        import urllib.request
        from urllib.parse import urlparse

        with urllib.request.urlopen(source) as resp:  # noqa: S310 (caller-supplied URL)
            data = resp.read()
        name = Path(urlparse(source).path).name or "download.png"
        path = _persist(data, work_dir, name)
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"image not found: {path}")

    with Image.open(path) as img:
        width, height = img.size
    return ImageAsset(path=path, width=width, height=height)


def _persist(data: bytes, work_dir: str | Path | None, name: str) -> Path:
    base = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="i2l_ingest_"))
    base.mkdir(parents=True, exist_ok=True)
    path = base / name
    path.write_bytes(data)
    return path


def generate_image(prompt: str, *, out_dir: Path) -> ImageAsset:
    """Generate a character image via an image-generation API (gated, Phase 6 front door).

    Needs an API key + network; not part of the local Tier-1 product. Use ``load_image`` for uploads.
    """
    raise NotImplementedError(
        "ingest.generate_image is the Phase 6 hosted front door (needs an image-gen API key); "
        "use load_image for local uploads/URLs"
    )
