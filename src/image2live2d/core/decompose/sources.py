"""Stage 2 — Decompose. Single image -> semantic, inpainted, depth-ordered layers.

The production path wraps See-through (heavy ML; ``decompose``). Until that lands, ``from_layer_dir``
provides a **synthetic seam**: it assembles a ``LayerStack`` from a folder of pre-separated,
role-named PNGs. This lets the entire downstream spine (mesh -> rig -> IRR -> .inp) run and be
verified headless, with See-through as a drop-in replacement for this one function.

Layer files follow ``{draw_order}_{semantic_role}.png`` (e.g. ``00_face_base.png``,
``10_eye_white_l.png``, ``40_hair_front.png``). The integer prefix is the draw order; the remainder
must be a ``SemanticRole`` value. PNG dimensions are read from the IHDR chunk with the stdlib, so
this seam needs no image library.

``from_psd`` bridges See-through's actual output: it reads a layered PSD, maps each layer's name to
a ``SemanticRole``, normalizes every layer to a full-canvas PNG, and reuses ``from_layer_dir``. So
once See-through (R&D, needs a CUDA GPU) writes a ``.psd``, it drops straight into the pipeline.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..types import Layer, LayerStack, PreparedImage
from ...irr.schema import SemanticRole

_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def decompose(image: PreparedImage) -> LayerStack:
    """Run See-through to split the character into inpainted, semantically-labeled layers.

    TODO(phase1/see-through): wrap the See-through / ComfyUI pipeline; map its labels to
    ``SemanticRole``; inpaint occluded regions. Until then use ``from_layer_dir`` (synthetic seam).
    """
    raise NotImplementedError("decompose.decompose — use from_layer_dir until See-through is wired")


def png_size(path: str | Path) -> tuple[int, int]:
    """Read ``(width, height)`` from a PNG's IHDR chunk using only the stdlib."""
    with open(path, "rb") as f:
        header = f.read(24)
    if header[:8] != _PNG_SIG:
        raise ValueError(f"{path}: not a PNG (bad signature)")
    # IHDR data starts at byte 16: width (u32 BE), height (u32 BE).
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def parse_layer_name(stem: str) -> tuple[int, str]:
    """Parse ``{order}_{role}`` -> ``(order, role)``. Role may contain underscores."""
    head, sep, role = stem.partition("_")
    if not sep or not head.isdigit() or not role:
        raise ValueError(
            f"layer filename {stem!r} must be '{{draw_order}}_{{semantic_role}}', e.g. '10_eye_l'"
        )
    return int(head), role


def from_layer_dir(layer_dir: str | Path) -> LayerStack:
    """Assemble a ``LayerStack`` from a directory of ``{order}_{role}.png`` files.

    Each layer's bbox defaults to the full canvas ``(0, 0, 1, 1)`` — correct for full-canvas
    See-through-style layers, where the mesh stage tightens to the part's alpha. The canvas size is
    the max width/height across layers.
    """
    layer_dir = Path(layer_dir)
    pngs = sorted(layer_dir.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"no .png layers found in {layer_dir}")

    layers: list[Layer] = []
    canvas_w = canvas_h = 0
    for png in pngs:
        order, role_str = parse_layer_name(png.stem)
        try:
            role = SemanticRole(role_str)
        except ValueError as exc:
            raise ValueError(f"{png.name}: unknown semantic role {role_str!r}") from exc
        width, height = png_size(png)
        canvas_w, canvas_h = max(canvas_w, width), max(canvas_h, height)
        layers.append(
            Layer(
                id=png.stem,
                semantic_role=role,
                texture_path=png,
                draw_order=order,
                width=width,
                height=height,
            )
        )

    layers.sort(key=lambda layer: layer.draw_order)
    return LayerStack(layers=layers, canvas_width=canvas_w, canvas_height=canvas_h)


# --------------------------------------------------------------------------------------------------
# PSD adapter (See-through output -> LayerStack)
# --------------------------------------------------------------------------------------------------
RoleMapper = Callable[[str], SemanticRole]


@dataclass
class RawLayer:
    """One extracted PSD layer: its name, an RGBA image (layer-extent crop), and its top-left offset
    on the canvas. Kept backend/lib-agnostic so the assembly logic is testable without psd-tools."""

    name: str
    image: Any  # PIL.Image.Image (RGBA)
    offset: tuple[int, int]  # (left, top) in canvas pixels


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def role_from_layer_name(name: str, *, default: SemanticRole = SemanticRole.other) -> SemanticRole:
    """Map a free-form PSD layer name to a ``SemanticRole`` with keyword heuristics.

    Handles side detection (``left``/``right``/trailing ``l``/``r`` tokens) and common synonyms.
    See-through's exact label set isn't pinned down, so this is deliberately fuzzy; pass a custom
    ``role_map`` to ``from_psd`` once the real labels are known. An exact ``SemanticRole`` value
    always wins.
    """
    n = _norm(name)
    if not n:
        return default
    try:
        return SemanticRole(n)  # exact match (e.g. "eye_white_l")
    except ValueError:
        pass

    tokens = set(n.split("_"))
    side = "l" if tokens & {"l", "left", "lt"} else "r" if tokens & {"r", "right", "rt"} else None
    compact = n.replace("_", "")  # so "twin_tail" matches the keyword "twintail"

    def has(*ks: str) -> bool:
        return any(k in n or k in compact for k in ks)

    def sided(base: str) -> SemanticRole:
        return SemanticRole(f"{base}_{side or 'l'}")  # default to left when side is ambiguous

    if has("background", "backdrop") or n == "bg":
        return SemanticRole.background
    # See-through worn-item naming: "*wear" (headwear/eyewear/earwear/neckwear/handwear/legwear/
    # footwear/topwear/bottomwear). The substring "wear" CONTAINS "ear", so these collide with the
    # facial/limb keywords below — classify them FIRST. (Calibrated against real See-through output.)
    if "wear" in n:
        # classify by the token BEFORE "wear" (NOT substring — "wear" itself contains "ear", and
        # "footwear" contains "ear", which is what made everything collide before).
        prefix = n.split("wear")[0].rstrip("_")
        if prefix in ("head", "eye", "ear", "hand", "wrist", "glass", "face"):
            return SemanticRole.accessory  # headwear/eyewear/earwear/handwear = worn accessories
        return SemanticRole.clothing       # top/bottom/leg/foot/neck-wear = garments
    if has("hair", "bang", "fringe", "ahoge", "ponytail", "twintail", "braid", "bun", "sideburn"):
        if has("back", "ponytail", "twintail", "braid", "bun"):
            return SemanticRole.hair_back
        if has("side", "sideburn"):
            return SemanticRole.hair_side
        if has("front", "bang", "fringe", "ahoge"):
            return SemanticRole.hair_front
        return SemanticRole.hair_front
    if has("eyebrow", "brow"):
        return sided("eyebrow")
    if has("pupil", "iris", "irid", "catchlight", "highlight"):  # "irides" = See-through's iris layer
        return sided("pupil")
    if has("sclera") or (has("white") and has("eye")):
        return sided("eye_white")
    if has("eyelash", "lash", "eyeline", "eyelid") or (has("eye") and not has("eyebrow")):
        return sided("eye")
    if has("nose"):
        return SemanticRole.nose
    if has("mouth", "lip", "teeth", "tongue"):
        return SemanticRole.mouth
    if has("ear") and not has("forearm", "earring"):  # avoid forearm/earring collisions
        return sided("ear")
    if has("blush", "cheek"):
        return SemanticRole.blush
    if has("neck"):
        return SemanticRole.neck
    if has("torso", "body", "chest", "breast", "waist", "hip", "pelvis", "belly"):
        return SemanticRole.torso
    if has("arm", "sleeve", "elbow", "forearm"):
        return sided("arm")
    if has("hand", "finger", "fist", "palm"):
        return sided("hand")
    if has("leg", "foot", "feet", "thigh", "knee", "shoe", "boot", "shin", "calf"):
        return sided("leg")
    if has("cloth", "dress", "shirt", "skirt", "outfit", "costume", "collar", "ribbon", "bow",
           "jacket", "coat", "cape", "scarf", "tie", "apron", "vest", "sock", "stocking", "uniform"):
        return SemanticRole.clothing
    if has("accessory", "hat", "cap", "glasses", "horn", "tail", "wing", "acc", "earring",
           "necklace", "crown", "headband", "clip", "halo", "mask", "badge"):
        return SemanticRole.accessory
    if has("face", "skin", "head", "base"):
        return SemanticRole.face_base
    return default


def raws_to_stack(
    raws: list[RawLayer],
    canvas: tuple[int, int],
    work_dir: str | Path,
    *,
    role_map: RoleMapper = role_from_layer_name,
) -> LayerStack:
    """Normalize extracted layers to full-canvas ``{order}_{role}.png`` files and assemble a
    ``LayerStack`` via ``from_layer_dir``. ``raws`` must be ordered bottom -> top (increasing draw
    order). Requires Pillow."""
    from PIL import Image

    width, height = canvas
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    for png in work.glob("*.png"):  # keep the dir clean / idempotent
        png.unlink()

    order = 0
    for raw in raws:
        role = role_map(raw.name)
        full = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        full.paste(raw.image.convert("RGBA"), raw.offset)
        # See-through emits COMBINED (non-L/R) facial parts (one "eyelash"/"eyewhite"/"irides"/
        # "eyebrow" layer covering both eyes). If the layer maps to a default-left facial role and the
        # name carries no side, split its two horizontal blobs into proper _l / _r parts so both-side
        # rigging (blink/eyeball/brow L+R) is authored.
        full = _clean_alpha(full)
        outputs = [(role, full)]
        if role in _SIDE_FLIP and not _has_side_token(raw.name):
            split = _split_lr(full)
            if split is not None:
                left, right = split
                outputs = [(role, left), (_SIDE_FLIP[role], right)]
        elif role in _LIMB_CANDIDATE_ROLES and not _has_side_token(raw.name):
            limb = _limb_split(full)             # both arms / both legs bundled in one layer -> L/R
            if limb is not None:
                lrole, rrole, limg, rimg = limb
                outputs = [(lrole, limg), (rrole, rimg)]
        for out_role, img in outputs:
            img.save(work / f"{order:02d}_{out_role.value}.png")
            order += 1

    return from_layer_dir(work)


# Facial roles that role_from_layer_name defaults to the LEFT side; their right-side counterparts.
_SIDE_FLIP: dict[SemanticRole, SemanticRole] = {
    SemanticRole.eye_l: SemanticRole.eye_r,
    SemanticRole.eye_white_l: SemanticRole.eye_white_r,
    SemanticRole.pupil_l: SemanticRole.pupil_r,
    SemanticRole.eyebrow_l: SemanticRole.eyebrow_r,
}


# Roles whose layer is a plausible LIMB bundle (See-through has no arm/leg vocab, so it packs both arms
# into one layer — often mislabeled accessory — and both legs+socks into a legwear/clothing layer).
_LIMB_CANDIDATE_ROLES = frozenset({
    SemanticRole.accessory, SemanticRole.clothing, SemanticRole.torso, SemanticRole.other,
})


def _limb_split(image, *, threshold: int = 8, min_run: int = 3):
    """If ``image`` is a two-limb bundle (both arms, or both legs), return ``(left_role, right_role,
    left_img, right_img)``; else ``None``.

    See-through emits arms as ONE layer and legs+socks as ONE layer, so limbs can't move independently
    and read as a rigid board. We detect the bundle geometrically (its own labels are unreliable) and
    cut it into left/right full-canvas parts, which the rig then articulates (swing + elbow/knee bend).

    Heuristic (image coords, y-DOWN, normalized): the layer must form **exactly two** horizontally
    separated alpha blobs, each tall enough to be a limb. Legs vs arms is decided by vertical reach —
    legs descend to the feet (bottom of frame); arms sit in the upper body with the blobs out to the
    sides (a torso gap between them). Conservative: anything ambiguous returns ``None`` (kept whole)."""
    import numpy as np

    a = np.array(image.convert("RGBA"))[:, :, 3] >= threshold
    if not a.any():
        return None
    H, W = a.shape
    col = a.any(axis=0)
    runs: list[tuple[int, int]] = []
    start = None
    for x in range(W):
        if col[x] and start is None:
            start = x
        elif not col[x] and start is not None:
            runs.append((start, x - 1))
            start = None
    if start is not None:
        runs.append((start, W - 1))
    runs = [r for r in runs if r[1] - r[0] + 1 >= min_run]
    if len(runs) != 2:
        return None

    l0, l1, r0, r1 = runs[0][0] / W, runs[0][1] / W, runs[1][0] / W, runs[1][1] / W
    ys = np.where(a.any(axis=1))[0]
    y_bottom = ys.max() / H                      # 1.0 == bottom of frame (feet)
    y_top = ys.min() / H
    if (y_bottom - y_top) < 0.15:                # too short vertically to be a limb (e.g. earrings)
        return None
    gap = r0 - l1
    if gap < 0.01:                               # blobs must actually be separated
        return None

    def _cut():
        from PIL import ImageDraw
        mid = int(((runs[0][1] + runs[1][0]) // 2))
        left, right = image.copy(), image.copy()
        ImageDraw.Draw(left).rectangle([mid, 0, W, H], fill=(0, 0, 0, 0))
        ImageDraw.Draw(right).rectangle([0, 0, mid - 1, H], fill=(0, 0, 0, 0))
        return left, right  # left = smaller-x blob = character's RIGHT... resolved by _side below

    # Screen-left blob is the character's own right side in a front-facing portrait; label by role
    # convention (_l = character's left = screen right).
    left_img, right_img = _cut()
    if y_bottom >= 0.85 and l0 >= 0.28 and r1 <= 0.72:        # reaches the feet, central column
        return (SemanticRole.leg_r, SemanticRole.leg_l, left_img, right_img)
    if y_bottom <= 0.82 and l0 < 0.42 and r1 > 0.58:          # upper body, blobs out to the sides
        return (SemanticRole.arm_r, SemanticRole.arm_l, left_img, right_img)
    return None


def _clean_alpha(image, floor: int = 12):
    """Zero out faint-alpha pixels (RGBA) below ``floor``.

    See-through's layers carry faint pixels (alpha ~1-31, with RGB) across their bounding boxes. A
    straight-alpha compositor hides them, but a GPU renderer (nijigenerate) blends them into visible
    rectangular halos — the "blocky face" artifact when many soft facial layers overlap. Hard-clipping
    near-transparent pixels to fully transparent removes the halos without touching the visible art.
    Requires numpy (decompose extra)."""
    import numpy as np

    arr = np.array(image.convert("RGBA"))
    arr[arr[:, :, 3] < floor] = 0
    from PIL import Image as _Image
    return _Image.fromarray(arr, "RGBA")


def _has_side_token(name: str) -> bool:
    return bool(set(_norm(name).split("_")) & {"l", "left", "lt", "r", "right", "rt"})


def _split_lr(image, *, threshold: int = 8, min_run: int = 3):
    """Split a combined (both-sides) facial layer into (left, right) full-canvas RGBA images.

    Detects exactly two horizontally-separated alpha blobs (the two eyes/brows) and cuts at the gap
    between them. Returns ``None`` if the alpha doesn't form exactly two horizontal clusters (e.g. a
    single connected eyeline), in which case the caller keeps the layer whole.

    Column presence is computed at full vertical resolution (alpha over *all* rows, via numpy). An
    earlier version subsampled every ``h//200``-th row "for speed", which fragmented thin features
    like eyelashes into >2 spurious runs on hi-res canvases (a 1280px-tall See-through layer sampled
    every 6th row), so two-eyed lashes never split and the right eye went missing. Noise-width runs
    (< ``min_run`` px) are dropped so a stray alpha column can't break the two-blob test."""
    import numpy as np
    from PIL import ImageDraw

    w, h = image.size
    col_has = (np.array(image.convert("RGBA"))[:, :, 3] >= threshold).any(axis=0)

    runs: list[tuple[int, int]] = []
    start = None
    for x in range(w):
        if col_has[x] and start is None:
            start = x
        elif not col_has[x] and start is not None:
            runs.append((start, x - 1))
            start = None
    if start is not None:
        runs.append((start, w - 1))
    runs = [r for r in runs if r[1] - r[0] + 1 >= min_run]  # ignore single-pixel noise columns
    if len(runs) != 2:
        return None

    gap_mid = (runs[0][1] + runs[1][0]) // 2
    left, right = image.copy(), image.copy()
    ImageDraw.Draw(left).rectangle([gap_mid, 0, w, h], fill=(0, 0, 0, 0))   # keep left blob
    ImageDraw.Draw(right).rectangle([0, 0, gap_mid - 1, h], fill=(0, 0, 0, 0))  # keep right blob
    return left, right


def _raw_layers_from_psd(psd_path: str | Path) -> tuple[list[RawLayer], tuple[int, int]]:
    """Extract leaf layers from a PSD (bottom -> top) using psd-tools."""
    from psd_tools import PSDImage  # optional dep (decompose extra)

    psd = PSDImage.open(psd_path)
    raws: list[RawLayer] = []
    for layer in psd.descendants():
        if layer.is_group() or not layer.has_pixels():
            continue
        pil = layer.topil()
        if pil is None:
            continue
        left, top = layer.offset
        raws.append(RawLayer(name=layer.name or "layer", image=pil, offset=(int(left), int(top))))
    return raws, (psd.width, psd.height)


def from_psd(
    psd_path: str | Path,
    work_dir: str | Path,
    *,
    role_map: RoleMapper = role_from_layer_name,
) -> LayerStack:
    """Read a layered PSD (e.g. See-through output) into a ``LayerStack``.

    Layers are mapped to semantic roles, flattened onto full-canvas PNGs in ``work_dir`` (named
    ``{order}_{role}.png``), and assembled via ``from_layer_dir``. Requires the ``decompose`` extra
    (psd-tools + Pillow). Pass ``role_map`` to override the name->role heuristics.
    """
    raws, canvas = _raw_layers_from_psd(psd_path)
    if not raws:
        raise ValueError(f"{psd_path}: no pixel layers found in PSD")
    return raws_to_stack(raws, canvas, work_dir, role_map=role_map)


# --------------------------------------------------------------------------------------------------
# Remote decompose service (See-through on a GPU box — see docs/DECOMPOSE_SERVICE.md)
# --------------------------------------------------------------------------------------------------
def from_service(
    image_path: str | Path,
    service_url: str,
    work_dir: str | Path,
    *,
    role_map: RoleMapper = role_from_layer_name,
    token: str | None = None,
    timeout: float = 1800.0,
    poll_interval: float = 5.0,
) -> LayerStack:
    """Decompose a single flat image via a remote See-through service, into a ``LayerStack``.

    Uses an **async job** protocol so no single HTTP connection is held across the minutes-long
    inference (a NAT/idle timeout would drop it): POST the image to ``/decompose`` -> get a ``job_id``;
    poll ``/jobs/{id}`` until done; GET ``/jobs/{id}/result`` for the layered ``.psd``; then run
    ``from_psd``. This is the Tier-2 entry point — GPU work runs remotely, the rig builds locally.
    ``token`` sets ``X-Auth-Token`` if required. Requires the ``decompose`` extra (psd-tools + Pillow).
    """
    import json
    import time
    import urllib.error
    import urllib.request

    image_path = Path(image_path)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    if not image_path.is_file():
        raise FileNotFoundError(f"image not found: {image_path}")
    base = service_url.rstrip("/")

    def _get(path: str, *, data: bytes | None = None, want_json: bool, t: float):
        headers = {"X-Auth-Token": token} if token else {}
        if data is not None:
            headers["Content-Type"] = "application/octet-stream"
        req = urllib.request.Request(base + path, data=data,
                                     method="POST" if data is not None else "GET", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=t) as resp:  # noqa: S310 (operator URL)
                body = resp.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"decompose service {exc.code}: "
                               f"{exc.read().decode(errors='replace')[:800]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"decompose service unreachable at {base}: {exc.reason}") from exc
        except OSError as exc:  # e.g. ConnectionResetError/BrokenPipe mid-upload through the tunnel
            raise RuntimeError(f"decompose service connection error: {exc}") from exc
        return json.loads(body) if want_json else body

    img_bytes = image_path.read_bytes()

    def _retryable(exc: Exception) -> bool:
        # transient failures worth re-submitting the whole job for: the service restarted and dropped
        # the in-memory job map (404 "unknown job id"), or the tunnel/connection blipped mid-request
        # (reset by peer, broken pipe, unreachable) — common while a fresh GPU + SSH tunnel settle.
        s = str(exc).lower()
        return any(k in s for k in ("unknown job id", "decompose service 404", "connection error",
                                    "reset by peer", "broken pipe", "unreachable"))

    def _run_once() -> bytes:
        # submit (short request) -> poll (each poll short, survives long inference behind NAT) -> result
        job_id = _get("/decompose", data=img_bytes, want_json=True, t=120.0)["job_id"]
        waited = 0.0
        while waited < timeout:
            time.sleep(poll_interval)
            waited += poll_interval
            status = _get(f"/jobs/{job_id}", want_json=True, t=30.0)
            if status["status"] == "done":
                break
            if status["status"] == "error":
                raise RuntimeError(f"decompose failed on the service: {status.get('error')}")
        else:
            raise RuntimeError(f"decompose timed out after {timeout:.0f}s (job {job_id})")
        return _get(f"/jobs/{job_id}/result", want_json=False, t=180.0)

    # Retry the whole job if the service dropped it (a fresh service can restart once and lose the job
    # map -> 404 "unknown job id"); a re-submit then succeeds.
    attempts = 3
    for attempt in range(attempts):
        try:
            psd_bytes = _run_once()
            break
        except RuntimeError as exc:
            if _retryable(exc) and attempt < attempts - 1:
                time.sleep(10.0)  # let the service/tunnel settle, then re-submit
                continue
            raise

    psd_path = work / f"{image_path.stem}.psd"
    psd_path.write_bytes(psd_bytes)
    return from_psd(psd_path, work, role_map=role_map)
