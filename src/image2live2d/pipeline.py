"""End-to-end orchestration: image -> Rig (IRR) -> emitted model.

This wires the shared stages together and hands the resulting ``Rig`` to a chosen ``Emitter``.
Stages currently raise ``NotImplementedError``; this module fixes the *contract* and call order so
implementation can land stage-by-stage behind the Phase gates in ``docs/MVP_ARCHITECTURE.md``.
"""

from __future__ import annotations

from pathlib import Path

from .backends.base import Emitter
from .backends.nijilive import NijiliveEmitter
from .core import decompose, ingest, landmark, mesh, motion, physics, preprocess
from .core.assemble import assemble_rig
from .core.landmark import Landmarks
from .core.rig import author_rig, select_template
from .core.types import LayerStack
from .irr.schema import Rig


def rig_from_stack(stack: LayerStack, *, name: str, source: str | None = None) -> Rig:
    """Build a validated ``Rig`` from an already-decomposed ``LayerStack`` (stages 3-6).

    This is the headless entry point used once decomposition exists (real or synthetic), letting the
    whole spine run without standing up See-through.
    """
    meshes = mesh.build_meshes(stack)
    _lift_occluded_accessories(stack, meshes)
    template = select_template(stack)
    landmarks = _safe_landmarks(stack)
    authoring = author_rig(stack, meshes, template, landmarks=landmarks)
    phys = _safe_physics(stack, authoring.parameters, meshes)
    anims = motion.generate_idle(authoring.parameters)
    return assemble_rig(
        name=name,
        source=source,
        stack=stack,
        meshes=meshes,
        deformers=authoring.deformers,
        parameters=authoring.parameters,
        physics=phys,
        archetype=template.name,
        animations=anims,
    )


def build_rig(source: str | Path, *, name: str) -> Rig:
    """Run the full shared pipeline from a source image (stages 0-6) and return a ``Rig``."""
    image = ingest.load_image(source)
    prepared = preprocess.prepare(image)
    stack = decompose.decompose(prepared)
    return rig_from_stack(stack, name=name, source=str(source))


def _lift_occluded_accessories(stack: LayerStack, meshes) -> None:
    """Raise head ornaments (e.g. a hair clip/flower) above the front hair.

    See-through's depth model sometimes orders a small head ``accessory`` *behind* ``hair_front`` (the
    flower ends up hidden by the bangs). When an accessory overlaps the front-hair region but is drawn
    under it, lift its draw_order just above the hair so it shows — the way it does in the source art.
    Accessories that don't overlap the hair (sleeves, wrist cuffs) are untouched. Mutates ``stack``.
    """
    from .irr.schema import SemanticRole

    mbp = {m.part_id: m for m in meshes}

    def bbox(pid):
        m = mbp.get(pid)
        if not m:
            return None
        xs = [x for x, _ in m.vertices]
        ys = [y for _, y in m.vertices]
        return (min(xs), min(ys), max(xs), max(ys))

    hair_roles = (SemanticRole.hair_front, SemanticRole.hair_side, SemanticRole.hair_back)
    hair = [(L, b) for L in stack.layers if L.semantic_role in hair_roles and (b := bbox(L.id))]
    if not hair:
        return

    def overlap_frac(a, b) -> float:
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        area = max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)
        return ix * iy / area

    nxt = max(L.draw_order for L, _ in hair)
    for L in stack.layers:
        if L.semantic_role is not SemanticRole.accessory:
            continue
        ab = bbox(L.id)
        if ab is None:
            continue
        # the hair parts this accessory overlaps but is currently drawn under
        occluding = [hL.draw_order for hL, hb in hair
                     if overlap_frac(ab, hb) > 0.3 and hL.draw_order > L.draw_order]
        if occluding:
            nxt += 1
            L.draw_order = nxt


def _safe_physics(stack: LayerStack, parameters, meshes=None):
    """Tolerate a not-yet-implemented physics stage so the spine still runs end-to-end."""
    try:
        return physics.generate_physics(stack, parameters, meshes=meshes)
    except NotImplementedError:
        return []


def _safe_landmarks(stack: LayerStack) -> Landmarks | None:
    """Extract silhouette landmarks if possible; tolerate a missing Pillow / gated ML so the spine
    still runs (the solver falls back to bbox heuristics when landmarks are None)."""
    try:
        return landmark.extract_landmarks(stack)
    except (ImportError, NotImplementedError):
        return None


def convert(
    source: str | Path,
    out_dir: str | Path,
    *,
    name: str = "model",
    emitter: Emitter | None = None,
) -> Path:
    """Full conversion: image -> emitted model file. Defaults to the nijilive (Route B) emitter."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rig = build_rig(source, name=name)
    return (emitter or NijiliveEmitter()).emit(rig, out)
