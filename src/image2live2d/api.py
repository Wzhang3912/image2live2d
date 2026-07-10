"""Public API — the stable, documented entry points for converting layered art to a model.

These wrap the pipeline (decompose seam → mesh → rig → physics → motion → IRR → emit) behind a few
small functions so callers never touch internals. Two inputs are supported today (Tier 1, clean
license): a **labeled layer directory** (``{order}_{role}.png``) and a **PSD** (e.g. See-through
output). Both produce a nijilive ``.inp`` (Route B) and/or a Live2D bundle (Route A, JSON-only until a
``.moc3`` template is supplied).

Single flat-image input (Tier 2) is gated on a clean-license decomposer — see ``docs/PHASE5_PLAN.md``.

Example::

    from image2live2d import convert_psd
    result = convert_psd("hero.psd", "out/")
    print(result.inp_path, result.qa.passed)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .backends.nijilive import NijiliveEmitter
from .core import decompose
from .core.qa import RigReport, evaluate
from .core.types import LayerStack
from .irr.schema import Rig
from .pipeline import rig_from_stack


@dataclass
class ConversionResult:
    """What a conversion produced: the rig, where the model was written, and its QA report."""

    name: str
    rig: Rig
    inp_path: Path | None
    live2d_path: Path | None
    qa: RigReport

    @property
    def passed(self) -> bool:
        return self.qa.passed


def rig_from_layer_dir(layer_dir: str | Path, *, name: str | None = None) -> Rig:
    """Build a validated ``Rig`` from a directory of ``{order}_{role}.png`` layers."""
    layer_dir = Path(layer_dir)
    return rig_from_stack(decompose.from_layer_dir(layer_dir),
                          name=name or layer_dir.name, source=str(layer_dir))


def rig_from_psd(psd_path: str | Path, work_dir: str | Path, *, name: str | None = None) -> Rig:
    """Build a validated ``Rig`` from a layered PSD (extracts layer PNGs into ``work_dir``)."""
    psd_path = Path(psd_path)
    stack = decompose.from_psd(psd_path, work_dir)
    return rig_from_stack(stack, name=name or psd_path.stem, source=str(psd_path))


def convert_layers(
    layer_dir: str | Path,
    out_dir: str | Path,
    *,
    name: str | None = None,
    live2d: bool = False,
) -> ConversionResult:
    """Convert a labeled layer directory to a nijilive ``.inp`` (and optionally a Live2D bundle)."""
    layer_dir = Path(layer_dir)
    rig = rig_from_layer_dir(layer_dir, name=name)
    return _emit(rig, out_dir, asset_root=layer_dir, live2d=live2d)


def convert_psd(
    psd_path: str | Path,
    out_dir: str | Path,
    *,
    name: str | None = None,
    work_dir: str | Path | None = None,
    live2d: bool = False,
) -> ConversionResult:
    """Convert a layered PSD to a nijilive ``.inp`` (and optionally a Live2D bundle)."""
    psd_path = Path(psd_path)
    out = Path(out_dir)
    work = Path(work_dir) if work_dir else out / f"{(name or psd_path.stem)}_layers"
    rig = rig_from_psd(psd_path, work, name=name)
    return _emit(rig, out, asset_root=work, live2d=live2d)


def convert_stack(
    stack: LayerStack,
    out_dir: str | Path,
    *,
    name: str,
    asset_root: str | Path,
    live2d: bool = False,
) -> ConversionResult:
    """Convert an already-built ``LayerStack`` (for callers that ran their own decomposition)."""
    return _emit(rig_from_stack(stack, name=name), out_dir, asset_root=Path(asset_root), live2d=live2d)


def _emit(rig: Rig, out_dir: str | Path, *, asset_root: Path, live2d: bool) -> ConversionResult:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    inp_path = NijiliveEmitter(asset_root=asset_root).emit(rig, out)

    live2d_path: Path | None = None
    if live2d:
        from .backends.live2d import Live2DEmitter  # local import: keep API import light

        live2d_path = Live2DEmitter(asset_root=asset_root).build(rig, out / f"{rig.meta.name}_live2d").model3_path

    return ConversionResult(
        name=rig.meta.name,
        rig=rig,
        inp_path=inp_path,
        live2d_path=live2d_path,
        qa=evaluate(rig, rig.meta.name),
    )
