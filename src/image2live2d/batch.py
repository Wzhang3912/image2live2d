"""Batch conversion — a folder of PSDs / layer dirs -> one ``.inp`` each + an aggregate QA report.

The product workflow for converting a whole roster at once. Each input is converted independently; one
bad input is reported as an error and doesn't abort the run. The aggregate QA reuses
``core.qa.batch`` so the pass-rate gate is identical to the single-file path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .api import ConversionResult, convert_layers, convert_psd
from .core import decompose
from .core.qa import BatchReport, batch as qa_batch


@dataclass
class BatchItem:
    name: str
    input_path: Path
    result: ConversionResult | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.result is not None


@dataclass
class BatchOutcome:
    items: list[BatchItem] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def converted(self) -> int:
        return sum(1 for i in self.items if i.ok)

    @property
    def errors(self) -> list[BatchItem]:
        return [i for i in self.items if not i.ok]

    def qa(self) -> BatchReport:
        return qa_batch([(i.name, i.result.rig) for i in self.items if i.ok])

    def format(self) -> str:
        lines = []
        for i in self.items:
            if not i.ok:
                lines.append(f"ERROR   {i.name[:24]:24}  {i.error}")
        lines.append(self.qa().format() if self.converted else "no inputs converted")
        if self.errors:
            lines.append(f"errors: {len(self.errors)}/{self.total}")
        return "\n".join(lines)


def _is_layer_dir(p: Path) -> bool:
    """A directory holding at least one ``{order}_{role}.png`` file."""
    if not p.is_dir():
        return False
    for png in p.glob("*.png"):
        try:
            decompose.parse_layer_name(png.stem)
            return True
        except ValueError:
            continue
    return False


def discover_inputs(root: str | Path) -> list[Path]:
    """Find convertible inputs under ``root``: ``.psd`` files and layer directories.

    If ``root`` is itself a PSD or a layer dir, returns just ``[root]``. Otherwise scans one level for
    ``*.psd`` files and immediate subdirectories that look like layer dirs.
    """
    root = Path(root)
    if root.is_file() and root.suffix.lower() == ".psd":
        return [root]
    if _is_layer_dir(root):
        return [root]
    if not root.is_dir():
        return []
    found = sorted(root.glob("*.psd"))
    found += sorted(d for d in root.iterdir() if _is_layer_dir(d))
    return found


def convert_batch(
    inputs, out_dir: str | Path, *, live2d: bool = False
) -> BatchOutcome:
    """Convert each input (a PSD or layer dir) into ``out_dir``; collect results + errors.

    ``inputs`` is an iterable of paths (e.g. from ``discover_inputs``)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    outcome = BatchOutcome()
    for raw in inputs:
        path = Path(raw)
        name = path.stem if path.suffix.lower() == ".psd" else path.name
        try:
            if path.suffix.lower() == ".psd":
                result = convert_psd(path, out, name=name, live2d=live2d)
            else:
                result = convert_layers(path, out, name=name, live2d=live2d)
            outcome.items.append(BatchItem(name=name, input_path=path, result=result))
        except (FileNotFoundError, ValueError, ImportError) as exc:
            outcome.items.append(BatchItem(name=name, input_path=path, error=str(exc)))
    return outcome
