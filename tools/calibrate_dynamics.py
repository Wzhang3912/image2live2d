#!/usr/bin/env python3
"""CLI: calibrate the dynamics score against a LOCAL corpus of pro Live2D models.

The scoring/metrics logic lives in ``image2live2d.core.structure.calibrate`` (pure, unit-tested); this
is just the shell that reads a corpus off disk and prints a report. The corpus is your own collection
of professional models and is **gitignored** — real ``.moc3`` / ``.physics3.json`` assets are
copyrighted and must never be committed (this repo is code-only).

Corpus manifest (JSON), paths relative to the manifest's directory::

    {
      "models": [
        {
          "name": "some_character",
          "physics3": "some_character.physics3.json",
          "layers": [
            {"id": "hair_front", "role": "hair_front", "texture": "hair_front.png",
             "params": ["ParamHairFront"]},
            {"id": "collar", "role": "clothing", "texture": "collar.png", "params": ["ParamCollar"]}
          ]
        }
      ]
    }

Each layer's PNG must span the whole canvas (same assumption as the pipeline). A layer counts as
ground-truth "needs physics" when any of its ``params`` appears among the model's ``.physics3.json``
output params. Run::

    python tools/calibrate_dynamics.py --corpus path/to/corpus/manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the package importable when run straight from a checkout (no install needed).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from image2live2d.core.structure.calibrate import (  # noqa: E402
    best_thresholds,
    evaluate,
    physics3_output_params,
)
from image2live2d.core.structure.dynamics import analyze_stack  # noqa: E402
from image2live2d.core.types import Layer, LayerStack  # noqa: E402
from image2live2d.irr.schema import SemanticRole  # noqa: E402


def _load_labeled(manifest_path: Path, samples: int | None):
    """Yield ``(PartDynamics, pro_has_physics)`` for every meshable layer across the corpus."""
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    labeled = []
    for model in manifest.get("models", []):
        truth_params = physics3_output_params(json.loads((root / model["physics3"]).read_text()))
        layers, want = [], {}
        for i, ly in enumerate(model["layers"]):
            layers.append(Layer(
                id=ly["id"], semantic_role=SemanticRole(ly["role"]),
                texture_path=root / ly["texture"], draw_order=ly.get("draw_order", i * 10),
                width=ly.get("width", 0), height=ly.get("height", 0),
            ))
            want[ly["id"]] = any(p in truth_params for p in ly.get("params", []))
        stack = LayerStack(layers=layers, canvas_width=1, canvas_height=1)
        kw = {"samples": samples} if samples else {}
        for d in analyze_stack(stack, **kw):
            labeled.append((d, want.get(d.part_id, False)))
    return labeled


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Calibrate the dynamics score against a pro-model corpus.")
    ap.add_argument("--corpus", required=True, type=Path, help="path to the corpus manifest.json")
    ap.add_argument("--samples", type=int, default=None, help="probe grid size (default: module default)")
    ap.add_argument("--verbose", action="store_true", help="list each part's verdict vs the label")
    args = ap.parse_args(argv)

    labeled = _load_labeled(args.corpus, args.samples)
    if not labeled:
        print("no labeled parts found in corpus", file=sys.stderr)
        return 1

    if args.verbose:
        from image2live2d.core.structure.calibrate import predicted_physics
        print("part                     score  free  verdict   pro   result")
        for d, truth in sorted(labeled, key=lambda x: (x[1], x[0].part_id)):
            pred = predicted_physics(d)
            tag = "ok" if pred == truth else ("MISS(fn)" if truth else "OVER(fp)")
            print(f"  {d.part_id:22} {d.score:.2f}  {d.free_edge_ratio:.2f}  "
                  f"{d.verdict.value:8} {'phys' if truth else 'rigid':5} {tag}")
        print()

    at_default = evaluate(labeled)
    best = best_thresholds(labeled)
    n_phys = sum(1 for _, t in labeled if t)
    print(f"corpus: {len(labeled)} parts, {n_phys} with pro physics\n")
    print("current defaults:")
    print(f"  precision={at_default.precision:.3f} recall={at_default.recall:.3f} "
          f"f1={at_default.f1:.3f} acc={at_default.accuracy:.3f} "
          f"(tp={at_default.tp} fp={at_default.fp} fn={at_default.fn} tn={at_default.tn})")
    m = best.metrics
    print("\nbest thresholds on this corpus:")
    print(f"  gentle_t={best.gentle_t:.2f} free_edge_floor={best.free_edge_floor:.2f}")
    print(f"  precision={m.precision:.3f} recall={m.recall:.3f} f1={m.f1:.3f} acc={m.accuracy:.3f} "
          f"(tp={m.tp} fp={m.fp} fn={m.fn} tn={m.tn})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
