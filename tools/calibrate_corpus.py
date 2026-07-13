#!/usr/bin/env python3
"""Corpus-wide P1b calibration — run the dynamics-score calibrator over *every* pro model under a root
and pool the agreement, instead of eyeballing one model at a time (``calibrate_moc3.py``).

    python tools/calibrate_corpus.py "/path/to/VTube Studio/.../StreamingAssets"
    python tools/calibrate_corpus.py --root "$VTS_STREAMING_ASSETS"   # same, from an env var

It walks the root for every ``*.physics3.json`` with a sibling ``*.moc3``, runs :func:`build_corpus`
on each, and prints a per-model table + the **pooled** precision/recall/F1 (confusion counts are
additive across models — see :func:`calibrate.pool_metrics`). Models the name-based labeler can't read
are listed with a reason, not silently dropped, so the corpus's *reach* is honest.

THE MEASURED CEILING (VTube Studio library, 2026-07): of the local models, exactly **one** — Akari
(``akari.moc3``, a semantically-named pro rig) — is richly labelable and calibratable (P≈0.79 R≈0.92
F1≈0.85). The rest are excluded for concrete, non-fixable-by-tweaking reasons:

  * **Opaque drawable ids** (``ArtMesh54`` …): the official Cubism samples (Hiyori) and many item
    models name every mesh ``ArtMesh<n>``. Our ground-truth labeler and part-grouping are *name-based*
    (a strand's ``hair_left2..8`` segments merge into ``hair_left``), so opaque ids collapse the whole
    model into one blob and no role can be inferred. Perturbation labeling (``drawables_moved_by``) is
    naming-independent and reads Hiyori's physics cleanly, BUT scoring must then drop to *per-drawable*
    granularity — and there the geometry score ranks rigged vs unrigged at only AUC≈0.46 (Akari, where
    the ``ShirtTop`` physics param cascades to 73% of drawables) to ≈0.62 (Hiyori). The score is
    discriminative at the **part** level (a whole strand), not the sub-mesh drawable level, so
    per-drawable breadth is a dead end.
  * **No decodable physics targets**: some item rigs ship a ``.physics3.json`` whose output-param names
    don't prefix-match any drawable name, so zero parts get a positive label (nothing to calibrate on).
  * **No physics at all**: the VTS mascots (wanko/hijiki/tororo) ship no ``.physics3.json``.

Conclusion: breadth is bottlenecked by **labelability** (semantic naming + part-vs-drawable
granularity), not model count. This runner is the honest instrument — it will pool a real N>1 corpus
the moment more semantically-named pro rigs are available locally; today it reports N=1 with the rest
excluded for stated reasons. Models are read locally and never committed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibrate_moc3 import build_corpus  # noqa: E402
from image2live2d.core.structure.calibrate import (  # noqa: E402
    best_thresholds,
    evaluate,
    pool_metrics,
    roc_auc,
)


def discover(root: Path) -> list[tuple[Path, Path]]:
    """Every (``.moc3``, ``.physics3.json``) pair under ``root`` — a physics file with a sibling moc3,
    sorted for deterministic output."""
    pairs: list[tuple[Path, Path]] = []
    for phys in sorted(root.rglob("*.physics3.json")):
        mocs = sorted(phys.parent.glob("*.moc3"))
        if mocs:
            pairs.append((mocs[0], phys))
    return pairs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pool the dynamics calibration over a corpus of pro models.")
    ap.add_argument("root", nargs="?", type=Path, help="directory to walk for *.moc3 + *.physics3.json")
    ap.add_argument("--root", dest="root_opt", type=Path, default=None,
                    help="same as the positional root (or set VTS_STREAMING_ASSETS)")
    ap.add_argument("--core", default=None, help="path to Live2DCubismCore (else auto-found)")
    ap.add_argument("--align", action="store_true", help="collapse depth-layered parts before scoring")
    args = ap.parse_args(argv)

    root = args.root or args.root_opt or (
        Path(os.environ["VTS_STREAMING_ASSETS"]) if "VTS_STREAMING_ASSETS" in os.environ else None)
    if root is None:
        print("give a corpus root (positional, --root, or $VTS_STREAMING_ASSETS)", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    pairs = discover(root)
    if not pairs:
        print(f"no .moc3 + .physics3.json pairs under {root}", file=sys.stderr)
        return 1

    print(f"{'model':40} {'parts':>5} {'phys':>4} {'prec':>5} {'rec':>5} {'f1':>5} {'auc':>5}  note")
    usable: list = []       # per-model Metrics that entered the pool
    all_pairs: list = []     # every (dynamics, truth) for the pooled best-threshold sweep
    for moc3, physics3 in pairs:
        name = moc3.parent.name[:38]
        try:
            labeled, _ = build_corpus(moc3, physics3, args.core, align=args.align)
        except Exception as e:                       # a model the core/labeler can't read at all
            print(f"{name:40} {'—':>5} {'—':>4} {'—':>5} {'—':>5} {'—':>5} {'—':>5}  ERR {type(e).__name__}")
            continue
        if not labeled:
            print(f"{name:40} {0:5} {'—':>4} {'—':>5} {'—':>5} {'—':>5} {'—':>5}  skip: opaque/unnamed ids")
            continue
        n_phys = sum(1 for _, t in labeled if t)
        if n_phys == 0:
            print(f"{name:40} {len(labeled):5} {0:4} {'—':>5} {'—':>5} {'—':>5} {'—':>5}  "
                  f"skip: no decodable physics targets")
            continue
        m = evaluate(labeled)
        auc = roc_auc(labeled)
        astr = f"{auc:.3f}" if auc is not None else "  —"
        print(f"{name:40} {len(labeled):5} {n_phys:4} {m.precision:5.3f} {m.recall:5.3f} "
              f"{m.f1:5.3f} {astr:>5}  ok")
        usable.append(m)
        all_pairs.extend(labeled)

    print()
    if usable:
        pooled = pool_metrics(usable)
        print(f"POOLED over {len(usable)} model(s): precision={pooled.precision:.3f} "
              f"recall={pooled.recall:.3f} f1={pooled.f1:.3f} acc={pooled.accuracy:.3f} "
              f"(tp={pooled.tp} fp={pooled.fp} fn={pooled.fn} tn={pooled.tn})")
        best = best_thresholds(all_pairs)
        b = best.metrics
        print(f"best pooled thresholds: gentle_t={best.gentle_t:.2f} "
              f"free_edge_floor={best.free_edge_floor:.2f} -> f1={b.f1:.3f}")
    else:
        print("no labelable models in this corpus (see per-model notes above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
