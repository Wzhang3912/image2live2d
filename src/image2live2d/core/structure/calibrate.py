"""P1b — calibrate the dynamics score against a corpus of pro Live2D models.

The dynamics detector (:mod:`.dynamics`) decides *which parts need physics* from geometry, but its
thresholds (``_DYNAMIC_T`` / ``_GENTLE_T`` / ``_FREE_EDGE_FLOOR``) encode a rigger's *taste* — a
judgment call. This module measures that judgment against ground truth. A professional
``.physics3.json`` states exactly which parameters the artist gave physics; point our scorer at the
same character, treat "verdict != rigid" as our prediction of "needs physics", and compare.

The interesting output is the **threshold sweep**: vary ``gentle_t`` / ``free_edge_floor`` and find the
values that best agree with the corpus, so the defaults in :mod:`.dynamics` can be tuned to real pro
work rather than guessed. Everything here is pure (no IO, no Pillow, no ML) so it is unit-testable with
synthetic parts; ``tools/calibrate_dynamics.py`` is the CLI that feeds it a local (gitignored) corpus
of real models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .dynamics import _FREE_EDGE_FLOOR, _GENTLE_T, PartDynamics

# Default sweep grids. gentle_t is the primary knob for the binary "needs physics?" decision (dynamic vs
# gentle both count as physics); the free-edge floor is the safety net that rescues an obvious hanging
# edge whose score fell short. Kept coarse so a sweep over a modest corpus stays fast and legible.
_GENTLE_GRID: tuple[float, ...] = (0.20, 0.25, 0.30, 0.33, 0.36, 0.40, 0.45, 0.50)
_FLOOR_GRID: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 1.01)   # 1.01 = floor effectively disabled


@dataclass(frozen=True)
class Metrics:
    """Binary-classification metrics for "part needs physics" (predicted vs pro ground truth)."""

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else 0.0


@dataclass(frozen=True)
class SweepPoint:
    """One threshold combination and the agreement it produced on the corpus."""

    gentle_t: float
    free_edge_floor: float
    metrics: Metrics


def physics3_output_params(doc: dict) -> set[str]:
    """The set of *output* (physics-driven) parameter ids in a parsed Live2D ``.physics3.json`` — i.e.
    exactly the parameters the artist attached physics to. This is the corpus ground truth: a part is
    "needs physics" if one of the params that deforms it appears here. Malformed entries are skipped."""
    out: set[str] = set()
    for setting in doc.get("PhysicsSettings", []) or []:
        for o in setting.get("Output", []) or []:
            pid = (o.get("Destination") or {}).get("Id")
            if isinstance(pid, str) and pid:
                out.add(pid)
    return out


def predicted_physics(
    d: PartDynamics, *, gentle_t: float = _GENTLE_T, free_edge_floor: float = _FREE_EDGE_FLOOR,
) -> bool:
    """Our binary prediction "this part needs physics" at the given thresholds — the exact rule
    :func:`dynamics._verdict` uses for *non-rigid*, recomputed from the stored signals so a sweep can
    move the thresholds without re-scoring: eligible AND (score over gentle_t OR an obvious free edge)."""
    return d.sway_eligible and (d.score >= gentle_t or d.free_edge_ratio >= free_edge_floor)


def evaluate(
    labeled: Iterable[tuple[PartDynamics, bool]],
    *,
    gentle_t: float = _GENTLE_T,
    free_edge_floor: float = _FREE_EDGE_FLOOR,
) -> Metrics:
    """Confusion metrics over ``(part_dynamics, pro_has_physics)`` pairs at the given thresholds."""
    tp = fp = fn = tn = 0
    for d, truth in labeled:
        pred = predicted_physics(d, gentle_t=gentle_t, free_edge_floor=free_edge_floor)
        if pred and truth:
            tp += 1
        elif pred and not truth:
            fp += 1
        elif not pred and truth:
            fn += 1
        else:
            tn += 1
    return Metrics(tp, fp, fn, tn)


def sweep(
    labeled: Iterable[tuple[PartDynamics, bool]],
    *,
    gentle_grid: Iterable[float] = _GENTLE_GRID,
    floor_grid: Iterable[float] = _FLOOR_GRID,
) -> list[SweepPoint]:
    """Every ``(gentle_t, free_edge_floor)`` combination and its metrics, best agreement first.

    Ranked by F1, breaking ties toward the *stricter* (higher) gentle threshold — the dynamics score is
    biased toward restraint, so when two thresholds fit equally we prefer the one that rigs less."""
    pairs = list(labeled)
    points = [
        SweepPoint(g, f, evaluate(pairs, gentle_t=g, free_edge_floor=f))
        for g in gentle_grid
        for f in floor_grid
    ]
    points.sort(key=lambda p: (p.metrics.f1, p.gentle_t, p.free_edge_floor), reverse=True)
    return points


def best_thresholds(
    labeled: Iterable[tuple[PartDynamics, bool]],
    *,
    gentle_grid: Iterable[float] = _GENTLE_GRID,
    floor_grid: Iterable[float] = _FLOOR_GRID,
) -> SweepPoint:
    """The single best-agreeing threshold combination on the corpus (see :func:`sweep`)."""
    return sweep(labeled, gentle_grid=gentle_grid, floor_grid=floor_grid)[0]
