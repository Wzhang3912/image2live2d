"""P1b — dynamics-score calibration harness. Proves the pure scoring-vs-ground-truth logic (physics3
parsing, confusion metrics, threshold sweep) without any real (copyrighted) models — synthetic parts
with known scores stand in for a corpus.
"""

from __future__ import annotations

from image2live2d.core.structure import (
    Metrics,
    best_thresholds,
    evaluate,
    physics3_output_params,
    pool_metrics,
    predicted_physics,
    roc_auc,
    sweep,
)
from image2live2d.core.structure.dynamics import (
    DynamicsVerdict,
    PartDynamics,
    PhysicalClass,
)
from image2live2d.irr.schema import SemanticRole as R


def _dyn(pid, score, free_edge=0.0, eligible=True):
    """A PartDynamics carrying just the fields the calibration rule reads (score/free_edge/eligible)."""
    return PartDynamics(
        part_id=pid, role=R.clothing, free_edge_ratio=free_edge, cantilever=0.0, slenderness=1.0,
        principal_angle=0.0, material_prior=0.0, anchor=(0.0, 0.0), coverage=0.5,
        sway_eligible=eligible, score=score,
        verdict=DynamicsVerdict.rigid, physical_class=PhysicalClass.rigid,
    )


def test_physics3_output_params_extracts_driven_ids():
    doc = {"PhysicsSettings": [
        {"Output": [{"Destination": {"Id": "ParamHairFront"}}, {"Destination": {"Id": "ParamHairBack"}}]},
        {"Output": [{"Destination": {"Id": "ParamSkirtC"}}, {"Destination": {}}, {"bogus": 1}]},
    ]}
    assert physics3_output_params(doc) == {"ParamHairFront", "ParamHairBack", "ParamSkirtC"}
    assert physics3_output_params({}) == set()          # tolerates an empty / malformed doc


def test_predicted_physics_uses_score_floor_and_eligibility():
    assert predicted_physics(_dyn("a", 0.7))                       # score over gentle_t -> physics
    assert not predicted_physics(_dyn("b", 0.1))                   # low score -> no physics
    assert predicted_physics(_dyn("c", 0.1, free_edge=0.9))        # free-edge floor rescues it
    assert not predicted_physics(_dyn("d", 0.9, eligible=False))   # skin/limb never sways


def test_evaluate_confusion_and_scores():
    labeled = [
        (_dyn("tp", 0.7), True),    # predicted physics, truly has it
        (_dyn("fp", 0.7), False),   # predicted physics, artist didn't
        (_dyn("fn", 0.1), True),    # missed a part that has physics
        (_dyn("tn", 0.1), False),   # correctly left rigid
    ]
    m = evaluate(labeled)
    assert (m.tp, m.fp, m.fn, m.tn) == (1, 1, 1, 1)
    assert m.precision == 0.5 and m.recall == 0.5 and m.f1 == 0.5 and m.accuracy == 0.5


def test_sweep_finds_a_perfectly_separating_threshold():
    # physics parts score in [0.40, 0.50]; rigid parts in [0.10, 0.20] -> a gentle_t ~0.3 separates them.
    labeled = [(_dyn(f"p{i}", s), True) for i, s in enumerate((0.40, 0.45, 0.50))]
    labeled += [(_dyn(f"n{i}", s), False) for i, s in enumerate((0.10, 0.15, 0.20))]
    best = best_thresholds(labeled)
    assert best.metrics.f1 == 1.0                        # a threshold separates the corpus perfectly
    assert 0.20 < best.gentle_t <= 0.40
    # the sweep is ranked best-first and never proposes something worse than the perfect split
    ranked = sweep(labeled)
    assert ranked[0].metrics.f1 == 1.0
    assert all(ranked[0].metrics.f1 >= p.metrics.f1 for p in ranked)


def test_sweep_tie_breaks_toward_restraint():
    # one physics part at a high score, one rigid at a low score: many thresholds score f1=1.0; the top
    # pick should be the strictest (highest gentle_t) among them, since the detector favors restraint.
    labeled = [(_dyn("hi", 0.80), True), (_dyn("lo", 0.10), False)]
    ranked = [p for p in sweep(labeled) if p.metrics.f1 == 1.0]
    assert ranked[0].gentle_t == max(p.gentle_t for p in ranked)


def test_pool_metrics_sums_confusion_across_models():
    # each model's confusion counts are independent decisions -> pooling is a straight sum.
    a = Metrics(tp=3, fp=1, fn=1, tn=5)     # precision 0.75, recall 0.75
    b = Metrics(tp=1, fp=1, fn=3, tn=5)     # precision 0.50, recall 0.25
    pooled = pool_metrics([a, b])
    assert (pooled.tp, pooled.fp, pooled.fn, pooled.tn) == (4, 2, 4, 10)
    assert pooled.precision == 4 / 6 and pooled.recall == 4 / 8
    assert pool_metrics([]) == Metrics(0, 0, 0, 0)   # empty corpus -> zero confusion


def test_roc_auc_ranking_agreement():
    # rigged parts all score above rigid ones -> perfect ranking (1.0), regardless of any threshold.
    labeled = [(_dyn("p1", 0.6), True), (_dyn("p2", 0.7), True),
               (_dyn("n1", 0.2), False), (_dyn("n2", 0.3), False)]
    assert roc_auc(labeled) == 1.0
    # fully inverted ranking -> 0.0; a tie between one pos and one neg -> 0.5 (chance).
    assert roc_auc([(_dyn("p", 0.1), True), (_dyn("n", 0.9), False)]) == 0.0
    assert roc_auc([(_dyn("p", 0.5), True), (_dyn("n", 0.5), False)]) == 0.5
    # undefined when a class is empty (can't rank rigged vs non-rigged with only one class present)
    assert roc_auc([(_dyn("p", 0.5), True)]) is None
    assert roc_auc([]) is None
