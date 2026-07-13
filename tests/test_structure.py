"""P1 — dynamics-score detector (core.structure). Pure-core tests with synthetic silhouettes.

Each part is an analytic rectangle sampler over the full canvas (u right, v DOWN, [0,1]); a small
scene lets the cross-part free-edge detector see real "is the gap filled by another part?" cases.
Asserts the four signals and the verdict/class on hand-built shapes whose right answer is obvious:
a hanging strand -> dynamic strand; a hem sheet -> dynamic sheet; an enclosed part -> rigid; an
articulated limb -> rigid (ineligible), never a strand.
"""

from __future__ import annotations

import pytest

from image2live2d.core.structure import (
    DynamicsVerdict, PartProbe, PhysicalClass, score_dynamics,
)
from image2live2d.irr.schema import SemanticRole as R


def rect(u0: float, u1: float, v0: float, v1: float):
    """A sampler that is opaque inside the (u, v) box (v down), transparent outside."""
    def s(u: float, v: float) -> int:
        return 255 if (u0 <= u <= u1 and v0 <= v <= v1) else 0
    return s


def _scene() -> list[PartProbe]:
    # A coherent little character (v down: small v = top of canvas).
    return [
        PartProbe("head", R.face_base, rect(0.15, 0.85, 0.05, 0.36)),        # wide head band
        PartProbe("side_hair", R.hair_side, rect(0.20, 0.28, 0.12, 0.60)),   # strand hanging below it
        PartProbe("torso", R.torso, rect(0.34, 0.66, 0.32, 0.58)),
        PartProbe("skirt", R.clothing, rect(0.30, 0.70, 0.56, 0.82)),        # hem off the torso
        PartProbe("collar", R.clothing, rect(0.42, 0.58, 0.30, 0.40)),       # enclosed by head+torso
        PartProbe("arm_l", R.arm_l, rect(0.04, 0.12, 0.36, 0.70)),           # slender but articulated
        PartProbe("back_hair", R.hair_back, rect(0.0, 0.0, 0.0, 0.0)),       # empty layer
    ]


def _by_id(samples: int = 100) -> dict[str, object]:
    return {d.part_id: d for d in score_dynamics(_scene(), samples=samples)}


def test_empty_layer_is_skipped():
    assert "back_hair" not in _by_id()


def test_all_scores_bounded():
    for d in _by_id().values():
        assert 0.0 <= d.score <= 1.0
        assert 0.0 <= d.free_edge_ratio <= 1.0
        assert 0.0 <= d.cantilever <= 1.0
        assert d.slenderness >= 1.0 - 1e-9


def test_hanging_strand_is_dynamic_strand():
    d = _by_id()["side_hair"]
    assert d.sway_eligible
    assert d.free_edge_ratio > 0.4          # substantially opens into empty space (vs ~0 enclosed)
    assert d.cantilever > 0.4               # hangs well past its (top) attachment
    assert d.slenderness > 2.5              # long and thin
    assert d.verdict is DynamicsVerdict.dynamic
    assert d.physical_class is PhysicalClass.strand
    # anchor (attachment centroid) sits up near the head, above the strand's mass centroid.
    assert d.anchor[1] > 0.55               # y up: near the top of the canvas


def test_hem_is_dynamic_sheet():
    d = _by_id()["skirt"]
    assert d.verdict is DynamicsVerdict.dynamic
    assert d.slenderness < 2.5              # wide, not a strand
    assert d.cantilever > 0.4               # hangs from the waist
    assert d.physical_class is PhysicalClass.sheet


def test_enclosed_part_is_rigid():
    d = _by_id()["collar"]
    assert d.free_edge_ratio < 0.25         # every border is filled by another part
    assert d.verdict is DynamicsVerdict.rigid
    assert d.physical_class is PhysicalClass.rigid


def test_articulated_limb_is_not_a_strand():
    # An arm is slender and hangs freely, but limbs ARTICULATE about a joint — they must never be
    # scored as sway-physics strands. The eligibility gate forces rigid regardless of geometry.
    d = _by_id()["arm_l"]
    assert not d.sway_eligible
    assert d.verdict is DynamicsVerdict.rigid
    assert d.physical_class is PhysicalClass.rigid
    assert d.material_prior == 0.0


def test_name_hint_boosts_material_prior():
    # A lone accessory whose id names a soft danging element gets a prior boost (0.5 -> ~0.7).
    [d] = score_dynamics([PartProbe("hair_ribbon", R.accessory, rect(0.45, 0.55, 0.1, 0.6))],
                         samples=80)
    assert d.material_prior > 0.65
    assert d.verdict is DynamicsVerdict.dynamic   # lone part: every edge is free


def test_attachment_fraction_complements_free_edge():
    # attachment_fraction is the pinned complement of the free-edge ratio (attached vs free boundary).
    d = _by_id()
    for part in ("side_hair", "collar", "skirt"):
        assert abs(d[part].attachment_fraction - (1.0 - d[part].free_edge_ratio)) < 1e-9
    # an enclosed collar is mostly pinned; a hanging strand is much freer.
    assert d["collar"].attachment_fraction > 0.6
    assert d["side_hair"].attachment_fraction < d["collar"].attachment_fraction


def test_depth_isolation_backed_vs_free():
    d = _by_id()
    for part in d.values():
        assert 0.0 <= part.depth_isolation <= 1.0
    # the collar sits under the head + torso (its footprint is backed) -> low isolation; the strand's
    # footprint is largely its own -> higher.
    assert d["collar"].depth_isolation < d["side_hair"].depth_isolation
    # a lone part shares its footprint with nothing -> fully isolated.
    [solo] = score_dynamics([PartProbe("solo", R.hair_side, rect(0.4, 0.6, 0.1, 0.6))], samples=80)
    assert solo.depth_isolation == 1.0


def test_new_signals_do_not_change_the_score():
    # attachment_fraction / depth_isolation are diagnostics, never score terms: the score for a lone
    # strand is exactly the free/cantilever/slender/material blend, unaffected by the new fields.
    [d] = score_dynamics([PartProbe("strand", R.hair_side, rect(0.48, 0.52, 0.10, 0.95))], samples=80)
    expected = (0.45 * d.free_edge_ratio + 0.30 * d.cantilever
                + 0.15 * min((d.slenderness - 1.0) / 3.0, 1.0) + 0.10 * d.material_prior)
    assert abs(d.score - expected) < 1e-9


def test_samples_floor():
    with pytest.raises(ValueError):
        score_dynamics(_scene(), samples=1)
