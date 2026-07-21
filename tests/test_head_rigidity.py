"""The protected-region rigidity field that keeps the eyes/nose/mouth from foreshortening with the
head under the turn warp (RIVAL_HARVEST_BACKLOG T5). Pure geometry — no backend, no native core."""

from __future__ import annotations

from image2live2d.core.rig.head_rigidity import PROTECT, regions_from, rigidity_field


def test_protected_roles_carry_the_rival_weights():
    # Eyes rigid (1.0); nose/mouth partly (0.30) — the anime convention every rival implements.
    assert PROTECT["eye_l"] == 1.0 and PROTECT["eye_r"] == 1.0
    assert PROTECT["nose"] == 0.30 and PROTECT["mouth"] == 0.30
    assert "hair_front" not in PROTECT and "face_base" not in PROTECT


def test_regions_from_keeps_protected_roles_and_skips_absent_ones():
    regions = regions_from({"eye_l": (0.0, 0.0, 1.0, 1.0), "nose": None, "face_base": (2.0, 2.0, 3.0, 3.0)})
    # eye_l kept with its weight; nose is None (absent); face_base is not a protected role
    assert regions == [(0.0, 0.0, 1.0, 1.0, 1.0)]


def test_a_point_inside_an_eye_is_fully_rigid_and_rides_the_eye_centroid():
    regions = regions_from({"eye_l": (0.0, 0.0, 2.0, 1.0)})
    (w, cx, cy), = rigidity_field([(1.0, 0.5)], regions)
    assert w == 1.0
    assert (cx, cy) == (1.0, 0.5)      # bbox centre


def test_rigidity_ramps_to_zero_outside_the_bbox():
    # bbox 2 wide -> margin 0.5*2 = 1.0; a point one full margin past the edge is unprotected.
    regions = regions_from({"eye_l": (0.0, 0.0, 2.0, 1.0)})
    (w_edge, _, _), = rigidity_field([(3.0, 0.5)], regions)   # 1.0 past the x=2 edge == one margin
    assert w_edge == 0.0
    (w_half, _, _), = rigidity_field([(2.5, 0.5)], regions)   # halfway into the margin band
    assert 0.4 < w_half < 0.6


def test_a_stronger_region_wins_where_two_overlap():
    # A point inside both the eye (1.0) and a nearby nose (0.30) rides the eye — the max influence.
    regions = regions_from({"eye_l": (0.0, 0.0, 1.0, 1.0), "nose": (0.5, 0.0, 1.5, 1.0)})
    (w, cx, _), = rigidity_field([(0.75, 0.5)], regions)
    assert w == 1.0
    assert cx == 0.5                    # the eye's centroid, not the nose's


def test_an_unprotected_point_reports_itself_as_a_harmless_centroid():
    (w, cx, cy), = rigidity_field([(9.0, 9.0)], regions_from({"eye_l": (0.0, 0.0, 1.0, 1.0)}))
    assert w == 0.0
    assert (cx, cy) == (9.0, 9.0)       # its own position -> a rigid blend against it is a no-op
