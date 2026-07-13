"""Unit coverage for the pure name-heuristics in tools/calibrate_moc3.py (the real-model calibration
bridge). The ctypes/Cubism-core path needs a proprietary dylib + a local model, so it isn't tested in
CI; these deterministic helpers — physics-target decoding, segment→part grouping, role inference,
label matching — are, since they encode the model-naming assumptions the calibration rests on.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from calibrate_moc3 import (  # noqa: E402
    _has_physics,
    _role_of,
    group_key,
    physics_target_stems,
)
from image2live2d.irr.schema import SemanticRole as R


def test_physics_target_stems_decodes_param_names():
    stems = physics_target_stems([
        "Param_Angle_Rotation_3_hair_left", "Param_Angle_Rotation_1_tie_top_right",
        "ParamSkirtPhysicsA", "ParamBreastPhysicsX", "ShirtTopPhysicsA", "ParamAngleX",
    ])
    assert "hair_left" in stems
    assert "tie_top_right" in stems
    assert "skirt" in stems
    assert "breast" in stems
    assert "shirt_top" in stems
    # ParamAngleX has no "Physics"/rotation marker -> not a physics target
    assert not any("angle" in s for s in stems)


def test_group_key_merges_segments_and_shades():
    # a hair strand's segments + its shade layers collapse to one part
    assert group_key("hair_left2") == "hair_left"
    assert group_key("hair_left8") == "hair_left"
    assert group_key("tie_bottom_left_shade6") == "tie_bottom_left"
    assert group_key("eye_iris_left_shade") == "eye_iris_left"
    # distinct strands stay distinct
    assert group_key("hair_mid_left3") == "hair_mid_left"
    assert group_key("hair_left3") != group_key("hair_mid_left3")


def test_role_inference_gates_eligibility():
    assert _role_of("hair_left") is R.hair_side           # hair -> sway-eligible
    assert _role_of("skirt_main") is R.clothing           # cloth -> sway-eligible
    assert _role_of("eye_iris_left") is R.eye_l           # facial -> NOT eligible (stays rigid)
    assert _role_of("arm_main_left") is R.arm_l           # limb -> NOT eligible
    assert _role_of("qwerty_unknown") is None             # unknown -> skipped, not mislabeled


def test_has_physics_matches_targets_and_aliases():
    stems = {"hair_left", "skirt", "breast"}
    assert _has_physics("hair_left", stems)               # exact strand
    assert _has_physics("skirt_main", stems)              # prefix
    assert _has_physics("oppai_left", stems)              # breast -> oppai alias
    assert _has_physics("under_boob_shadow", stems)       # breast -> under_boob alias
    assert not _has_physics("eye_iris_left", stems)       # no physics target
