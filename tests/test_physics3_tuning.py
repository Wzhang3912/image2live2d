"""Cubism ``physics3.json`` tuning against the real-artist regime (RIVAL_HARVEST_BACKLOG T7).

The Mobility/Delay/Acceleration regime below is the union measured off two real pro rigs (Hiyori +
Akari) by ``tools/feel_parity.py``. Emitting outside it means a real Live2D runtime swings our parts
more weakly or stiffly than any artist would author — which is exactly what our cloth used to do.
"""

from __future__ import annotations

import importlib.util

import pytest

from image2live2d.backends.live2d.physics3 import _output_scale, physics3

# Union of Hiyori + Akari, as reported by tools/feel_parity.py.
REAL_MOBILITY = (0.71, 1.00)
REAL_DELAY = (0.60, 1.00)
REAL_ACCEL = (0.80, 3.00)

_HAS_PIL = importlib.util.find_spec("PIL") is not None


def test_hair_output_scale_is_per_role_not_flat():
    # Hiyori drives the fringe far less than the long back hair (1.522 vs 2.061). A flat scale both
    # under-drove the back hair and erased the front/back contrast that makes hair read as layered.
    front, side, back = (_output_scale(f"ParamHair{r}") for r in ("Front", "Side", "Back"))
    assert front < side < back
    assert front == pytest.approx(1.52) and back == pytest.approx(2.06)


def test_bounce_and_numbered_hair_variants_ride_their_base_role():
    assert _output_scale("ParamHairBackV") == _output_scale("ParamHairBack")
    assert _output_scale("ParamHairSide2") == _output_scale("ParamHairSide")


def test_cloth_and_unknown_outputs_stay_at_unity():
    # Skirts are driven by their own pendulum; scaling them up over-drives the fabric.
    assert _output_scale("ParamSkirtL") == 1.0
    assert _output_scale("ParamBustY") == 1.0


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not installed")
def test_every_emitted_pendulum_sits_in_the_real_artist_regime(tmp_path):
    """The regression this locks: our skirt used to emit raw mobility 0.61-0.65 — below the real floor
    (~0.71) — so it was clamped, i.e. we shipped cloth more heavily damped than any rig on hand."""
    from image2live2d.core import decompose
    from image2live2d.pipeline import rig_from_stack
    from image2live2d.samples import make_sample_fullbody

    layers = make_sample_fullbody(tmp_path / "fb", size=256)
    rig = rig_from_stack(decompose.from_layer_dir(layers), name="fb")
    settings = physics3(rig)["PhysicsSettings"]
    assert settings, "fullbody sample emits no physics"

    skirts = 0
    for s in settings:
        pid = s["Output"][0]["Destination"]["Id"]
        tip = s["Vertices"][-1]          # the swinging tip carries the material
        assert REAL_MOBILITY[0] <= tip["Mobility"] <= REAL_MOBILITY[1], f"{pid} mobility {tip}"
        assert REAL_DELAY[0] <= tip["Delay"] <= REAL_DELAY[1], f"{pid} delay {tip}"
        assert REAL_ACCEL[0] <= tip["Acceleration"] <= REAL_ACCEL[1], f"{pid} accel {tip}"
        if pid.startswith("ParamSkirt"):
            skirts += 1
            # strictly ABOVE the floor, not sitting on the clamp — the pre-T7 failure mode
            assert tip["Mobility"] > REAL_MOBILITY[0] + 0.05, f"{pid} is pinned to the clamp floor"
    assert skirts, "fullbody sample emits no skirt physics"


@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not installed")
def test_normalization_matches_the_reference_plus_minus_ten(tmp_path):
    from image2live2d.core import decompose
    from image2live2d.pipeline import rig_from_stack
    from image2live2d.samples import make_sample_fullbody

    rig = rig_from_stack(
        decompose.from_layer_dir(make_sample_fullbody(tmp_path / "fb", size=256)), name="fb")
    for s in physics3(rig)["PhysicsSettings"]:
        for axis in ("Position", "Angle"):
            assert s["Normalization"][axis]["Minimum"] == -10.0
            assert s["Normalization"][axis]["Maximum"] == 10.0
