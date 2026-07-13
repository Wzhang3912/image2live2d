"""Density-aware free-edge trust. On a densely LAYERED scene (a pro rig stacks meshes behind every
part) the free-edge and cantilever cues collapse — everything is backed, so nothing "opens into void"
and no attachment reads as above the mass. The score then leans on cantilever re-anchored at the part
top + slenderness. Below the density floor this is a no-op, so our sparse pipeline stays byte-identical.
"""

from __future__ import annotations

from image2live2d.core.structure.dynamics import (
    _DENSITY_LO,
    _W_CANTILEVER,
    _W_FREE_EDGE,
    _W_MATERIAL,
    _W_SLENDER,
    DynamicsVerdict,
    PartProbe,
    _score_weights,
    score_dynamics,
)
from image2live2d.irr.schema import SemanticRole as R

_BASE = (_W_FREE_EDGE, _W_CANTILEVER, _W_SLENDER, _W_MATERIAL)


def _rect(x0, y0, x1, y1):
    def sampler(u, v):
        return 255 if (x0 <= u <= x1 and y0 <= v <= y1) else 0
    return sampler


def _probe(pid, role, sampler):
    return PartProbe(pid, role, sampler)


def test_score_weights_sparse_is_base_dense_reweights():
    assert _score_weights(0.0) == _BASE                 # sparse -> base weights exactly (byte-identical)
    assert _score_weights(_DENSITY_LO) == _BASE         # still base at the floor
    dense = _score_weights(1.0)
    assert dense[0] < _W_FREE_EDGE                      # free-edge trusted less
    assert dense[1] > _W_CANTILEVER and dense[2] > _W_SLENDER   # weight moved onto cantilever/slenderness
    assert dense[3] == _W_MATERIAL                      # material untouched
    assert abs(sum(dense) - 1.0) < 1e-9                 # still a valid weighting


def test_dense_layering_still_finds_a_backed_hanging_strand():
    full = _rect(0.0, 0.0, 1.0, 1.0)
    probes = [
        _probe("sheet1", R.clothing, full),             # three full-canvas layers = dense backing
        _probe("sheet2", R.clothing, full),
        _probe("sheet3", R.clothing, full),
        _probe("strand", R.hair_side, _rect(0.48, 0.28, 0.52, 0.95)),   # thin, hangs low
        _probe("face", R.face_base, _rect(0.30, 0.05, 0.70, 0.45)),     # not sway-eligible
    ]
    d = {x.part_id: x for x in score_dynamics(probes)}
    assert d["strand"].free_edge_ratio < 0.2            # fully backed -> the void cue has collapsed...
    assert d["strand"].verdict is not DynamicsVerdict.rigid   # ...yet the strand is still caught
    assert d["face"].verdict is DynamicsVerdict.rigid   # the eligibility gate still holds under density


def test_sparse_scene_unaffected():
    # the same strand alone (sparse) is scored by the base weights and is obviously dynamic
    probes = [_probe("strand", R.hair_side, _rect(0.48, 0.10, 0.52, 0.95))]
    d = {x.part_id: x for x in score_dynamics(probes)}
    assert d["strand"].free_edge_ratio > 0.8            # a lone strand opens into void on all sides
    assert d["strand"].verdict is not DynamicsVerdict.rigid
