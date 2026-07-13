"""P2 — per-strand hair. Twin-tails / multi hair parts get their own param + pendulum (independent),
while a single strand per role is unchanged. Also covers geometry-scaled material and id minting.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from image2live2d.core.physics import generate_physics
from image2live2d.core.rig import author_rig, select_template
from image2live2d.core.structure import hair_strands
from image2live2d.core.types import Layer, LayerStack
from image2live2d.irr.params import make_parameter
from image2live2d.irr.schema import Mesh
from image2live2d.irr.schema import SemanticRole as R


def _mesh(pid, x0, y0, x1, y1):
    verts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return Mesh(part_id=pid, vertices=verts, uvs=[(0.0, 0.0)] * 4, triangles=[(0, 1, 2), (0, 2, 3)])


def _scene(parts):
    layers, meshes = [], []
    for i, (pid, role, box) in enumerate(parts):
        layers.append(Layer(id=pid, semantic_role=role, texture_path=Path(f"{pid}.png"),
                            draw_order=i * 10, width=64, height=64))
        meshes.append(_mesh(pid, *box))
    return LayerStack(layers=layers, canvas_width=64, canvas_height=64), meshes


# A face + two side-tails of different lengths (left longer than right).
_TWINTAILS = [
    ("face", R.face_base, (0.35, 0.60, 0.65, 0.95)),
    ("tail_l", R.hair_side, (0.20, 0.20, 0.30, 0.80)),   # height 0.60
    ("tail_r", R.hair_side, (0.70, 0.30, 0.80, 0.80)),   # height 0.50
]


def test_hair_strands_names_and_geometry_scaling():
    stack, meshes = _scene(_TWINTAILS)
    specs = {s.part_id: s for s in hair_strands(stack, meshes)}
    assert specs["tail_l"].param_id == "ParamHairSide"      # first part keeps the base id
    assert specs["tail_r"].param_id == "ParamHairSide2"     # extra part gets a suffix
    # longer tail -> heavier + longer pendulum (more lag); shorter tail -> lighter
    assert specs["tail_l"].mass > specs["tail_r"].mass
    assert specs["tail_l"].length > specs["tail_r"].length


def test_twintails_get_independent_sway_params():
    stack, meshes = _scene(_TWINTAILS)
    params = {p.id: p for p in author_rig(stack, meshes, select_template(stack)).parameters}
    assert "ParamHairSide" in params and "ParamHairSide2" in params

    def parts_moved(pid):
        return {k for kf in params[pid].keyforms for k, offs in kf.mesh_offsets.items()
                if any(dx or dy for dx, dy in offs)}

    assert parts_moved("ParamHairSide") == {"tail_l"}       # each param drives only its own strand
    assert parts_moved("ParamHairSide2") == {"tail_r"}


def test_twintails_get_independent_pendulums_with_distinct_material():
    stack, meshes = _scene(_TWINTAILS)
    params = author_rig(stack, meshes, select_template(stack)).parameters
    rigs = {r.output_param: r for r in generate_physics(stack, params, meshes=meshes)}
    assert "ParamHairSide" in rigs and "ParamHairSide2" in rigs
    assert rigs["ParamHairSide"].mass > rigs["ParamHairSide2"].mass   # geometry-scaled -> desync
    # both are driven by the head turn, independently
    assert rigs["ParamHairSide"].driver_param == "ParamAngleX"
    assert rigs["ParamHairSide2"].driver_param == "ParamAngleX"


def test_single_strand_unchanged():
    stack, meshes = _scene([
        ("face", R.face_base, (0.35, 0.60, 0.65, 0.95)),
        ("side", R.hair_side, (0.20, 0.20, 0.30, 0.80)),
    ])
    params = author_rig(stack, meshes, select_template(stack)).parameters
    ids = {p.id for p in params}
    assert "ParamHairSide" in ids and "ParamHairSide2" not in ids
    rigs = {r.output_param: r for r in generate_physics(stack, params, meshes=meshes)}
    assert rigs["ParamHairSide"].mass == 1.4                # base tuning, factor 1.0


def test_physics_without_meshes_still_splits_but_uses_base_tuning():
    stack, meshes = _scene(_TWINTAILS)
    params = author_rig(stack, meshes, select_template(stack)).parameters
    rigs = {r.output_param: r for r in generate_physics(stack, params)}   # no meshes
    assert "ParamHairSide" in rigs and "ParamHairSide2" in rigs           # still split
    assert rigs["ParamHairSide"].mass == rigs["ParamHairSide2"].mass == 1.4  # base tuning


def test_make_parameter_mints_strand_ids():
    p = make_parameter("ParamHairSide2")
    assert (p.min, p.max, p.default) == (-1.0, 1.0, 0.0)
    assert make_parameter("ParamHairFront3").id == "ParamHairFront3"
    with pytest.raises(KeyError):
        make_parameter("ParamBogus9")
