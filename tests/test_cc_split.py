"""P2 (intra-layer) — connected-component strand split. A single hair layer whose mesh has two
disconnected lobes (twin-tails fused into one part by the decomposer) becomes two independent strands,
each with its own param + pendulum, moving only its own vertices. A single connected blob is unchanged.
"""

from __future__ import annotations

from pathlib import Path

from image2live2d.core.mesh import grid_mesh
from image2live2d.core.physics import generate_physics
from image2live2d.core.rig import author_rig, select_template
from image2live2d.core.structure import hair_strands, mesh_components
from image2live2d.core.types import Layer, LayerStack
from image2live2d.irr.schema import Mesh
from image2live2d.irr.schema import SemanticRole as R


def _two_lobe_mesh(pid: str) -> Mesh:
    """Two vertical bars with a transparent gap between them -> grid_mesh drops the gap cells, leaving
    two disconnected triangle clusters."""
    def alpha(u: float, v: float) -> int:
        return 255 if (u < 0.35 or u > 0.65) else 0
    return grid_mesh(pid, (0.1, 0.2, 0.9, 0.8), alpha, grid=14)


def _one_blob_mesh(pid: str) -> Mesh:
    return grid_mesh(pid, (0.2, 0.2, 0.4, 0.8), lambda u, v: 255, grid=8)


def test_mesh_components_finds_two_lobes():
    comps = mesh_components(_two_lobe_mesh("hair"))
    assert len(comps) == 2
    a, b = set(comps[0]), set(comps[1])
    assert not (a & b)                                   # disjoint
    assert a | b == set(range(len(_two_lobe_mesh("hair").vertices)))  # cover everything


def test_single_blob_is_one_component():
    comps = mesh_components(_one_blob_mesh("hair"))
    assert len(comps) == 1


def _scene_with_fused_hair():
    m = _two_lobe_mesh("hair")
    face = grid_mesh("face", (0.3, 0.55, 0.7, 0.95), lambda u, v: 255, grid=4)
    stack = LayerStack(layers=[
        Layer(id="face", semantic_role=R.face_base, texture_path=Path("face.png"),
              draw_order=0, width=64, height=64),
        Layer(id="hair", semantic_role=R.hair_side, texture_path=Path("hair.png"),
              draw_order=1, width=64, height=64),
    ], canvas_width=64, canvas_height=64)
    return stack, [face, m], m


def test_fused_layer_splits_into_two_strands():
    stack, meshes, _ = _scene_with_fused_hair()
    specs = [s for s in hair_strands(stack, meshes) if s.role is R.hair_side]
    assert [s.param_id for s in specs] == ["ParamHairSide", "ParamHairSide2"]
    assert all(s.part_id == "hair" for s in specs)       # same part, two strands
    ia, ib = set(specs[0].vertex_indices), set(specs[1].vertex_indices)
    assert ia and ib and not (ia & ib)                   # each owns a disjoint, non-empty lobe


def test_each_strand_moves_only_its_own_lobe():
    stack, meshes, _ = _scene_with_fused_hair()
    specs = {s.param_id: set(s.vertex_indices) for s in hair_strands(stack, meshes)
             if s.role is R.hair_side}
    params = {p.id: p for p in author_rig(stack, meshes, select_template(stack)).parameters}

    def nonzero_idx(pid):
        out = set()
        for kf in params[pid].keyforms:
            for i, (dx, dy) in enumerate(kf.mesh_offsets.get("hair", [])):
                if dx or dy:
                    out.add(i)
        return out

    nz1, nz2 = nonzero_idx("ParamHairSide"), nonzero_idx("ParamHairSide2")
    assert nz1 and nz2 and not (nz1 & nz2)               # disjoint moving sets -> independent
    assert nz1 <= specs["ParamHairSide"]                 # each stays within its own lobe
    assert nz2 <= specs["ParamHairSide2"]


def test_fused_layer_gets_two_pendulums():
    stack, meshes, _ = _scene_with_fused_hair()
    params = author_rig(stack, meshes, select_template(stack)).parameters
    outs = {r.output_param for r in generate_physics(stack, params, meshes=meshes)}
    assert {"ParamHairSide", "ParamHairSide2"} <= outs
