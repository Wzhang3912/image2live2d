"""P2 — per-strand hair. Twin-tails / multi hair parts get their own param + pendulum (independent),
while a single strand per role is unchanged. Also covers geometry-scaled material and id minting.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from image2live2d.core.physics import generate_physics
from image2live2d.core.rig import author_rig, select_template
from image2live2d.core.structure import hair_strands, split_lobe_by_tips
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


def test_hair_sways_like_a_cantilever_not_a_sliding_sheet():
    """The roots stay clamped to the scalp; the motion belongs to the tips.

    A linear taper let the whole sheet shear sideways: a fringe slid bodily off the forehead, exposing
    the hairline so the character read as balding. Hair is a cantilever — stiff where it is attached —
    and the real thing is far more tip-concentrated than linear: measured through the native Cubism core,
    a mid-strand vertex of Hiyori's bangs moves 0.11x its tip, where a linear taper would give 0.5x.
    """
    from pathlib import Path

    from image2live2d.core.mesh import grid_mesh
    from image2live2d.core.rig import author_rig, select_template
    from image2live2d.core.types import Layer, LayerStack
    from image2live2d.irr.schema import SemanticRole as R

    layers = [Layer(id="hair_front", semantic_role=R.hair_front, texture_path=Path("h.png"),
                    draw_order=90, width=64, height=64),
              Layer(id="face_base", semantic_role=R.face_base, texture_path=Path("f.png"),
                    draw_order=0, width=64, height=64)]
    meshes = [grid_mesh("hair_front", (0.30, 0.60, 0.70, 0.95), lambda u, v: 255, grid=8),
              grid_mesh("face_base", (0.30, 0.55, 0.70, 0.92), lambda u, v: 255, grid=2)]
    stack = LayerStack(layers=layers, canvas_width=64, canvas_height=64)
    params = author_rig(stack, meshes, select_template(stack)).parameters

    sway = next(p for p in params if p.id.startswith("ParamHairFront"))
    kf = max(sway.keyforms, key=lambda k: k.value)
    offs = kf.mesh_offsets["hair_front"]
    hm = next(m for m in meshes if m.part_id == "hair_front")
    top = max(y for _, y in hm.vertices)
    length = top - min(y for _, y in hm.vertices)

    def dx_at(depth_frac, tol=0.08):
        vals = [abs(offs[i][0]) for i, (_, y) in enumerate(hm.vertices)
                if abs((top - y) / length - depth_frac) < tol]
        return sum(vals) / len(vals)

    root, mid, tip = dx_at(0.0), dx_at(0.5), dx_at(1.0)
    assert root == pytest.approx(0.0, abs=1e-9)      # clamped at the scalp
    assert tip > 0.0                                 # the tips still swing
    # quadratic: mid moves ~1/4 of the tip, not the ~1/2 a shearing sheet would give
    assert mid / tip < 0.35


def _hair_mesh(pid, bottom_of, *, cols=24, rows=6):
    """A hair sheet whose BOTTOM edge follows ``bottom_of(x)`` (y is y-DOWN, so a larger value hangs
    lower). One connected grid; triangles stitch neighbouring columns so it is a single component."""
    verts, index = [], {}
    for c in range(cols):
        x = c / (cols - 1)
        top, bot = 0.15, bottom_of(x)
        for r in range(rows):
            index[(c, r)] = len(verts)
            verts.append((x, top + (bot - top) * r / (rows - 1)))
    tris = []
    for c in range(cols - 1):
        for r in range(rows - 1):
            a, b = index[(c, r)], index[(c + 1, r)]
            cc, d = index[(c, r + 1)], index[(c + 1, r + 1)]
            tris.append((a, b, cc))
            tris.append((b, d, cc))
    return Mesh(part_id=pid, vertices=verts, uvs=[(0.0, 0.0)] * len(verts), triangles=tris)


def _two_locks(x):   # two hanging locks at x~0.2 and x~0.8, a higher saddle between
    import math
    dip = max(math.exp(-((x - 0.2) ** 2) / 0.01), math.exp(-((x - 0.8) ** 2) / 0.01))
    return 0.55 + 0.35 * dip


def test_split_lobe_by_tips_finds_two_locks():
    m = _hair_mesh("hair", _two_locks)
    groups = split_lobe_by_tips(m, list(range(len(m.vertices))))
    assert len(groups) == 2
    # each lock owns a contiguous x-slice; the split falls near the saddle (x~0.5)
    left_max = max(m.vertices[i][0] for i in groups[0])
    right_min = min(m.vertices[i][0] for i in groups[1])
    assert left_max <= right_min                       # no interleaving
    assert 0.35 < left_max < 0.65                       # split near the saddle, not at an edge


def test_a_round_lobe_is_not_force_split():
    # a single smooth bump (one tip) must stay one strand — a bun is not two locks
    m = _hair_mesh("hair", lambda x: 0.9 - 0.3 * (2 * x - 1) ** 2)
    assert split_lobe_by_tips(m, list(range(len(m.vertices)))) == [list(range(len(m.vertices)))]


def test_connected_hair_sheet_with_two_locks_yields_two_strands():
    layers = [Layer(id="face", semantic_role=R.face_base, texture_path=Path("f.png"),
                    draw_order=0, width=64, height=64),
              Layer(id="fringe", semantic_role=R.hair_front, texture_path=Path("h.png"),
                    draw_order=10, width=64, height=64)]
    meshes = [_mesh("face", 0.35, 0.6, 0.65, 0.95), _hair_mesh("fringe", _two_locks)]
    stack = LayerStack(layers=layers, canvas_width=64, canvas_height=64)

    specs = [s for s in hair_strands(stack, meshes) if s.role is R.hair_front]
    assert {s.param_id for s in specs} == {"ParamHairFront", "ParamHairFront2"}
    # each strand owns a real vertex subset (not the whole mesh) and they are disjoint
    assert all(s.vertex_indices for s in specs)
    a, b = (set(s.vertex_indices) for s in specs)
    assert not (a & b)
