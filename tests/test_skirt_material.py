"""P3 — material from geometry for the skirt hem. A garment's pendulum mass/length now scale with its
actual size (a long dress swings bigger/slower than a mini-skirt), while the L/C/R zone structure and
the mesh-less (base-tuning) path stay unchanged.
"""

from __future__ import annotations

from pathlib import Path

from image2live2d.core.physics import generate_physics
from image2live2d.core.rig import author_rig, select_template
from image2live2d.core.structure import material_from_geometry, skirt_zones
from image2live2d.core.types import Layer, LayerStack
from image2live2d.irr.schema import Mesh
from image2live2d.irr.schema import SemanticRole as R

_BASE = (1.8, 0.25, 1.5)   # the centre-zone base tuning


def _mesh(pid, x0, y0, x1, y1):
    verts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return Mesh(part_id=pid, vertices=verts, uvs=[(0.0, 0.0)] * 4, triangles=[(0, 1, 2), (0, 2, 3)])


def _scene(skirt_box):
    parts = [("torso", R.torso, (0.35, 0.42, 0.65, 0.75)),
             ("skirt", R.clothing, skirt_box)]
    layers, meshes = [], []
    for i, (pid, role, box) in enumerate(parts):
        layers.append(Layer(id=pid, semantic_role=role, texture_path=Path(f"{pid}.png"),
                            draw_order=i * 10, width=64, height=64))
        meshes.append(_mesh(pid, *box))
    return LayerStack(layers=layers, canvas_width=64, canvas_height=64), meshes


def test_material_from_geometry_scales_with_size():
    # reference-sized garment -> base tuning verbatim
    assert material_from_geometry(_BASE, 0.22, 0.09) == _BASE
    # a longer hem -> a longer, floppier pendulum
    _, d_long, l_long = material_from_geometry(_BASE, 0.44, 0.09)
    assert l_long > _BASE[2] and d_long < _BASE[1]
    # more fabric area -> more mass (more lag)
    m_big, _, _ = material_from_geometry(_BASE, 0.22, 0.18)
    assert m_big > _BASE[0]


def _center_length(skirt_box, meshes_stack):
    stack, meshes = meshes_stack
    z = next(z for z in skirt_zones(stack, meshes) if z.param_id == "ParamSkirtC")
    return z.length


def test_longer_skirt_gets_a_longer_pendulum():
    long_zones = skirt_zones(*_scene((0.30, 0.10, 0.70, 0.42)))    # hem hangs from 0.42 down to 0.10
    short_zones = skirt_zones(*_scene((0.30, 0.32, 0.70, 0.42)))   # short hem
    long_c = next(z for z in long_zones if z.param_id == "ParamSkirtC")
    short_c = next(z for z in short_zones if z.param_id == "ParamSkirtC")
    assert long_c.length > short_c.length
    # the L/C/R structure is intact
    assert {z.param_id for z in long_zones} == {"ParamSkirtL", "ParamSkirtC", "ParamSkirtR"}


def test_physics_geometry_scaled_with_meshes_base_without():
    stack, meshes = _scene((0.30, 0.10, 0.70, 0.42))               # a long skirt
    params = author_rig(stack, meshes, select_template(stack)).parameters
    scaled = {r.output_param: r for r in generate_physics(stack, params, meshes=meshes)}
    base = {r.output_param: r for r in generate_physics(stack, params)}   # no meshes
    assert scaled["ParamSkirtC"].length > base["ParamSkirtC"].length      # geometry made it longer
    assert base["ParamSkirtC"].length == _BASE[2]                         # mesh-less path = base tuning


def test_skirt_structure_and_drivers_unchanged():
    stack, meshes = _scene((0.30, 0.25, 0.70, 0.42))
    params = author_rig(stack, meshes, select_template(stack)).parameters
    rigs = {r.output_param: r for r in generate_physics(stack, params, meshes=meshes)}
    assert {"ParamSkirtL", "ParamSkirtC", "ParamSkirtR"} <= set(rigs)
    # centre zone is driven by body sway, not a leg
    assert rigs["ParamSkirtC"].driver_param == "ParamBodyAngleX"
