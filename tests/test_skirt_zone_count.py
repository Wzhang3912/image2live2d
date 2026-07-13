"""P3b — width-driven skirt zone count. A reference-width hem keeps the three L/C/R lobes (byte-
identical); a markedly wider hem breaks into more evenly-tiled interior lobes (ParamSkirtC1, C2 …) so a
full skirt ripples in more independent zones, while a narrow one never drops below three.
"""

from __future__ import annotations

from pathlib import Path

from image2live2d.core.physics import generate_physics
from image2live2d.core.rig import author_rig, select_template
from image2live2d.core.structure import skirt_zones
from image2live2d.core.structure.skirt import _MAX_ZONES, _REF_SPAN, _zone_count
from image2live2d.core.types import Layer, LayerStack
from image2live2d.irr.schema import Mesh
from image2live2d.irr.schema import SemanticRole as R


def _mesh(pid, x0, y0, x1, y1, cols=1):
    # A cols×1 grid (cols+1 columns, 2 rows) so interior hem windows have vertices to deform — real
    # skirt meshes are dense grids; a bare 4-corner quad would leave narrow interior windows empty.
    xs = [x0 + (x1 - x0) * i / cols for i in range(cols + 1)]
    verts = [(x, y0) for x in xs] + [(x, y1) for x in xs]
    top = cols + 1
    tris = []
    for c in range(cols):
        bl, br, tl, tr = c, c + 1, top + c, top + c + 1
        tris += [(bl, br, tr), (bl, tr, tl)]
    return Mesh(part_id=pid, vertices=verts, uvs=[(0.0, 0.0)] * len(verts), triangles=tris)


def _scene(skirt_box):
    parts = [("torso", R.torso, (0.35, 0.42, 0.65, 0.75), 1),
             ("skirt", R.clothing, skirt_box, 8)]                # dense hem so every zone window bites
    layers, meshes = [], []
    for i, (pid, role, box, cols) in enumerate(parts):
        layers.append(Layer(id=pid, semantic_role=role, texture_path=Path(f"{pid}.png"),
                            draw_order=i * 10, width=64, height=64))
        meshes.append(_mesh(pid, *box, cols=cols))
    return LayerStack(layers=layers, canvas_width=64, canvas_height=64), meshes


# A reference-width hem spans _REF_SPAN of the canvas; a wide one is markedly wider.
_REF_BOX = (0.30, 0.20, 0.70, 0.44)                 # span 0.40 -> 3 zones
_WIDE_BOX = (0.20, 0.20, 0.80, 0.44)                # span 0.60 -> 4 zones (one extra interior)


def test_zone_count_scales_with_width_and_caps():
    assert _zone_count(0.25) == 3                    # narrow never drops below 3
    assert _zone_count(_REF_SPAN) == 3               # reference width -> 3
    assert _zone_count(0.60) == 4                    # +1 lobe
    assert _zone_count(0.76) == 5                    # +2 lobes
    assert _zone_count(10.0) == _MAX_ZONES           # capped


def test_reference_width_stays_lcr():
    zones = skirt_zones(*_scene(_REF_BOX))
    assert [z.param_id for z in zones] == ["ParamSkirtL", "ParamSkirtC", "ParamSkirtR"]


def test_wide_hem_adds_evenly_tiled_interior_zones():
    zones = skirt_zones(*_scene(_WIDE_BOX))
    ids = [z.param_id for z in zones]
    assert ids == ["ParamSkirtL", "ParamSkirtC", "ParamSkirtC1", "ParamSkirtR"]
    # windows evenly tile the span (all equal width) and their centres march strictly left->right
    assert len({round(z.half_width, 6) for z in zones}) == 1
    centers = [z.center_x for z in zones]
    assert centers == sorted(centers) and len(set(centers)) == len(centers)
    cx0, cx1 = 0.20, 0.80
    assert all(cx0 < c < cx1 for c in centers)


def test_author_mints_the_extra_interior_param():
    stack, meshes = _scene(_WIDE_BOX)
    params = {p.id: p for p in author_rig(stack, meshes, select_template(stack)).parameters}
    assert "ParamSkirtC1" in params
    moved = any(any(dx or dy for dx, dy in kf.mesh_offsets.get("skirt", []))
                for kf in params["ParamSkirtC1"].keyforms)
    assert moved                                     # the extra lobe actually deforms the hem


def test_physics_wires_every_zone_of_a_wide_hem():
    stack, meshes = _scene(_WIDE_BOX)
    params = author_rig(stack, meshes, select_template(stack)).parameters
    rigs = {r.output_param: r for r in generate_physics(stack, params, meshes=meshes)}
    assert {"ParamSkirtL", "ParamSkirtC", "ParamSkirtC1", "ParamSkirtR"} <= set(rigs)
    # the extra interior lobe is a body-driven springy pendulum, like the centre zone
    assert rigs["ParamSkirtC1"].driver_param == "ParamBodyAngleX"
    assert rigs["ParamSkirtC1"].length > 0
