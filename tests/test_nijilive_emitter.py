"""Phase 0 milestone: IRR Rig -> nijilive .inp, verified by reading it back.

We can't run the D runtime here, so we validate (a) the container frames correctly, (b) the puppet
JSON matches nijilive's strict shape constraints (the ones its loader ``enforce``s), and (c) the
embedded IRR round-trips.
"""

from __future__ import annotations

import json

from image2live2d.backends.nijilive import NijiliveEmitter
from image2live2d.backends.nijilive.emitter import IRR_EXT_NAME
from image2live2d.backends.nijilive.inp import InpFile
from image2live2d.backends.nijilive.puppet import NO_TEXTURE, build_puppet, solid_png
from image2live2d.irr import Rig
from image2live2d.irr.example import build_example_rig


def _iter_parts(node):
    """Yield every Part node in the tree (parts may now be nested under a 'head' group node)."""
    if node.get("type") == "Part":
        yield node
    for c in node.get("children", []):
        yield from _iter_parts(c)


def test_emit_example_writes_loadable_inp(tmp_path):
    rig = build_example_rig()
    out = NijiliveEmitter().emit(rig, tmp_path)
    assert out.exists() and out.suffix == ".inp"

    inp = InpFile.read(out)
    puppet = json.loads(inp.payload)

    # Container: two textures (face + mouth), both valid PNGs.
    assert len(inp.textures) == 2
    assert all(t.data[:8] == b"\x89PNG\r\n\x1a\n" for t in inp.textures)

    # Top-level keys present.
    assert set(puppet) >= {"meta", "physics", "nodes", "param", "automation", "animations"}
    assert puppet["meta"]["name"] == "example"


def test_puppet_node_and_mesh_shape():
    build = build_puppet(build_example_rig())
    root = build.puppet["nodes"]
    assert root["type"] == "Node"
    parts = list(_iter_parts(root))
    assert {p["name"] for p in parts} == {"face_base", "mouth"}
    for p in parts:
        assert p["type"] == "Part"
        mesh = p["mesh"]
        # verts/uvs are FLAT [x,y,x,y,...]; 4 verts -> 8 floats; 2 tris -> 6 indices
        assert len(mesh["verts"]) == 8
        assert len(mesh["uvs"]) == 8
        assert len(mesh["indices"]) == 6
        assert mesh["origin"] == [0.0, 0.0]
        # albedo slot + 2 empty slots
        assert len(p["textures"]) == 3
        assert p["textures"][1] == NO_TEXTURE


def test_draw_order_inverts_to_zsort():
    """nijilive draws lowest zsort on top; IRR uses higher draw_order = on top. The emitter must
    negate so a higher-draw_order part (mouth) ends up with a *lower* zsort than face_base."""
    build = build_puppet(build_example_rig())
    by_name = {p["name"]: p for p in _iter_parts(build.puppet["nodes"])}
    assert by_name["mouth"]["zsort"] < by_name["face_base"]["zsort"]


def test_y_is_negated_for_nijilive():
    """IRR is y-up; nijilive is y-down. Emitter must negate vertex y (and deform dy)."""
    build = build_puppet(build_example_rig())
    face = next(p for p in _iter_parts(build.puppet["nodes"]) if p["name"] == "face_base")
    # y is negated (IRR y-up -> nijilive y-down). face_base is nested under the re-centered 'head'
    # group, so verts are offset by the neck pivot — assert the negation *invariant* (top/bottom pairs
    # equal, 1000-unit span, bottom > top) rather than absolute values.
    ys = face["mesh"]["verts"][1::2]
    assert ys[0] == ys[1] and ys[2] == ys[3]
    assert ys[0] - ys[2] == 500.0 * 2  # span preserved
    assert ys[0] > ys[2]               # IRR-bottom maps to larger y-down (negated)


def test_deform_binding_grid_matches_axis_points():
    """nijilive's loader enforces values/isSet dims == axis_points dims. Verify we satisfy it."""
    build = build_puppet(build_example_rig())
    param = next(p for p in build.puppet["param"] if p["name"] == "ParamMouthOpenY")

    ax_x, ax_y = param["axis_points"]
    assert ax_x == [0.0, 1.0]  # keyforms at value 0 and 1 over range [0,1]
    assert ax_y == [0.0]

    binding = param["bindings"][0]
    assert binding["param_name"] == "deform"
    values, is_set = binding["values"], binding["isSet"]
    # outer == len(ax_x), inner == len(ax_y)
    assert len(values) == len(ax_x)
    assert len(is_set) == len(ax_x)
    for x_cell, set_row in zip(values, is_set):
        assert len(x_cell) == len(ax_y)
        assert len(set_row) == len(ax_y)
        # each deform cell = one [dx,dy] per vertex (mouth quad = 4 verts)
        assert len(x_cell[0]) == 4
        assert all(len(pair) == 2 for pair in x_cell[0])

    # at max keypoint, the bottom two mouth verts moved in y (scaled), others zero
    max_cell = values[1][0]
    moved = [pair for pair in max_cell if pair != [0.0, 0.0]]
    assert len(moved) == 2


def test_embedded_irr_roundtrips(tmp_path):
    rig = build_example_rig()
    out = NijiliveEmitter().emit(rig, tmp_path)
    inp = InpFile.read(out)
    entry = next(e for e in inp.ext if e.name == IRR_EXT_NAME)
    restored = Rig.model_validate_json(entry.payload.decode("utf-8"))
    assert restored == rig


def test_solid_png_is_valid_png():
    png = solid_png(4, 3)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert b"IHDR" in png[:24] and png.endswith(b"IEND" + png[-4:])


def _eye_crossfade_rig():
    from image2live2d.irr.schema import (
        Keyform, Mesh, Meta, Parameter, Part, Rig, SemanticRole, Texture,
    )
    tri = [(0.45, 0.45), (0.55, 0.45), (0.5, 0.55)]
    uvs, tris = [(0, 0), (1, 0), (0.5, 1)], [(0, 1, 2)]
    return Rig(
        meta=Meta(name="t"),
        textures=[Texture(id="t0", path="t0.png", width=64, height=64)],
        parts=[
            Part(id="eye", semantic_role=SemanticRole.eye_l, texture_id="t0", draw_order=5),
            Part(id="lash", semantic_role=SemanticRole.eye_closed_l, texture_id="t0", draw_order=6),
        ],
        meshes=[Mesh(part_id="eye", vertices=tri, uvs=uvs, triangles=tris),
                Mesh(part_id="lash", vertices=tri, uvs=uvs, triangles=tris)],
        parameters=[Parameter(id="ParamEyeLOpen", min=0.0, max=1.0, default=1.0, keyforms=[
            Keyform(value=0.0, opacity_overrides={"eye": 0.0, "lash": 1.0}),
            Keyform(value=1.0, opacity_overrides={"eye": 1.0, "lash": 0.0}),
        ])],
    )


def test_opacity_overrides_become_nijilive_opacity_bindings():
    # A closed-eye crossfade: the .inp must carry opacity bindings (not just static part opacity), or the
    # synthesised lash line shows over the OPEN eye at rest. axis is ascending (closed=0 -> open=1).
    build = build_puppet(_eye_crossfade_rig())
    param = next(p for p in build.puppet["param"] if p["name"] == "ParamEyeLOpen")
    assert param["axis_points"][0] == [0.0, 1.0]

    uuid_name = {}
    for n in _walk_nodes(build.puppet):
        uuid_name[n["uuid"]] = n.get("name")
    opac = {uuid_name.get(b["node"]): [cell[0] for cell in b["values"]]
            for b in param["bindings"] if b["param_name"] == "opacity"}
    assert opac["eye"] == [0.0, 1.0]     # open eye: gone when closed, shown when open
    assert opac["lash"] == [1.0, 0.0]    # lash line: shown when closed, gone when open (invisible at rest)


def _walk_nodes(puppet):
    def walk(n):
        if isinstance(n, dict):
            if "uuid" in n:
                yield n
            for c in n.get("children", []) or []:
                yield from walk(c)
        elif isinstance(n, list):
            for c in n:
                yield from walk(c)
    yield from walk(puppet.get("nodes", puppet))
