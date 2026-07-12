"""P1 — RigGraph assembly (core.structure.graph). Pure tests for kinematic parenting + anchors.

``build_rig_graph`` reproduces the rig's prior head/body/accessory split from meshes alone (no alpha),
so it drops into ``author_rig`` with no output change. These tests pin the parenting rules directly;
the byte-identical guarantee for the whole pipeline is enforced by the existing golden suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from image2live2d.core.structure import (
    BODY, HEAD, DynamicsVerdict, RigGraph, build_rig_graph,
)
from image2live2d.core.types import Layer, LayerStack
from image2live2d.irr.schema import Mesh
from image2live2d.irr.schema import SemanticRole as R


def _mesh(pid: str, x0: float, y0: float, x1: float, y1: float) -> Mesh:
    verts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return Mesh(part_id=pid, vertices=verts, uvs=[(0.0, 0.0)] * 4, triangles=[(0, 1, 2), (0, 2, 3)])


def _scene(parts):
    """parts: list of (id, role, (x0, y0, x1, y1)) in model space (y up)."""
    layers, meshes = [], []
    for i, (pid, role, box) in enumerate(parts):
        layers.append(Layer(id=pid, semantic_role=role, texture_path=Path(f"{pid}.png"),
                            draw_order=i * 10, width=64, height=64))
        meshes.append(_mesh(pid, *box))
    return LayerStack(layers=layers, canvas_width=64, canvas_height=64), meshes


def _graph(parts) -> RigGraph:
    stack, meshes = _scene(parts)
    return build_rig_graph(stack, meshes)


def test_head_and_body_roles_ride_their_group():
    g = _graph([
        ("face", R.face_base, (0.3, 0.6, 0.7, 0.95)),
        ("hair", R.hair_front, (0.28, 0.6, 0.72, 0.98)),
        ("torso", R.torso, (0.35, 0.1, 0.65, 0.6)),
        ("skirt", R.clothing, (0.3, 0.0, 0.7, 0.2)),
    ])
    assert g.parent_of("face") == HEAD
    assert g.parent_of("hair") == HEAD
    assert g.parent_of("torso") == BODY
    assert g.parent_of("skirt") == BODY


def test_accessory_binds_to_nearest_group():
    parts = [
        ("face", R.face_base, (0.30, 0.60, 0.70, 0.95)),
        ("torso", R.torso, (0.35, 0.10, 0.65, 0.55)),
        ("bow", R.accessory, (0.42, 0.66, 0.58, 0.78)),   # up by the head
        ("belt", R.accessory, (0.40, 0.30, 0.60, 0.38)),  # down on the body
    ]
    g = _graph(parts)
    assert g.parent_of("bow") == HEAD
    assert g.parent_of("belt") == BODY


def test_accessory_falls_to_the_only_group_present():
    head_only = _graph([
        ("face", R.face_base, (0.3, 0.6, 0.7, 0.95)),
        ("charm", R.accessory, (0.4, 0.1, 0.6, 0.2)),     # low, but no body exists
    ])
    assert head_only.parent_of("charm") == HEAD

    body_only = _graph([
        ("torso", R.torso, (0.35, 0.1, 0.65, 0.6)),
        ("pin", R.accessory, (0.4, 0.9, 0.6, 0.98)),      # high, but no head exists
    ])
    assert body_only.parent_of("pin") == BODY


def test_background_has_no_parent():
    g = _graph([("bg", R.background, (0.0, 0.0, 1.0, 1.0)),
                ("face", R.face_base, (0.3, 0.6, 0.7, 0.9))])
    assert g.parent_of("bg") is None


def test_anchor_is_top_centre_and_nodes_are_stack_ordered():
    g = _graph([
        ("face", R.face_base, (0.20, 0.30, 0.60, 0.80)),
        ("torso", R.torso, (0.35, 0.10, 0.65, 0.55)),
    ])
    face = g.node("face")
    assert face.anchor == (0.40, 0.80)                    # (centre-x, top-y), y up
    assert [n.part_id for n in g.nodes] == ["face", "torso"]
    assert {n.part_id for n in g.children(HEAD)} == {"face"}


def test_parts_without_a_mesh_are_skipped():
    stack, meshes = _scene([("face", R.face_base, (0.3, 0.6, 0.7, 0.9)),
                            ("ghost", R.accessory, (0.4, 0.4, 0.6, 0.6))])
    meshes = [m for m in meshes if m.part_id != "ghost"]   # drop ghost's mesh
    g = build_rig_graph(stack, meshes)
    assert g.node("ghost") is None
    assert g.node("face") is not None


def test_analyze_structure_attaches_dynamics(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    # two full-canvas PNGs: a face block and a thin side strand hanging below it
    def png(path, box):
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        px = img.load()
        x0, y0, x1, y1 = box
        for y in range(y0, y1):
            for x in range(x0, x1):
                px[x, y] = (255, 0, 0, 255)
        img.save(path)

    from image2live2d.core.structure import analyze_structure

    fp = tmp_path / "face.png"
    sp = tmp_path / "side.png"
    png(fp, (16, 4, 48, 26))     # face near the top (v down in pixels)
    png(sp, (18, 8, 24, 52))     # a slender strand hanging below
    stack = LayerStack(layers=[
        Layer(id="face", semantic_role=R.face_base, texture_path=fp, draw_order=0, width=64, height=64),
        Layer(id="side", semantic_role=R.hair_side, texture_path=sp, draw_order=1, width=64, height=64),
    ], canvas_width=64, canvas_height=64)
    # model space (y up): face high, strand hanging down from it
    meshes = [_mesh("face", 0.25, 0.6, 0.75, 0.94), _mesh("side", 0.28, 0.18, 0.38, 0.88)]

    g = analyze_structure(stack, meshes, samples=48)
    assert g.node("face").verdict is not None
    assert g.node("side").verdict in (DynamicsVerdict.gentle, DynamicsVerdict.dynamic)
