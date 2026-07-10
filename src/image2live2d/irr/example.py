"""Build a minimal but valid example ``Rig``.

This is the Phase-0 smoke test: it proves the IRR schema, validators, and standard-parameter
catalog hang together, and gives emitters/tests a known-good fixture to consume. The example is a
trivial two-part face (a base + a mouth) with a single ``ParamMouthOpenY`` parameter that opens
the mouth.

Run directly::

    python -m image2live2d.irr.example
"""

from __future__ import annotations

from .params import make_parameter
from .schema import (
    Keyform,
    Mesh,
    Meta,
    Part,
    Rig,
    SemanticRole,
    Texture,
)


def _quad_mesh(part_id: str) -> Mesh:
    """A unit quad (2 triangles) centered at origin."""
    return Mesh(
        part_id=part_id,
        vertices=[(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)],
        uvs=[(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)],
        triangles=[(0, 1, 2), (0, 2, 3)],
    )


def build_example_rig() -> Rig:
    """Construct and return a minimal valid rig (raises on any integrity error)."""
    textures = [
        Texture(id="tex_face", path="textures/face.png", width=512, height=512),
        Texture(id="tex_mouth", path="textures/mouth.png", width=128, height=64),
    ]
    parts = [
        Part(id="face_base", semantic_role=SemanticRole.face_base, texture_id="tex_face", draw_order=0),
        Part(id="mouth", semantic_role=SemanticRole.mouth, texture_id="tex_mouth", draw_order=10),
    ]
    meshes = [_quad_mesh("face_base"), _quad_mesh("mouth")]

    # ParamMouthOpenY: at 0 the mouth is at rest; at 1 the lower two vertices drop (mouth opens).
    mouth_open = make_parameter("ParamMouthOpenY")
    mouth_open.keyforms = [
        Keyform(value=0.0, mesh_offsets={"mouth": [(0.0, 0.0)] * 4}),
        Keyform(
            value=1.0,
            mesh_offsets={"mouth": [(0.0, -0.15), (0.0, -0.15), (0.0, 0.0), (0.0, 0.0)]},
        ),
    ]

    return Rig(
        meta=Meta(name="example", archetype="portrait_front"),
        textures=textures,
        parts=parts,
        meshes=meshes,
        parameters=[mouth_open],
    )


def main() -> None:
    from .validate import format_issues, lint

    rig = build_example_rig()
    print(rig.model_dump_json(indent=2))
    print("\n--- lint ---")
    print(format_issues(lint(rig)))


if __name__ == "__main__":
    main()
