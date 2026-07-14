"""Phase 1 spine tests: decompose seam -> mesh -> author_rig -> assemble -> QA, plus an
end-to-end synthetic image dir -> loadable .inp (Pillow-gated)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from image2live2d.backends.nijilive import NijiliveEmitter
from image2live2d.backends.nijilive.inp import InpFile
from image2live2d.backends.nijilive.puppet import solid_png
from image2live2d.core import assemble, decompose
from image2live2d.core.mesh import grid_mesh
from image2live2d.core.qa import deform_at, sweep_report
from image2live2d.core.rig import author_rig, select_template
from image2live2d.core.types import Layer, LayerStack
from image2live2d.irr.schema import Mesh, SemanticRole
from image2live2d.irr.validate import Severity, lint

_HAS_PIL = importlib.util.find_spec("PIL") is not None


# --- fixtures (no ML / Pillow) --------------------------------------------------------------------
def _opaque(_u: float, _v: float) -> int:
    return 255


def _part(part_id: str, role: SemanticRole, order: int, rect) -> tuple[Layer, Mesh]:
    layer = Layer(
        id=part_id,
        semantic_role=role,
        texture_path=Path(f"{part_id}.png"),
        draw_order=order,
        width=64,
        height=64,
    )
    mesh = grid_mesh(part_id, rect, _opaque, grid=2)
    return layer, mesh


def _face_fixture() -> tuple[LayerStack, list[Mesh]]:
    # rects are model-space (y up), distinct so deforms are non-trivial
    spec = [
        ("face_base", SemanticRole.face_base, 0, (0.2, 0.1, 0.8, 0.9)),
        ("hair_front", SemanticRole.hair_front, 90, (0.2, 0.6, 0.8, 0.95)),
        ("eyebrow_l", SemanticRole.eyebrow_l, 40, (0.30, 0.66, 0.45, 0.70)),
        ("eyebrow_r", SemanticRole.eyebrow_r, 40, (0.55, 0.66, 0.70, 0.70)),
        ("eye_white_l", SemanticRole.eye_white_l, 50, (0.30, 0.55, 0.45, 0.63)),
        ("eye_white_r", SemanticRole.eye_white_r, 50, (0.55, 0.55, 0.70, 0.63)),
        ("eye_l", SemanticRole.eye_l, 51, (0.30, 0.55, 0.45, 0.63)),
        ("eye_r", SemanticRole.eye_r, 52, (0.55, 0.55, 0.70, 0.63)),
        ("pupil_l", SemanticRole.pupil_l, 60, (0.35, 0.56, 0.41, 0.62)),
        ("pupil_r", SemanticRole.pupil_r, 61, (0.59, 0.56, 0.65, 0.62)),
        ("nose", SemanticRole.nose, 55, (0.47, 0.45, 0.53, 0.52)),
        ("mouth", SemanticRole.mouth, 70, (0.42, 0.30, 0.58, 0.38)),
    ]
    layers = []
    meshes = []
    for pid, role, order, rect in spec:
        layer, mesh = _part(pid, role, order, rect)
        layers.append(layer)
        meshes.append(mesh)
    return LayerStack(layers=layers, canvas_width=64, canvas_height=64), meshes


def _authored_rig():
    stack, meshes = _face_fixture()
    template = select_template(stack)
    authoring = author_rig(stack, meshes, template)
    return assemble.assemble_rig(
        name="fix",
        source=None,
        stack=stack,
        meshes=meshes,
        deformers=authoring.deformers,
        parameters=authoring.parameters,
        physics=[],
        archetype=template.name,
    )


# --- assemble -------------------------------------------------------------------------------------
def test_assemble_binds_parts_and_textures():
    stack, _ = _face_fixture()
    textures = assemble.textures_for(stack)
    parts = assemble.parts_for(stack)
    assert len(textures) == len(parts) == len(stack.layers)
    assert {p.texture_id for p in parts} == {t.id for t in textures}
    # 1:1 id derivation
    assert assemble.texture_id_for("mouth") == "tex_mouth"


# --- decompose seam -------------------------------------------------------------------------------
def test_from_layer_dir_reads_roles_and_order(tmp_path):
    (tmp_path / "20_mouth.png").write_bytes(solid_png(40, 20))
    (tmp_path / "00_face_base.png").write_bytes(solid_png(64, 64))
    (tmp_path / "10_eye_l.png").write_bytes(solid_png(16, 8))
    stack = decompose.from_layer_dir(tmp_path)
    assert [lyr.draw_order for lyr in stack.layers] == [0, 10, 20]  # sorted by order
    assert [lyr.semantic_role for lyr in stack.layers] == [
        SemanticRole.face_base, SemanticRole.eye_l, SemanticRole.mouth
    ]
    assert stack.canvas_width == 64 and stack.canvas_height == 64
    # png_size reads IHDR directly
    assert decompose.png_size(tmp_path / "20_mouth.png") == (40, 20)


def test_from_layer_dir_rejects_bad_name(tmp_path):
    (tmp_path / "facebase.png").write_bytes(solid_png(8, 8))
    with pytest.raises(ValueError):
        decompose.from_layer_dir(tmp_path)


def test_from_layer_dir_rejects_unknown_role(tmp_path):
    (tmp_path / "00_wing.png").write_bytes(solid_png(8, 8))
    with pytest.raises(ValueError):
        decompose.from_layer_dir(tmp_path)


# --- author_rig -----------------------------------------------------------------------------------
def test_author_rig_creates_expected_params():
    rig = _authored_rig()
    ids = rig.parameter_ids()
    for expected in [
        "ParamEyeLOpen", "ParamEyeROpen", "ParamMouthOpenY", "ParamMouthForm",
        "ParamAngleX", "ParamAngleY", "ParamAngleZ",
        "ParamEyeBallX", "ParamEyeBallY", "ParamBrowLY", "ParamBrowRY", "ParamBreath",
    ]:
        assert expected in ids, expected


def test_blink_collapses_eye_at_closed():
    rig = _authored_rig()
    closed = deform_at(rig, "ParamEyeLOpen", 0.0)["eye_l"]
    open_ = deform_at(rig, "ParamEyeLOpen", 1.0)["eye_l"]
    rest = rig.mesh_for("eye_l").vertices
    # open == rest (zero offset); closed collapses toward the eye's vertical centre
    assert open_ == [tuple(v) for v in rest]
    assert closed != [tuple(v) for v in rest]
    ys_closed = [y for _, y in closed]
    assert max(ys_closed) - min(ys_closed) < (max(y for _, y in rest) - min(y for _, y in rest))


def test_mouth_open_is_a_lens_cavity():
    rig = _authored_rig()
    opened = deform_at(rig, "ParamMouthOpenY", 1.0)["mouth"]
    rest = rig.mesh_for("mouth").vertices
    cx = (min(x for x, _ in rest) + max(x for x, _ in rest)) / 2.0
    cy = (min(y for _, y in rest) + max(y for _, y in rest)) / 2.0
    deltas = [oy - ry for (_, oy), (_, ry) in zip(opened, rest)]
    # lens open: the lower lip drops and the upper lip rises a smaller amount (jaw does most of it).
    assert min(deltas) < 0.0                                   # lower lip drops
    assert max(deltas) > 0.0                                   # upper lip rises (not a flat jaw-slide)
    assert abs(min(deltas)) > abs(max(deltas))                 # lower drop dominates the upper rise
    # the corners are anchored: a centre vertex moves more than a corner vertex at the same height band.
    lower = [i for i, (_, y) in enumerate(rest) if y < cy]
    corner = max(lower, key=lambda i: abs(rest[i][0] - cx))
    centre = min(lower, key=lambda i: abs(rest[i][0] - cx))
    assert abs(deltas[centre]) > abs(deltas[corner])


def test_head_turn_is_a_coherent_warp():
    rig = _authored_rig()
    right = deform_at(rig, "ParamAngleX", 30.0)
    rest = rig.mesh_for("face_base").vertices
    dxs = [nx - rx for (nx, _), (rx, _) in zip(right["face_base"], rest)]
    # a true warp varies per vertex across the part (a uniform translate would give identical dx)
    assert max(dxs) - min(dxs) > 1e-3
    # rest pose at value 0 is identity
    at_rest = deform_at(rig, "ParamAngleX", 0.0)
    assert at_rest["face_base"] == [tuple(v) for v in rest]


def test_mouth_form_raises_corners_on_smile():
    rig = _authored_rig()
    rest = rig.mesh_for("mouth").vertices
    cx = (min(x for x, _ in rest) + max(x for x, _ in rest)) / 2.0
    corner = max(range(len(rest)), key=lambda i: abs(rest[i][0] - cx))  # furthest from centre
    smile = deform_at(rig, "ParamMouthForm", 1.0)["mouth"]
    frown = deform_at(rig, "ParamMouthForm", -1.0)["mouth"]
    # corner goes up at smile, down at frown (IRR y-up)
    assert smile[corner][1] > rest[corner][1]
    assert frown[corner][1] < rest[corner][1]


def test_breath_bobs_every_part_up():
    rig = _authored_rig()
    inhale = deform_at(rig, "ParamBreath", 1.0)
    assert set(inhale) == rig.part_ids()  # affects all parts
    for pid, positions in inhale.items():
        rest = rig.mesh_for(pid).vertices
        assert all(ny > ry for (_, ny), (_, ry) in zip(positions, rest))


def test_authored_rig_is_clean_and_passes_sweep():
    rig = _authored_rig()
    issues = lint(rig)
    assert not [i for i in issues if i.severity is Severity.warning], issues
    report = sweep_report(rig)
    assert report.frames > 0
    assert report.passed, report.issues


# --- end-to-end (Pillow-gated) --------------------------------------------------------------------
@pytest.mark.skipif(not _HAS_PIL, reason="Pillow not installed")
def test_end_to_end_synthetic_dir_to_inp(tmp_path):
    from PIL import Image
    from image2live2d.pipeline import rig_from_stack

    layer_dir = tmp_path / "layers"
    layer_dir.mkdir()
    # full-canvas RGBA layers with a centred opaque blob per part
    for name in ("00_face_base", "10_eye_l", "11_eye_r", "20_mouth"):
        img = Image.new("RGBA", (64, 64), (255, 0, 0, 0))
        for x in range(20, 44):
            for y in range(20, 44):
                img.putpixel((x, y), (200, 120, 120, 255))
        img.save(layer_dir / f"{name}.png")

    stack = decompose.from_layer_dir(layer_dir)
    rig = rig_from_stack(stack, name="synthetic", source="synthetic")
    # no synthesised cavity here: this stub mouth is a solid 40x20 block, not a thin lip line, so
    # core.synth reads it as a mouth already drawn open and leaves the artist's own interior alone
    assert rig.part_ids() == {"00_face_base", "10_eye_l", "11_eye_r", "20_mouth"}
    assert "ParamMouthOpenY" in rig.parameter_ids()

    out = NijiliveEmitter(asset_root=layer_dir).emit(rig, tmp_path)
    inp = InpFile.read(out)
    puppet = json.loads(inp.payload)
    assert len(inp.textures) == 4
    assert all(t.data[:8] == b"\x89PNG\r\n\x1a\n" for t in inp.textures)
    def _part_names(node):  # parts may be nested under the 'head' group node now
        acc = {node["name"]} if node.get("type") == "Part" else set()
        for c in node.get("children", []):
            acc |= _part_names(c)
        return acc
    assert _part_names(puppet["nodes"]) == rig.part_ids()
    assert {p["name"] for p in puppet["param"]} >= {"ParamMouthOpenY"}
