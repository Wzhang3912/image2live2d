"""Regression gate over the decomposed-character corpus.

Every threshold in the repair layer is calibrated on real characters, and for a long time all of them
were GPT-generated, 1280x1280, standing square to the camera with their arms at their sides. Three
separate rules turned out to encode that pose (see `core.structure.limbs`), and a fourth measured the
mouth against a face_base that only covered a forehead (see `core.qa.harness`). None of those were
visible from a unit test, because a unit test uses the geometry we thought was typical.

So the corpus is the gate. `tests/data/character_corpus.json` records, per character, the capabilities
the rig must still have, the plausibility codes it must still raise, and floors for part and physics
counts. A change that silently costs a character its arms fails here even when every unit test passes.

The images themselves are NOT in the repo -- it is public and code-only, and the hand-drawn half is
CC BY-SA. Point `I2L_CHARACTER_CORPUS` at a directory of `<name>/` layer folders (or drop them in
`out/corpus/`) and these run; otherwise they skip, like the .moc3 tests that need proprietary samples.
Regenerate the expectations after an intended improvement -- the floors are floors, so gaining parts
passes and losing them does not.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

_EXPECTATIONS = Path(__file__).parent / "data" / "character_corpus.json"
_DEFAULT_ROOT = Path(__file__).parent.parent / "out" / "corpus"


def _corpus_root() -> Path | None:
    root = Path(os.environ.get("I2L_CHARACTER_CORPUS", _DEFAULT_ROOT))
    return root if root.is_dir() else None


def _expected() -> dict:
    return json.loads(_EXPECTATIONS.read_text())


def _cases() -> list[str]:
    """Characters that are both expected and actually present locally."""
    root = _corpus_root()
    if root is None:
        return []
    return sorted(name for name in _expected() if (root / name).is_dir())


@pytest.fixture(scope="module")
def rigs() -> dict:
    pytest.importorskip("PIL")
    from image2live2d.core import decompose
    from image2live2d.pipeline import rig_from_stack

    root = _corpus_root()
    return {name: rig_from_stack(decompose.from_layer_dir(root / name), name=name)
            for name in _cases()}


def test_the_expectations_file_is_well_formed():
    """Runs with or without the corpus, so a malformed table can't hide behind a skip."""
    for name, want in _expected().items():
        assert set(want) == {"capabilities", "plausibility", "min_parts", "min_physics_chains"}, name
        assert want["min_parts"] > 0 and want["min_physics_chains"] >= 0, name


@pytest.mark.skipif(not _cases(), reason="no decomposed character corpus (set I2L_CHARACTER_CORPUS)")
@pytest.mark.parametrize("name", _cases())
def test_a_character_keeps_the_capabilities_it_had(name, rigs):
    """The capability set is the contract: this is what the puppet can actually *do*. Losing one is the
    failure that motivated the corpus -- `wikipetan_mop` shipped with no arms at all because three pose
    assumptions rejected them, and nothing in the suite noticed."""
    from image2live2d.core.qa.capability import rig_capabilities

    want = _expected()[name]["capabilities"]
    rep = rig_capabilities(rigs[name])
    assert [c for c in want if not rep.has(c)] == []


@pytest.mark.skipif(not _cases(), reason="no decomposed character corpus (set I2L_CHARACTER_CORPUS)")
@pytest.mark.parametrize("name", _cases())
def test_a_character_raises_the_plausibility_codes_it_should(name, rigs):
    """Exact, in both directions. A new warning on a character we consider correct is a false positive,
    and a vanished one means a real defect stopped being reported -- `face_base_incomplete` on
    `wikipetan_stand` is the only entry, and it must neither spread nor disappear."""
    from image2live2d.core.qa.harness import plausibility_issues

    want = set(_expected()[name]["plausibility"])
    assert {i.code for i in plausibility_issues(rigs[name])} == want


@pytest.mark.skipif(not _cases(), reason="no decomposed character corpus (set I2L_CHARACTER_CORPUS)")
@pytest.mark.parametrize("name", _cases())
def test_a_character_does_not_lose_parts_or_physics(name, rigs):
    """Floors, not equalities: splitting a bundled pair or finding another hair strand is an improvement
    and must pass. A drop means something stopped being separated."""
    from image2live2d.core.qa.capability import rig_capabilities

    want = _expected()[name]
    rep = rig_capabilities(rigs[name])
    assert rep.parts >= want["min_parts"]
    assert rep.physics_chains >= want["min_physics_chains"]


@pytest.mark.skipif(not _cases(), reason="no decomposed character corpus (set I2L_CHARACTER_CORPUS)")
@pytest.mark.parametrize("name", _cases())
def test_every_limb_pivot_sits_inside_the_limb_it_rotates(name, rigs):
    """Backlog T10: limb joints were the top-centre of the limb's bbox, which put 16 of 34 shoulders
    *outside* the limb -- a bent arm's bbox centre is in the air beside it. Now they follow the limb's
    own silhouette axis. Held on all six hand-drawn characters, so it is an invariant, not a fit."""
    from image2live2d.core.rig.author import _limb_joints, _union_bbox
    from image2live2d.irr.schema import SemanticRole as R

    limbs = (R.arm_l, R.arm_r, R.leg_l, R.leg_r)
    rig = rigs[name]
    outside = []
    for part in rig.parts:
        if part.semantic_role not in limbs:
            continue
        meshes = [m for m in rig.meshes if m.part_id == part.id]
        if not meshes:
            continue
        shoulder, _, _ = _limb_joints(meshes)
        _, y0, _, y1 = _union_bbox(meshes)
        # sample just inside the limb, below the joint, so a pivot exactly on the top edge counts
        probe = (shoulder[0], shoulder[1] - 0.02 * (y1 - y0))
        if not _point_in_meshes(meshes, probe):
            outside.append(part.id)
    assert outside == []


def _point_in_meshes(meshes, point) -> bool:
    px, py = point
    for mesh in meshes:
        v = mesh.vertices
        for a, b, c in mesh.triangles:
            (ax, ay), (bx, by), (cx, cy) = v[a], v[b], v[c]
            d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by)
            d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy)
            d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay)
            if not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0)):
                return True
    return False
