"""Unit tests for the vendored See-through empty-head-crop guard patcher.

The patcher rewrites See-through's ``apply_layerdiff`` so a degenerate head crop (empty ``'head'``
segment, seen on non-human faces) skips head refinement instead of crashing ``cv2.resize``. These
tests use a synthetic fixture that mirrors the upstream code shape — we do not vendor See-through's
source — and assert the rewrite is valid, correctly scoped, idempotent, and non-fatal on drift.
"""
import ast
import importlib.util
from pathlib import Path

_PATCHER = (
    Path(__file__).resolve().parents[1] / "service/seethrough/patches/guard_empty_head_crop.py"
)
_spec = importlib.util.spec_from_file_location("guard_empty_head_crop", _PATCHER)
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)


# A synthetic stand-in for the v3 head-refine block of apply_layerdiff: the crash line (ANCHOR)
# followed by the head pipeline, all inside an ``elif`` whose ``else: raise`` must stay put.
FIXTURE = '''\
def apply_layerdiff(pipeline, resolution):
    tag_version = pipeline.tag_version
    if tag_version == 'v2':
        legacy()
    elif tag_version == 'v3':
        head_img = images[2]
        hx0, hy0, hw, hh = cv2.boundingRect(mask)
        input_head, box = _crop_head(input_img, [hx, hy, hw, hh])
        ih, iw = input_head.shape[:2]
        input_head, pad_size, pad_pos = center_square_pad_resize(input_head, resolution, return_pad_info=True)

        Image.fromarray(input_head).save(head_path)

        for rst, tag in zip(outputs, head_tag_list):
            full = canvas.copy()
            Image.fromarray(full).save(tag_path)

    else:
        raise
    return done
'''


def test_patch_produces_valid_python_and_guards_the_crash_line():
    out, status = guard.patch_text(FIXTURE)
    assert status == "patched"
    ast.parse(out)  # must stay syntactically valid
    assert guard.MARKER in out
    # The crash line is now nested under the degeneracy guard, not run unconditionally.
    lines = out.splitlines()
    guard_i = next(i for i, ln in enumerate(lines) if ln.strip() == "if ih == 0 or iw == 0:")
    crash_i = next(i for i, ln in enumerate(lines) if ln.strip().startswith("input_head, pad_size"))
    assert crash_i > guard_i
    # The crash line is indented deeper than the guard (i.e. inside the else branch).
    guard_indent = len(lines[guard_i]) - len(lines[guard_i].lstrip())
    crash_indent = len(lines[crash_i]) - len(lines[crash_i].lstrip())
    assert crash_indent > guard_indent


def test_guard_reads_ih_iw_that_are_defined_before_it():
    # The guard condition uses ih/iw, which must be assigned *outside* (before) the guard block.
    out, _ = guard.patch_text(FIXTURE)
    lines = out.splitlines()
    assign_i = next(i for i, ln in enumerate(lines) if ln.strip() == "ih, iw = input_head.shape[:2]")
    guard_i = next(i for i, ln in enumerate(lines) if ln.strip() == "if ih == 0 or iw == 0:")
    assert assign_i < guard_i


def test_the_enclosing_else_raise_is_preserved():
    out, _ = guard.patch_text(FIXTURE)
    # The elif/else structure of apply_layerdiff must survive the re-indentation intact.
    assert "\n    else:\n        raise\n" in out


def test_patch_is_idempotent():
    once, s1 = guard.patch_text(FIXTURE)
    twice, s2 = guard.patch_text(once)
    assert s1 == "patched"
    assert s2 == "already-patched"
    assert twice == once


def test_missing_anchor_is_non_fatal_and_leaves_text_untouched():
    unrelated = "def f():\n    return 1\n"
    out, status = guard.patch_text(unrelated)
    assert status == "anchor-not-found"
    assert out == unrelated


def test_main_returns_nonzero_when_anchor_absent(tmp_path):
    p = tmp_path / "nothing.py"
    p.write_text("x = 1\n")
    assert guard.main(str(p)) == 3
    assert p.read_text() == "x = 1\n"  # untouched


def test_main_patches_a_file_in_place(tmp_path):
    p = tmp_path / "iu.py"
    p.write_text(FIXTURE)
    assert guard.main(str(p)) == 0
    patched = p.read_text()
    assert guard.MARKER in patched
    ast.parse(patched)
    # second application is a clean no-op
    assert guard.main(str(p)) == 0
