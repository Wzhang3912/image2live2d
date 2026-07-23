"""Unit tests for the vendored See-through missing-tag-PNG guard patcher.

The patcher makes ``load_parts`` skip a tag whose ``.png`` is absent instead of crashing ``load_part``'s
bare ``Image.open``. Uses a synthetic fixture mirroring the upstream loop — we do not vendor See-through
source — and asserts the rewrite is valid, correctly scoped inside the loop, idempotent, and non-fatal.
"""
import ast
import importlib.util
from pathlib import Path

_PATCHER = (
    Path(__file__).resolve().parents[1] / "service/seethrough/patches/guard_missing_part_pngs.py"
)
_spec = importlib.util.spec_from_file_location("guard_missing_part_pngs", _PATCHER)
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)


# Mirrors load_parts' tag loop: the guard's `continue` must land inside the `for` loop, and the
# existence check must precede the load_part call.
FIXTURE = '''\
def load_parts(srcp, rotate=False, pad=0, min_width=64):
    tag2pd = {}
    part_id = 0
    min_sz = 12
    for tag, partdict in infos['parts'].items():
        p = load_part(osp.join(srcp, tag + '.png'), rotate=rotate, pad=pad, min_width=min_width, min_sz=min_sz)
        if p is not None:
            tag2pd[tag] = p
            part_id += 1
    return fullpage, infos, part_dict_list
'''


def test_patch_inserts_existence_check_before_load_part_and_stays_valid():
    out, status = guard.patch_text(FIXTURE)
    assert status == "patched"
    ast.parse(out)  # also proves `continue` is inside the loop (else SyntaxError)
    assert guard.MARKER in out
    lines = out.splitlines()
    check_i = next(i for i, ln in enumerate(lines) if ln.strip().startswith("if not osp.exists"))
    load_i = next(i for i, ln in enumerate(lines) if ln.strip().startswith("p = load_part"))
    assert check_i < load_i
    # same indentation as the load_part call (i.e. same loop-body level)
    assert (len(lines[check_i]) - len(lines[check_i].lstrip())) == (
        len(lines[load_i]) - len(lines[load_i].lstrip())
    )


def test_the_guard_continue_sits_inside_the_for_loop():
    out, _ = guard.patch_text(FIXTURE)
    tree = ast.parse(out)
    # find the for-loop and assert a Continue node exists within it
    fors = [n for n in ast.walk(tree) if isinstance(n, ast.For)]
    assert fors, "expected a for-loop"
    assert any(isinstance(n, ast.Continue) for n in ast.walk(fors[0]))


def test_patch_is_idempotent():
    once, s1 = guard.patch_text(FIXTURE)
    twice, s2 = guard.patch_text(once)
    assert s1 == "patched"
    assert s2 == "already-patched"
    assert twice == once


def test_missing_anchor_is_non_fatal():
    unrelated = "def f():\n    return 1\n"
    out, status = guard.patch_text(unrelated)
    assert status == "anchor-not-found"
    assert out == unrelated


def test_main_patches_a_file_and_is_a_noop_second_time(tmp_path):
    p = tmp_path / "io.py"
    p.write_text(FIXTURE)
    assert guard.main(str(p)) == 0
    assert guard.MARKER in p.read_text()
    ast.parse(p.read_text())
    assert guard.main(str(p)) == 0


def test_main_returns_nonzero_when_anchor_absent(tmp_path):
    p = tmp_path / "nothing.py"
    p.write_text("x = 1\n")
    assert guard.main(str(p)) == 3
    assert p.read_text() == "x = 1\n"
