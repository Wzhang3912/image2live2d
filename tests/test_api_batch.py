"""Phase 5 — public API (convert_*) + batch processor."""

from __future__ import annotations

import pytest

import image2live2d
from image2live2d import convert_layers
from image2live2d.batch import convert_batch, discover_inputs


def _make_layers(root, names):
    """Create N tiny sample layer dirs under root; return their paths."""
    from image2live2d.samples import make_sample_layers, make_sample_fullbody
    gens = {"portrait": make_sample_layers, "fullbody": make_sample_fullbody}
    out = []
    for n in names:
        gen = gens.get(n, make_sample_layers)
        out.append(gen(root / n))
    return out


# --------------------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------------------
def test_package_reexports_api():
    assert callable(image2live2d.convert_layers)
    assert callable(image2live2d.convert_psd)
    assert image2live2d.__version__ >= "0.2.0"


def test_convert_layers_writes_inp_and_qa(tmp_path):
    pytest.importorskip("PIL")
    layer_dir = _make_layers(tmp_path, ["portrait"])[0]
    result = convert_layers(layer_dir, tmp_path / "out")
    assert result.inp_path.is_file() and result.inp_path.suffix == ".inp"
    assert result.live2d_path is None
    assert result.passed
    assert result.qa.parts > 0


def test_convert_layers_with_live2d(tmp_path):
    pytest.importorskip("PIL")
    layer_dir = _make_layers(tmp_path, ["fullbody"])[0]
    result = convert_layers(layer_dir, tmp_path / "out", live2d=True)
    assert result.inp_path.is_file()
    assert result.live2d_path is not None and result.live2d_path.name.endswith(".model3.json")
    assert result.live2d_path.is_file()


# --------------------------------------------------------------------------------------------------
# batch
# --------------------------------------------------------------------------------------------------
def test_discover_inputs_finds_layer_dirs(tmp_path):
    pytest.importorskip("PIL")
    root = tmp_path / "roster"
    root.mkdir()
    _make_layers(root, ["a", "b"])
    found = discover_inputs(root)
    assert len(found) == 2
    assert {p.name for p in found} == {"a", "b"}


def test_discover_inputs_single_layer_dir(tmp_path):
    pytest.importorskip("PIL")
    layer_dir = _make_layers(tmp_path, ["solo"])[0]
    assert discover_inputs(layer_dir) == [layer_dir]


def test_convert_batch_aggregates_qa(tmp_path):
    pytest.importorskip("PIL")
    root = tmp_path / "roster"
    root.mkdir()
    _make_layers(root, ["portrait", "fullbody"])
    outcome = convert_batch(discover_inputs(root), tmp_path / "out")
    assert outcome.total == 2 and outcome.converted == 2
    assert not outcome.errors
    assert outcome.qa().pass_rate == 1.0
    # each produced an .inp
    for item in outcome.items:
        assert item.result.inp_path.is_file()


def test_convert_batch_reports_errors(tmp_path):
    # a directory that isn't a layer dir / psd -> discover returns nothing; an explicit bad path errors
    bad = tmp_path / "empty"
    bad.mkdir()
    outcome = convert_batch([bad], tmp_path / "out")
    assert outcome.total == 1 and outcome.converted == 0
    assert outcome.errors and outcome.errors[0].error
