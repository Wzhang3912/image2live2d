"""Local onboarding states and persistent model installation contract."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

from image2live2d.app import setup

spec = importlib.util.spec_from_file_location(
    "model_manager", Path(__file__).parents[1] / "service/seethrough/model_manager.py")
manager_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manager_module)


def test_unavailable_gpu_does_not_block_layers():
    with patch.object(setup, "companion", side_effect=OSError):
        result = setup.readiness()
    assert result["layers_ready"]
    assert not result["local"]["ready"]
    assert "./start.sh --gpu" in result["local"]["message"]


def test_ready_local_gpu():
    with patch.object(setup, "companion", return_value={"ready": True}):
        assert setup.readiness()["local"]["ready"]


def test_manifest_survives_restart_and_detects_missing_weights(tmp_path):
    manifest = {}
    for name in manager_module.REPOS:
        path = tmp_path / name
        path.mkdir()
        (path / 'weights.bin').write_bytes(b'model')
        manifest[name] = {"revision": "abc", "files": ["weights.bin"]}
    (tmp_path / 'installed.json').write_text(json.dumps(manifest))
    assert len(manager_module.ModelManager(tmp_path).paths()) == 2
    (tmp_path / 'depth/weights.bin').unlink()
    assert not manager_module.ModelManager(tmp_path).paths()


def test_partial_download_is_not_ready(tmp_path):
    (tmp_path / 'layerdiff').mkdir()
    assert not manager_module.ModelManager(tmp_path).paths()


def test_successful_download_records_revisions(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    def download(repo, revision, local_dir):
        assert revision == 'pinned-revision'
        local_dir.mkdir()
        (local_dir / 'model_index.json').write_text('{}')
        (local_dir / 'weights.bin').write_bytes(b'weights')
        return str(local_dir)

    hub = SimpleNamespace(
        HfApi=lambda: SimpleNamespace(model_info=lambda repo, **kwargs: SimpleNamespace(sha='pinned-revision')),
        snapshot_download=download)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', hub)
    monkeypatch.setattr(manager_module.shutil, 'disk_usage',
                        lambda _: SimpleNamespace(free=30 * 2**30))
    manager = manager_module.ModelManager(tmp_path)
    manager.lock.acquire()
    manager._download()
    assert manager.state == 'ready'
    assert not manager.lock.locked()
    assert len(manager_module.ModelManager(tmp_path).paths()) == 2


def test_failed_download_can_retry(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    def fail(*args, **kwargs):
        raise OSError('network unavailable')

    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        HfApi=lambda: SimpleNamespace(model_info=fail), snapshot_download=fail))
    monkeypatch.setattr(manager_module.shutil, 'disk_usage',
                        lambda _: SimpleNamespace(free=30 * 2**30))
    manager = manager_module.ModelManager(tmp_path)
    manager.lock.acquire()
    manager._download()
    assert manager.state == 'error'
    assert not manager.paths()
    assert not manager.lock.locked()


def test_setup_post_rejects_cross_origin_without_contacting_gpu():
    from email.message import Message
    from image2live2d.app.server import _make_handler
    handler = object.__new__(_make_handler())
    handler.path = '/api/setup/models'
    handler.headers = Message()
    handler.headers['Host'] = 'localhost:8000'
    handler.headers['Origin'] = 'https://unrelated.example'
    results = []
    handler._json = lambda code, value: results.append((code, value))
    with patch.object(setup, 'companion') as companion:
        handler.do_POST()
    assert results[0][0] == 403
    companion.assert_not_called()


def test_setup_post_starts_local_download():
    from email.message import Message
    from image2live2d.app.server import _make_handler
    handler = object.__new__(_make_handler())
    handler.path = '/api/setup/models'
    handler.headers = Message()
    handler.headers['Host'] = 'localhost:8000'
    handler.headers['Origin'] = 'http://localhost:8000'
    results = []
    handler._json = lambda code, value: results.append((code, value))
    with patch.object(setup, 'companion', return_value={'state': 'downloading'}) as companion:
        handler.do_POST()
    assert results == [(200, {'state': 'downloading'})]
    companion.assert_called_once_with(action=True)
