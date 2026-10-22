"""Integrity controls for the selected scientific-evidence bundle."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest


@pytest.fixture
def reproduction(tmp_path, monkeypatch):
    driver = Path(__file__).resolve().parents[1] / "paper/reproduce.py"
    spec = importlib.util.spec_from_file_location("referee_reproduction_test_module", driver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "src/model.py"
    source.parent.mkdir()
    source.write_bytes(b"physical_model = 1\n")
    value = b'{"match":0.7}\n'
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(source_sha256={"src/model.py": hashlib.sha256(source.read_bytes()).hexdigest()},
        data_sha256={"outputs/reference/result.json": hashlib.sha256(value).hexdigest()}, scope="test")))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "MANIFEST", manifest)
    return module, tmp_path, value


def archive(path, members):
    with tarfile.open(path, "w:gz") as stream:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            stream.addfile(info, io.BytesIO(data))


def test_restore_verified_data_and_detect_source_change(reproduction):
    module, root, data = reproduction
    path = root / "data.tar.gz"
    archive(path, [("outputs/reference/result.json", data)])
    assert module.verify()["status"] == "failed"
    assert module.restore(path)["status"] == "passed"
    (root / "src/model.py").write_bytes(b"physical_model = 2\n")
    assert module.verify()["changed"] == ["src/model.py"]


def test_restore_never_overwrites_changed_evidence(reproduction):
    module, root, data = reproduction
    target = root / "outputs/reference/result.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"unrelated evidence")
    path = root / "data.tar.gz"
    archive(path, [("outputs/reference/result.json", data)])
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        module.restore(path)
    assert target.read_bytes() == b"unrelated evidence"


def test_archive_with_unlisted_file_is_rejected_before_writing(reproduction):
    module, root, data = reproduction
    path = root / "data.tar.gz"
    archive(path, [("outputs/reference/result.json", data), ("extra.json", b"{}")])
    with pytest.raises(ValueError, match="member set"):
        module.restore(path)
    assert not (root / "outputs").exists()
