from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def _load_archive_module():
    module_path = ROOT / "packaging" / "create_archive.py"
    spec = importlib.util.spec_from_file_location(
        "module_to_flashcards_create_archive", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_create_archive_uses_zip64_and_keeps_bundle_directory(tmp_path, monkeypatch):
    archive_module = _load_archive_module()
    bundle = tmp_path / "ModuleToFlashcards"
    model = bundle / "models" / "model.gguf"
    model.parent.mkdir(parents=True)
    payload = b"portable-model" * 32
    model.write_bytes(payload)
    destination = tmp_path / "release.zip"

    monkeypatch.setattr(archive_module.zipfile, "ZIP64_LIMIT", 128)

    result = archive_module.create_archive(bundle, destination)

    assert result == destination.resolve()
    with zipfile.ZipFile(destination) as archive:
        member = "ModuleToFlashcards/models/model.gguf"
        assert archive.namelist() == [member]
        assert archive.getinfo(member).file_size == len(payload)
        assert archive.read(member) == payload
