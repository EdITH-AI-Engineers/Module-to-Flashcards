from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from portable_manifest import verify_manifest


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare_assets_module = _load_script(
    "module_to_flashcards_prepare_assets", "packaging/prepare_assets.py"
)
assemble_module = _load_script(
    "module_to_flashcards_assemble_portable", "packaging/assemble_portable.py"
)


def _valid_lock() -> dict:
    return json.loads((ROOT / "packaging" / "model-lock.json").read_text(encoding="utf-8"))


def test_prepare_rejects_moving_rebel_revision():
    lock = _valid_lock()
    lock["rebel"]["revision"] = "main"

    with pytest.raises(ValueError, match="immutable REBEL revision"):
        prepare_assets_module.validate_model_lock(lock)


def test_prepare_rejects_changed_qwen_digest():
    lock = _valid_lock()
    lock["qwen"]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="locked Qwen digest"):
        prepare_assets_module.validate_model_lock(lock)


def _fake_frozen_app(repo: Path) -> Path:
    frozen = repo / "build" / "frozen-dist" / "ModuleToFlashcards"
    (frozen / "runtime").mkdir(parents=True)
    (frozen / "ModuleToFlashcards.exe").write_bytes(b"exe")
    (frozen / "runtime" / "python.dll").write_bytes(b"runtime")
    return frozen


def _fake_assets(repo: Path) -> Path:
    assets = repo / "packaging" / "assets"
    (assets / "models" / "rebel-large").mkdir(parents=True)
    (assets / "models" / "qwen.gguf").write_bytes(b"qwen")
    (assets / "models" / "rebel-large" / "config.json").write_text(
        "{}", encoding="utf-8"
    )
    (assets / "tesseract" / "tessdata").mkdir(parents=True)
    (assets / "tesseract" / "tesseract.exe").write_bytes(b"ocr")
    (assets / "tesseract" / "tessdata" / "eng.traineddata").write_bytes(b"eng")
    return assets


def test_assemble_copies_assets_and_writes_verifiable_manifest(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    result = assemble_module.assemble_portable(
        repo_root=repo,
        frozen_app=_fake_frozen_app(repo),
        assets=_fake_assets(repo),
    )

    verify_manifest(
        result,
        result / "models" / "manifest.json",
        force_hash=True,
    )
    assert (result / "ModuleToFlashcards.exe").is_file()
    assert (result / "runtime" / "python.dll").is_file()
    assert (result / "tesseract" / "tesseract.exe").is_file()
    assert (result / "data" / "pipeline_output").is_dir()
    assert (result / "licenses" / "build-versions.txt").is_file()


def test_assemble_rejects_output_outside_repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    with pytest.raises(ValueError, match="distribution directory"):
        assemble_module.assemble_portable(
            repo_root=repo,
            frozen_app=_fake_frozen_app(repo),
            assets=_fake_assets(repo),
            output_dir=tmp_path / "outside",
        )


def test_assemble_rejects_incomplete_frozen_application(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    frozen = _fake_frozen_app(repo)
    (frozen / "ModuleToFlashcards.exe").unlink()

    with pytest.raises(ValueError, match="ModuleToFlashcards.exe"):
        assemble_module.assemble_portable(
            repo_root=repo,
            frozen_app=frozen,
            assets=_fake_assets(repo),
        )
