from pathlib import Path
import sys

import pytest

from local_qwen import MODEL_FILENAME
from portable_paths import (
    application_root,
    build_paths,
    prepare_data_directories,
)


def test_portable_paths_keep_models_and_generated_data_beside_executable(tmp_path):
    paths = build_paths(tmp_path, portable=True)

    assert paths.portable is True
    assert paths.models == tmp_path / "models"
    assert paths.qwen_model == tmp_path / "models" / MODEL_FILENAME
    assert paths.rebel_model == tmp_path / "models" / "rebel-large"
    assert paths.tesseract_exe == tmp_path / "tesseract" / "tesseract.exe"
    assert paths.tessdata == tmp_path / "tesseract" / "tessdata"
    assert paths.data == tmp_path / "data"
    assert paths.uploads == tmp_path / "data" / "uploads"
    assert paths.outputs == tmp_path / "data" / "pipeline_output"
    assert paths.temporary == tmp_path / "data" / "temporary"
    assert paths.logs == tmp_path / "data" / "logs"


def test_source_paths_preserve_existing_developer_directories(tmp_path):
    paths = build_paths(tmp_path, portable=False)

    assert paths.data == tmp_path
    assert paths.uploads == tmp_path / "pipeline_uploads"
    assert paths.outputs == tmp_path / "pipeline_output"
    assert paths.temporary == tmp_path / "pipeline_temporary"
    assert paths.logs == tmp_path / "pipeline_logs"


def test_application_root_uses_executable_parent_when_frozen(tmp_path, monkeypatch):
    executable = tmp_path / "ModuleToFlashcards.exe"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))

    assert application_root() == tmp_path.resolve()


def test_prepare_data_directories_creates_the_complete_portable_tree(tmp_path):
    paths = build_paths(tmp_path, portable=True)

    prepare_data_directories(paths)

    assert all(
        path.is_dir()
        for path in (
            paths.data,
            paths.uploads,
            paths.outputs,
            paths.temporary,
            paths.logs,
        )
    )


def test_prepare_data_directories_reports_the_unwritable_data_path(tmp_path):
    paths = build_paths(tmp_path, portable=True)
    paths.data.write_text("not a directory", encoding="utf-8")

    with pytest.raises(RuntimeError, match=str(paths.data).replace("\\", "\\\\")):
        prepare_data_directories(paths)
