from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
import sys

from local_qwen import MODEL_FILENAME


@dataclass(frozen=True)
class PortablePaths:
    root: Path
    models: Path
    qwen_model: Path
    rebel_model: Path
    tesseract_exe: Path
    tessdata: Path
    data: Path
    uploads: Path
    outputs: Path
    temporary: Path
    logs: Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def application_root() -> Path:
    base = Path(sys.executable) if is_frozen() else Path(__file__)
    return base.resolve().parent


def build_paths(
    root: Path | None = None,
    *,
    portable: bool | None = None,
) -> PortablePaths:
    resolved_root = (root or application_root()).resolve()
    portable = is_frozen() if portable is None else portable
    data = resolved_root / "data" if portable else resolved_root
    return PortablePaths(
        root=resolved_root,
        models=resolved_root / "models",
        qwen_model=resolved_root / "models" / MODEL_FILENAME,
        rebel_model=resolved_root / "models" / "rebel-large",
        tesseract_exe=resolved_root / "tesseract" / "tesseract.exe",
        tessdata=resolved_root / "tesseract" / "tessdata",
        data=data,
        uploads=data / "uploads" if portable else resolved_root / "pipeline_uploads",
        outputs=(
            data / "pipeline_output"
            if portable
            else resolved_root / "pipeline_output"
        ),
        temporary=(
            data / "temporary"
            if portable
            else resolved_root / "pipeline_temporary"
        ),
        logs=data / "logs" if portable else resolved_root / "pipeline_logs",
    )


def prepare_data_directories(paths: PortablePaths) -> None:
    directories = (
        paths.data,
        paths.uploads,
        paths.outputs,
        paths.temporary,
        paths.logs,
    )
    try:
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        marker: Path | None = None
        try:
            with NamedTemporaryFile(
                dir=paths.data,
                prefix=".write-test-",
                delete=False,
            ) as handle:
                marker = Path(handle.name)
        finally:
            if marker is not None:
                marker.unlink(missing_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"portable data directory is not writable: {paths.data}"
        ) from exc
