from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from portable_manifest import verify_manifest, write_manifest


_VERSION_PACKAGES = (
    "fastapi",
    "huggingface-hub",
    "llama-cpp-python",
    "Pillow",
    "PyMuPDF",
    "pytesseract",
    "sentencepiece",
    "torch",
    "transformers",
    "uvicorn",
)


def _distribution_output(repo_root: Path, output_dir: Path | None) -> Path:
    dist_root = (repo_root / "dist").resolve()
    output = Path(output_dir or dist_root / "ModuleToFlashcards").resolve()
    try:
        relative = output.relative_to(dist_root)
    except ValueError as exc:
        raise ValueError(
            f"distribution directory must be inside repository dist: {output}"
        ) from exc
    if not relative.parts:
        raise ValueError("distribution directory cannot be the repository dist root")
    return output


def _require_directory(path: Path, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        raise ValueError(f"{label} directory not found: {resolved}")
    return resolved


def _build_versions() -> str:
    lines = ["ModuleToFlashcards portable build dependencies"]
    for package in _VERSION_PACKAGES:
        try:
            value = metadata.version(package)
        except metadata.PackageNotFoundError:
            value = "not-installed"
        lines.append(f"{package}=={value}")
    return "\n".join(lines) + "\n"


def _copy_license_files(source: Path, destination: Path, prefix: str) -> None:
    for pattern in ("LICENSE*", "COPYING*", "NOTICE*"):
        for candidate in sorted(source.glob(pattern)):
            if candidate.is_file():
                shutil.copy2(candidate, destination / f"{prefix}-{candidate.name}")


def _safe_declared_path(value: object, label: str) -> PurePosixPath:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a relative file path")
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts or ":" in path.parts[0]:
        raise ValueError(f"{label} must be a safe relative file path: {value}")
    return path


def _declared_model_files(repo_root: Path) -> tuple[PurePosixPath, tuple[PurePosixPath, ...]]:
    lock_path = repo_root / "packaging" / "model-lock.json"
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        qwen = _safe_declared_path(lock["qwen"]["filename"], "Qwen filename")
        rebel = tuple(
            _safe_declared_path(item, "REBEL allow pattern")
            for item in lock["rebel"]["allow_patterns"]
        )
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"could not read asset declarations: {lock_path}") from exc
    if len({item.as_posix().casefold() for item in rebel}) != len(rebel):
        raise ValueError("REBEL asset declarations contain duplicate paths")
    return qwen, rebel


def _copy_declared_models(repo_root: Path, source: Path, destination: Path) -> None:
    qwen, rebel_files = _declared_model_files(repo_root)
    destination.mkdir()
    qwen_source = source.joinpath(*qwen.parts)
    if not qwen_source.is_file():
        raise ValueError(f"staged Qwen model is missing: {qwen.as_posix()}")
    shutil.copy2(qwen_source, destination.joinpath(*qwen.parts))

    rebel_source = source / "rebel-large"
    rebel_destination = destination / "rebel-large"
    for relative in rebel_files:
        source_file = rebel_source.joinpath(*relative.parts)
        if not source_file.is_file():
            raise ValueError(f"staged REBEL asset is missing: {relative.as_posix()}")
        target = rebel_destination.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target)


def _manifest_assets(output: Path) -> tuple[Path, ...]:
    roots = (output / "models", output / "tesseract", output / "licenses")
    files = []
    for root in roots:
        files.extend(
            candidate
            for candidate in root.rglob("*")
            if candidate.is_file() and candidate.name != "manifest.json"
        )
    return tuple(sorted(files, key=lambda item: item.as_posix().casefold()))


def assemble_portable(
    *,
    repo_root: Path = PROJECT_ROOT,
    frozen_app: Path,
    assets: Path,
    output_dir: Path | None = None,
) -> Path:
    repo_root = Path(repo_root).resolve()
    frozen_app = _require_directory(frozen_app, "frozen application")
    assets = _require_directory(assets, "staged assets")
    output = _distribution_output(repo_root, output_dir)
    executable = frozen_app / "ModuleToFlashcards.exe"
    runtime = frozen_app / "runtime"
    if not executable.is_file():
        raise ValueError(f"frozen application is missing ModuleToFlashcards.exe: {executable}")
    if not runtime.is_dir():
        raise ValueError(f"frozen application is missing runtime directory: {runtime}")
    models_source = _require_directory(assets / "models", "staged models")
    tesseract_source = _require_directory(assets / "tesseract", "staged Tesseract")
    if frozen_app == output:
        raise ValueError("frozen application and distribution directory must be different")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(frozen_app, output)
    _copy_declared_models(repo_root, models_source, output / "models")
    shutil.copytree(tesseract_source, output / "tesseract")

    for relative in ("uploads", "pipeline_output", "temporary", "logs"):
        (output / "data" / relative).mkdir(parents=True, exist_ok=True)

    licenses = output / "licenses"
    licenses.mkdir()
    (licenses / "build-versions.txt").write_text(_build_versions(), encoding="utf-8")
    _copy_license_files(repo_root, licenses, "project")
    _copy_license_files(tesseract_source, licenses, "tesseract")

    manifest = output / "models" / "manifest.json"
    write_manifest(output, _manifest_assets(output), manifest)
    verify_manifest(output, manifest, force_hash=True)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble the portable Windows bundle.")
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--frozen-app", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = assemble_portable(
        repo_root=args.repo_root,
        frozen_app=args.frozen_app,
        assets=args.assets,
        output_dir=args.output_dir,
    )
    print(f"Portable distribution assembled at {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
