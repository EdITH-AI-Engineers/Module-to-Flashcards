from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = Path(__file__).with_name("model-lock.json")
ASSETS_DIR = Path(__file__).with_name("assets")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")

_QWEN_LOCK = {
    "repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
    "revision": "7dabda4d13d513e3e842b20f0d435c732f172cbe",
    "filename": "qwen2.5-3b-instruct-q5_k_m.gguf",
    "size": 2438740384,
    "sha256": "2c63dde5f2c9ab1fd64d47dee2d34dade6ba9ff62442d1d20b5342310c982081",
}
_REBEL_REQUIRED = {
    "config.json",
    "model.safetensors",
    "added_tokens.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}
_TESSERACT_LOCK = {
    "tesseract.exe": "babb405f4366b480d02cd8ff2bac8d497170f6c1711ce6f3d5d8bf0fb7fa6ed9",
    "tessdata/eng.traineddata": "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2",
}


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} lock must be an object")
    return value


def validate_model_lock(lock: object) -> Mapping[str, Any]:
    payload = _mapping(lock, "model")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported model-lock schema")

    qwen = _mapping(payload.get("qwen"), "Qwen")
    revision = qwen.get("revision")
    if not isinstance(revision, str) or not _REVISION_PATTERN.fullmatch(revision):
        raise ValueError("Qwen requires an immutable revision")
    for key in ("repo_id", "revision", "filename", "size"):
        if qwen.get(key) != _QWEN_LOCK[key]:
            raise ValueError(f"unexpected locked Qwen {key}")
    if qwen.get("sha256") != _QWEN_LOCK["sha256"]:
        raise ValueError("unexpected locked Qwen digest")

    rebel = _mapping(payload.get("rebel"), "REBEL")
    rebel_revision = rebel.get("revision")
    if (
        not isinstance(rebel_revision, str)
        or not _REVISION_PATTERN.fullmatch(rebel_revision)
    ):
        raise ValueError("REBEL requires an immutable REBEL revision")
    patterns = rebel.get("allow_patterns")
    if not isinstance(patterns, list) or not all(isinstance(item, str) for item in patterns):
        raise ValueError("REBEL allow_patterns must be a string array")
    missing = sorted(_REBEL_REQUIRED.difference(patterns))
    if missing:
        raise ValueError("REBEL lock is missing required files: " + ", ".join(missing))
    if "pytorch_model.bin" in patterns:
        raise ValueError("REBEL lock must select safetensors instead of pickle weights")

    tesseract = _mapping(payload.get("tesseract"), "Tesseract")
    if tesseract.get("version") != "5.4.0.20240606":
        raise ValueError("unexpected locked Tesseract version")
    required = _mapping(tesseract.get("required_files"), "Tesseract files")
    if dict(required) != _TESSERACT_LOCK:
        raise ValueError("unexpected locked Tesseract file digests")
    if not all(_DIGEST_PATTERN.fullmatch(value) for value in required.values()):
        raise ValueError("malformed locked Tesseract digest")
    return payload


def load_model_lock(path: Path = LOCK_PATH) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read model lock: {path}") from exc
    return validate_model_lock(payload)


def _sha256(path: Path) -> str:
    from portable_manifest import sha256_file

    return sha256_file(path)


def _require_descendant(path: Path, parent: Path, label: str) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(Path(parent).resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must be inside {parent}") from exc
    return resolved


def _replace_tree(source: Path, destination: Path, *, allowed_parent: Path) -> None:
    destination = _require_descendant(destination, allowed_parent, "asset destination")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def prepare_assets(
    *,
    repo_root: Path = PROJECT_ROOT,
    tesseract_dir: Path,
    assets_dir: Path | None = None,
    qwen_download: Callable[..., str] | None = None,
    rebel_download: Callable[..., str] | None = None,
) -> Path:
    repo_root = Path(repo_root).resolve()
    packaging_root = repo_root / "packaging"
    destination = _require_descendant(
        assets_dir or packaging_root / "assets",
        packaging_root,
        "assets directory",
    )
    tesseract_source = Path(tesseract_dir).resolve()
    lock = load_model_lock(packaging_root / "model-lock.json")

    if qwen_download is None or rebel_download is None:
        from huggingface_hub import hf_hub_download, snapshot_download

        qwen_download = qwen_download or hf_hub_download
        rebel_download = rebel_download or snapshot_download

    models = destination / "models"
    models.mkdir(parents=True, exist_ok=True)
    qwen = lock["qwen"]
    qwen_result = Path(
        qwen_download(
            repo_id=qwen["repo_id"],
            revision=qwen["revision"],
            filename=qwen["filename"],
            local_dir=str(models),
        )
    )
    qwen_target = models / str(qwen["filename"])
    if not qwen_target.is_file() and qwen_result.is_file():
        shutil.copy2(qwen_result, qwen_target)
    if not qwen_target.is_file():
        raise ValueError(f"Qwen download did not produce {qwen_target.name}")
    if qwen_target.stat().st_size != qwen["size"]:
        raise ValueError("downloaded Qwen size does not match model lock")
    if _sha256(qwen_target) != qwen["sha256"]:
        raise ValueError("downloaded Qwen digest does not match model lock")

    rebel = lock["rebel"]
    rebel_target = models / "rebel-large"
    rebel_download(
        repo_id=rebel["repo_id"],
        revision=rebel["revision"],
        allow_patterns=list(rebel["allow_patterns"]),
        ignore_patterns=["pytorch_model.bin", "*.msgpack", "*.h5"],
        local_dir=str(rebel_target),
    )
    missing = [name for name in rebel["allow_patterns"] if not (rebel_target / name).is_file()]
    if missing:
        raise ValueError("REBEL snapshot is missing locked files: " + ", ".join(missing))

    tesseract_target = destination / "tesseract"
    if not tesseract_source.is_dir():
        raise ValueError(f"Tesseract directory not found: {tesseract_source}")
    _replace_tree(tesseract_source, tesseract_target, allowed_parent=destination)
    required_files = lock["tesseract"]["required_files"]
    for relative, expected_digest in required_files.items():
        candidate = tesseract_target.joinpath(*relative.split("/"))
        if not candidate.is_file():
            raise ValueError(f"Tesseract directory is missing {relative}")
        if _sha256(candidate) != expected_digest:
            raise ValueError(f"Tesseract digest mismatch: {relative}")
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage locked portable release assets.")
    parser.add_argument("--tesseract-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = prepare_assets(repo_root=args.repo_root, tesseract_dir=args.tesseract_dir)
    print(f"Prepared locked assets at {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
