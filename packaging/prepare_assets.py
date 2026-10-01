from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Callable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from portable_manifest import sha256_file


LOCK_PATH = Path(__file__).with_name("model-lock.json")
ASSETS_DIR = Path(__file__).with_name("assets")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")

_GENERATOR_LOCK = {
    "repo_id": "mistralai/Ministral-3-3B-Instruct-2512-GGUF",
    "revision": "eb599d408350ea2bb60452cb86be7c7b2fc28227",
    "filename": "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
    "size": 2147023008,
    "sha256": "9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8",
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
def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} lock must be an object")
    return value


def validate_model_lock(lock: object) -> Mapping[str, Any]:
    payload = _mapping(lock, "model")
    if payload.get("schema_version") != 2:
        raise ValueError("unsupported model-lock schema")

    generator = _mapping(payload.get("generator"), "generator")
    revision = generator.get("revision")
    if not isinstance(revision, str) or not _REVISION_PATTERN.fullmatch(revision):
        raise ValueError("generator requires an immutable revision")
    for key in ("repo_id", "revision", "filename", "size"):
        if generator.get(key) != _GENERATOR_LOCK[key]:
            raise ValueError(f"unexpected locked generator {key}")
    if generator.get("sha256") != _GENERATOR_LOCK["sha256"]:
        raise ValueError("unexpected locked generator digest")

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

    return payload


def load_model_lock(path: Path = LOCK_PATH) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read model lock: {path}") from exc
    return validate_model_lock(payload)


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _require_descendant(path: Path, parent: Path, label: str) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(Path(parent).resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must be inside {parent}") from exc
    return resolved


def prepare_assets(
    *,
    repo_root: Path = PROJECT_ROOT,
    assets_dir: Path | None = None,
    model_download: Callable[..., str] | None = None,
    rebel_download: Callable[..., str] | None = None,
) -> Path:
    repo_root = Path(repo_root).resolve()
    packaging_root = repo_root / "packaging"
    destination = _require_descendant(
        assets_dir or packaging_root / "assets",
        packaging_root,
        "assets directory",
    )
    lock = load_model_lock(packaging_root / "model-lock.json")

    if model_download is None or rebel_download is None:
        from huggingface_hub import hf_hub_download, snapshot_download

        model_download = model_download or hf_hub_download
        rebel_download = rebel_download or snapshot_download

    models = destination / "models"
    models.mkdir(parents=True, exist_ok=True)
    generator = lock["generator"]
    model_result = Path(
        model_download(
            repo_id=generator["repo_id"],
            revision=generator["revision"],
            filename=generator["filename"],
            local_dir=str(models),
        )
    )
    model_target = models / str(generator["filename"])
    if not model_target.is_file() and model_result.is_file():
        shutil.copy2(model_result, model_target)
    if not model_target.is_file():
        raise ValueError(f"model download did not produce {model_target.name}")
    if model_target.stat().st_size != generator["size"]:
        raise ValueError("downloaded model size does not match model lock")
    if _sha256(model_target) != generator["sha256"]:
        raise ValueError("downloaded model digest does not match model lock")

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

    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage locked portable release assets.")
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = prepare_assets(repo_root=args.repo_root)
    print(f"Prepared locked assets at {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
