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

_QWEN_LOCK = {
    "repo_id": "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF",
    "revision": "ae44f08e1392f39c0e474af10c3ff8355c8b6688",
    "filename": "Qwen_Qwen3-4B-Instruct-2507-Q5_K_M.gguf",
    "size": 2889513696,
    "sha256": "66713ce35a58a82fe87642d4ec13425bf9b9a46800fff5c49a665ef5701439dc",
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
