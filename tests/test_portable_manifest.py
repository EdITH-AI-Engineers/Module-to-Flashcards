from __future__ import annotations

import json
from pathlib import Path

import pytest

import portable_manifest
from portable_manifest import BundleVerificationError, verify_manifest, write_manifest


def _asset(root: Path, relative: str, content: bytes = b"model") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_verify_manifest_hashes_once_then_uses_matching_cache(tmp_path, monkeypatch):
    asset = _asset(tmp_path, "models/qwen.gguf")
    manifest = tmp_path / "models" / "manifest.json"
    cache = tmp_path / "data" / ".verification-cache.json"
    write_manifest(tmp_path, (asset,), manifest)

    verify_manifest(tmp_path, manifest, cache_path=cache)
    monkeypatch.setattr(
        portable_manifest,
        "sha256_file",
        lambda _path: pytest.fail("matching cache should avoid rehashing"),
    )

    records = verify_manifest(tmp_path, manifest, cache_path=cache)
    assert records[0].path == "models/qwen.gguf"


def test_verify_manifest_detects_changed_asset(tmp_path):
    asset = _asset(tmp_path, "models/qwen.gguf")
    manifest = tmp_path / "models" / "manifest.json"
    cache = tmp_path / "data" / ".verification-cache.json"
    write_manifest(tmp_path, (asset,), manifest)
    verify_manifest(tmp_path, manifest, cache_path=cache)

    asset.write_bytes(b"damaged")

    with pytest.raises(BundleVerificationError, match="models/qwen.gguf"):
        verify_manifest(tmp_path, manifest, cache_path=cache)


def test_force_hash_ignores_cached_verification(tmp_path, monkeypatch):
    asset = _asset(tmp_path, "models/qwen.gguf")
    manifest = tmp_path / "models" / "manifest.json"
    cache = tmp_path / "data" / ".verification-cache.json"
    write_manifest(tmp_path, (asset,), manifest)
    verify_manifest(tmp_path, manifest, cache_path=cache)
    original = portable_manifest.sha256_file
    calls = []
    monkeypatch.setattr(
        portable_manifest,
        "sha256_file",
        lambda path: calls.append(path) or original(path),
    )

    verify_manifest(tmp_path, manifest, cache_path=cache, force_hash=True)

    assert calls == [asset]


@pytest.mark.parametrize(
    "relative",
    ("../outside.bin", "/absolute.bin", "C:/absolute.bin"),
)
def test_verify_manifest_rejects_unsafe_asset_paths(tmp_path, relative):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": [{"path": relative, "size": 1, "sha256": "0" * 64}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(BundleVerificationError, match="unsafe asset path"):
        verify_manifest(tmp_path, manifest)


def test_verify_manifest_rejects_duplicate_normalized_paths(tmp_path):
    manifest = tmp_path / "manifest.json"
    record = {"size": 1, "sha256": "0" * 64}
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": [
                    {"path": "models/file.bin", **record},
                    {"path": "models\\file.bin", **record},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(BundleVerificationError, match="duplicate asset path"):
        verify_manifest(tmp_path, manifest)


def test_write_manifest_rejects_assets_outside_root(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    outside = _asset(tmp_path, "outside.bin")

    with pytest.raises(ValueError, match="outside bundle root"):
        write_manifest(root, (outside,), root / "manifest.json")
