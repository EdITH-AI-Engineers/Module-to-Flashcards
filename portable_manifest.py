from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from tempfile import NamedTemporaryFile
from typing import Any, Sequence


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class AssetRecord:
    path: str
    size: int
    sha256: str


class BundleVerificationError(RuntimeError):
    """Raised when a portable-bundle asset cannot be trusted."""


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_write(destination: Path, payload: object) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _relative_asset_path(root: Path, asset: Path) -> str:
    try:
        relative = asset.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"asset is outside bundle root: {asset}") from exc
    return relative.as_posix()


def write_manifest(
    root: Path,
    assets: Sequence[Path],
    destination: Path,
) -> None:
    root = Path(root).resolve()
    records = []
    for asset in assets:
        resolved = Path(asset).resolve()
        relative = _relative_asset_path(root, resolved)
        if not resolved.is_file():
            raise ValueError(f"asset is not a file: {asset}")
        records.append(
            AssetRecord(
                path=relative,
                size=resolved.stat().st_size,
                sha256=sha256_file(resolved),
            )
        )
    records.sort(key=lambda item: item.path.casefold())
    _atomic_json_write(
        destination,
        {
            "schema_version": 1,
            "files": [asdict(record) for record in records],
        },
    )


def _safe_relative_path(value: object) -> tuple[str, PurePosixPath]:
    if not isinstance(value, str) or not value.strip():
        raise BundleVerificationError("unsafe asset path: path must be a string")
    normalized_text = value.replace("\\", "/")
    relative = PurePosixPath(normalized_text)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or ":" in relative.parts[0]
    ):
        raise BundleVerificationError(f"unsafe asset path: {value}")
    normalized = relative.as_posix()
    if normalized in {"", "."}:
        raise BundleVerificationError(f"unsafe asset path: {value}")
    return normalized, relative


def _parse_manifest(payload: object) -> tuple[AssetRecord, ...]:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise BundleVerificationError("unsupported or malformed manifest schema")
    files = payload.get("files")
    if not isinstance(files, list):
        raise BundleVerificationError("manifest files must be an array")

    records = []
    seen = set()
    for raw in files:
        if not isinstance(raw, dict):
            raise BundleVerificationError("manifest file record must be an object")
        normalized, _ = _safe_relative_path(raw.get("path"))
        duplicate_key = normalized.casefold()
        if duplicate_key in seen:
            raise BundleVerificationError(f"duplicate asset path: {normalized}")
        seen.add(duplicate_key)

        size = raw.get("size")
        digest = raw.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise BundleVerificationError(f"invalid asset size: {normalized}")
        if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
            raise BundleVerificationError(f"invalid asset digest: {normalized}")
        records.append(AssetRecord(normalized, size, digest))
    return tuple(records)


def _read_cache(cache_path: Path | None) -> dict[str, Any]:
    if cache_path is None or not cache_path.is_file():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def verify_manifest(
    root: Path,
    manifest_path: Path,
    *,
    cache_path: Path | None = None,
    force_hash: bool = False,
) -> tuple[AssetRecord, ...]:
    root = Path(root).resolve()
    manifest_path = Path(manifest_path)
    cache_path = Path(cache_path) if cache_path is not None else None
    try:
        manifest_bytes = manifest_path.read_bytes()
        payload = json.loads(manifest_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleVerificationError(f"could not read bundle manifest: {manifest_path}") from exc

    records = _parse_manifest(payload)
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    cache = _read_cache(cache_path)
    cached_files = cache.get("files") if cache.get("manifest_sha256") == manifest_digest else {}
    if not isinstance(cached_files, dict):
        cached_files = {}
    updated_cache: dict[str, dict[str, object]] = {}

    for record in records:
        _, relative = _safe_relative_path(record.path)
        asset = root.joinpath(*relative.parts)
        try:
            resolved_asset = asset.resolve(strict=True)
            resolved_asset.relative_to(root)
        except (OSError, ValueError) as exc:
            raise BundleVerificationError(
                f"missing or unsafe bundled asset: {record.path}"
            ) from exc
        if not resolved_asset.is_file():
            raise BundleVerificationError(f"bundled asset is not a file: {record.path}")

        stat = resolved_asset.stat()
        if stat.st_size != record.size:
            raise BundleVerificationError(
                f"bundled asset size mismatch: {record.path}"
            )
        cache_record = {
            "sha256": record.sha256,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if force_hash or cached_files.get(record.path) != cache_record:
            if sha256_file(resolved_asset) != record.sha256:
                raise BundleVerificationError(
                    f"bundled asset digest mismatch: {record.path}"
                )
        updated_cache[record.path] = cache_record

    if cache_path is not None:
        _atomic_json_write(
            cache_path,
            {
                "schema_version": 1,
                "manifest_sha256": manifest_digest,
                "files": updated_cache,
            },
        )
    return records
