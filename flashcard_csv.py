from __future__ import annotations

from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
from typing import Iterable, Sequence
from uuid import UUID

from artifact_paths import flashcard_part_paths, flashcard_receipt_path

from flashcard_contract import (
    CARDS_PER_BLOCK,
    CARDS_PER_CLUSTER,
    CLUSTERS_PER_BLOCK,
    CLUSTERS_PER_MODULE,
)
from flashcard_types import FlashcardCluster, ModuleIdentity
from flashcard_validator import are_near_duplicates, validate_module


CSV_COLUMNS = (
    "Type",
    "Question",
    "Correct Option",
    "Wrong Option 1",
    "Wrong Option 2",
    "Wrong Option 3",
    "is_true",
    "expalanation",
    "hint",
    "difficulty",
    "cluster",
    "course code",
    "module number",
)


def parse_rendered_module(
    content: str,
    identity: ModuleIdentity,
    *,
    validate_course_code: bool = True,
) -> tuple[tuple[tuple[str, ...], ...], ...]:
    """Parse the two rendered CSV blocks and return card rows only."""
    first_label = f"Module {identity.module_number}.1"
    second_label = f"Module {identity.module_number}.2"
    lines = content.splitlines(keepends=True)
    second_indexes = [
        index
        for index, line in enumerate(lines)
        if line.rstrip("\r\n") == second_label
    ]
    if not lines or lines[0].rstrip("\r\n") != first_label or len(second_indexes) != 1:
        raise ValueError("rendered module must contain exactly two labeled blocks")

    second_index = second_indexes[0]
    block_texts = (
        "".join(lines[1:second_index]),
        "".join(lines[second_index + 1 :]),
    )
    blocks: list[tuple[tuple[str, ...], ...]] = []
    cluster_counts: Counter[str] = Counter()
    for position, block_text in enumerate(block_texts, start=1):
        try:
            rows = list(
                csv.reader(
                    io.StringIO(block_text.strip("\r\n"), newline=""),
                    strict=True,
                )
            )
        except csv.Error as exc:
            raise ValueError(f"block {position} is not valid CSV") from exc
        if not rows or tuple(rows[0]) != CSV_COLUMNS:
            raise ValueError(f"block {position} must begin with the flashcard CSV header")
        data_rows = tuple(tuple(row) for row in rows[1:])
        if len(data_rows) != CARDS_PER_BLOCK:
            raise ValueError(
                f"block {position} must contain exactly {CARDS_PER_BLOCK} flashcards, "
                f"received {len(data_rows)}"
            )
        block_cluster_counts: Counter[str] = Counter()
        for card_position, row in enumerate(data_rows, start=1):
            prefix = f"block {position} card {card_position}"
            if len(row) != len(CSV_COLUMNS):
                raise ValueError(
                    f"{prefix} must contain exactly {len(CSV_COLUMNS)} columns"
                )
            if validate_course_code and row[11] != identity.course_code:
                raise ValueError(
                    f"{prefix} has course code {row[11]!r}, "
                    f"expected {identity.course_code!r}"
                )
            if row[12] != identity.module_number:
                raise ValueError(
                    f"{prefix} has module number {row[12]!r}, "
                    f"expected {identity.module_number!r}"
                )
            try:
                cluster = str(UUID(row[10]))
            except (AttributeError, ValueError) as exc:
                raise ValueError(f"{prefix} has an invalid cluster UUID") from exc
            cluster_counts[cluster] += 1
            block_cluster_counts[cluster] += 1
        if len(block_cluster_counts) != CLUSTERS_PER_BLOCK or any(
            count != CARDS_PER_CLUSTER for count in block_cluster_counts.values()
        ):
            raise ValueError(
                f"block {position} must contain exactly "
                f"{CLUSTERS_PER_BLOCK} complete clusters"
            )
        blocks.append(data_rows)

    if len(cluster_counts) != CLUSTERS_PER_MODULE:
        raise ValueError(
            f"rendered module must contain exactly {CLUSTERS_PER_MODULE} clusters, "
            f"received {len(cluster_counts)}"
        )
    if any(count != CARDS_PER_CLUSTER for count in cluster_counts.values()):
        raise ValueError(
            f"every rendered cluster must contain exactly {CARDS_PER_CLUSTER} flashcards"
        )
    return tuple(blocks)


def _rows(
    identity: ModuleIdentity,
    clusters: Iterable[FlashcardCluster],
):
    for cluster in clusters:
        for card in cluster.cards:
            yield (
                card.type,
                card.question,
                card.correct_option,
                card.wrong_option_1,
                card.wrong_option_2,
                card.wrong_option_3,
                "" if card.is_true is None else str(card.is_true),
                card.expalanation,
                card.hint,
                str(card.difficulty),
                cluster.cluster,
                identity.course_code,
                identity.module_number,
            )


def _render_csv(
    identity: ModuleIdentity,
    clusters: Sequence[FlashcardCluster],
) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    writer.writerows(_rows(identity, clusters))
    return stream.getvalue()


def render_module(
    identity: ModuleIdentity,
    clusters: Sequence[FlashcardCluster],
) -> str:
    errors = validate_module(clusters)
    if errors:
        raise ValueError("cannot render invalid module: " + "; ".join(errors))

    first = _render_csv(identity, clusters[:CLUSTERS_PER_BLOCK])
    second = _render_csv(identity, clusters[CLUSTERS_PER_BLOCK:])
    content = (
        f"Module {identity.module_number}.1\n"
        f"{first}\n"
        f"Module {identity.module_number}.2\n"
        f"{second}"
    )
    parse_rendered_module(content, identity)
    return content


def render_module_parts(
    identity: ModuleIdentity,
    clusters: Sequence[FlashcardCluster],
) -> tuple[str, str]:
    errors = validate_module(clusters)
    if errors:
        raise ValueError("cannot render invalid module: " + "; ".join(errors))
    parts = (
        _render_csv(identity, clusters[:CLUSTERS_PER_BLOCK]),
        _render_csv(identity, clusters[CLUSTERS_PER_BLOCK:]),
    )
    parse_rendered_parts(parts, identity)
    return parts


def parse_rendered_parts(
    parts: tuple[str, str],
    identity: ModuleIdentity,
    *,
    validate_course_code: bool = True,
) -> tuple[tuple[tuple[str, ...], ...], ...]:
    """Validate both standard CSVs together, including cross-part uniqueness."""
    if len(parts) != 2:
        raise ValueError("module must contain exactly two CSV parts")
    blocks = []
    cluster_counts: Counter[str] = Counter()
    questions: list[str] = []
    for position, content in enumerate(parts, start=1):
        try:
            rows = list(csv.reader(io.StringIO(content, newline=""), strict=True))
        except csv.Error as exc:
            raise ValueError(f"part {position} is not valid CSV") from exc
        if not rows or tuple(rows[0]) != CSV_COLUMNS:
            raise ValueError(f"part {position} must begin with the flashcard CSV header")
        data_rows = tuple(tuple(row) for row in rows[1:])
        if len(data_rows) != CARDS_PER_BLOCK:
            raise ValueError(
                f"part {position} must contain exactly {CARDS_PER_BLOCK} flashcards, "
                f"received {len(data_rows)}"
            )
        block_counts: Counter[str] = Counter()
        for card_position, row in enumerate(data_rows, start=1):
            prefix = f"part {position} card {card_position}"
            if len(row) != len(CSV_COLUMNS):
                raise ValueError(f"{prefix} must contain exactly {len(CSV_COLUMNS)} columns")
            if validate_course_code and row[11] != identity.course_code:
                raise ValueError(f"{prefix} has an incorrect course code")
            if row[12] != identity.module_number:
                raise ValueError(f"{prefix} has an incorrect module number")
            try:
                cluster = str(UUID(row[10]))
            except (AttributeError, ValueError) as exc:
                raise ValueError(f"{prefix} has an invalid cluster UUID") from exc
            if cluster != row[10].casefold():
                raise ValueError(f"{prefix} has a noncanonical cluster UUID")
            cluster_counts[cluster] += 1
            block_counts[cluster] += 1
            questions.append(row[1])
        if len(block_counts) != CLUSTERS_PER_BLOCK or any(
            count != CARDS_PER_CLUSTER for count in block_counts.values()
        ):
            raise ValueError(f"part {position} must contain exactly ten complete clusters")
        blocks.append(data_rows)
    if len(cluster_counts) != CLUSTERS_PER_MODULE or any(
        count != CARDS_PER_CLUSTER for count in cluster_counts.values()
    ):
        raise ValueError("parts must contain twenty distinct complete clusters")
    for index, left in enumerate(questions):
        for right in questions[index + 1 :]:
            if are_near_duplicates(left, right):
                raise ValueError("parts contain near-duplicate questions")
    return tuple(blocks)


def write_module_parts(
    base: Path,
    parts: tuple[str, str],
    identity: ModuleIdentity,
) -> tuple[Path, Path]:
    """Publish two staged CSVs, then a receipt that certifies the exact pair."""
    parse_rendered_parts(parts, identity)
    paths = flashcard_part_paths(base)
    receipt = flashcard_receipt_path(base)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    try:
        digests = []
        for path, content in zip(paths, parts):
            data = content.encode("utf-8-sig")
            with tempfile.NamedTemporaryFile(
                mode="wb", delete=False, dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp",
            ) as handle:
                temporary = Path(handle.name)
                staged.append(temporary)
                handle.write(data)
                handle.flush()
            digests.append(hashlib.sha256(data).hexdigest())
        receipt_data = {
            "version": 1,
            "course_code": identity.course_code,
            "module_number": identity.module_number,
            "parts": [
                {"name": path.name, "sha256": digest}
                for path, digest in zip(paths, digests)
            ],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False, dir=receipt.parent,
            prefix=f".{receipt.name}.", suffix=".tmp",
        ) as handle:
            staged_receipt = Path(handle.name)
            staged.append(staged_receipt)
            json.dump(receipt_data, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
        for path, temporary in zip(paths, staged[:2]):
            temporary.replace(path)
        staged_receipt.replace(receipt)
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)
    return paths


def valid_written_parts(
    base: Path,
    identity: ModuleIdentity,
    *,
    validate_course_code: bool = True,
) -> bool:
    paths = flashcard_part_paths(base)
    receipt = flashcard_receipt_path(base)
    try:
        manifest = json.loads(receipt.read_text(encoding="utf-8"))
        expected = {
            "version": 1,
            "course_code": identity.course_code,
            "module_number": identity.module_number,
            "parts": [
                {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in paths
            ],
        }
        if manifest != expected:
            return False
        parse_rendered_parts(
            tuple(path.read_text(encoding="utf-8-sig") for path in paths),
            identity,
            validate_course_code=validate_course_code,
        )
        return True
    except (OSError, UnicodeError, ValueError, TypeError):
        return False


def migrate_legacy_module(
    base: Path,
    identity: ModuleIdentity,
) -> tuple[Path, Path] | None:
    """Convert a valid labeled combined CSV while retaining it as a backup."""
    base = Path(base)
    try:
        legacy = base.read_text(encoding="utf-8-sig")
        blocks = parse_rendered_module(legacy, identity)
        parts = []
        for rows in blocks:
            stream = io.StringIO(newline="")
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(CSV_COLUMNS)
            writer.writerows(rows)
            parts.append(stream.getvalue())
        pair = (parts[0], parts[1])
        parse_rendered_parts(pair, identity)
    except (OSError, UnicodeError, ValueError):
        return None
    paths = write_module_parts(base, pair, identity)
    return paths if valid_written_parts(base, identity) else None


def write_module_output(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            newline="",
            delete=False,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
