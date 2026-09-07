from __future__ import annotations

from collections import Counter
import csv
import io
from pathlib import Path
import tempfile
from typing import Iterable, Sequence
from uuid import UUID

from flashcard_contract import (
    CARDS_PER_BLOCK,
    CARDS_PER_CLUSTER,
    CLUSTERS_PER_BLOCK,
    CLUSTERS_PER_MODULE,
)
from flashcard_types import FlashcardCluster, ModuleIdentity
from flashcard_validator import validate_module


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


def write_module_output(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
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
