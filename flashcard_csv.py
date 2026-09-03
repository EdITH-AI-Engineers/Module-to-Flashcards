from __future__ import annotations

import csv
import io
from pathlib import Path
import tempfile
from typing import Iterable, Sequence

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

    first = _render_csv(identity, clusters[:10])
    second = _render_csv(identity, clusters[10:])
    return (
        f"Module {identity.module_number}.1\n"
        f"{first}\n"
        f"Module {identity.module_number}.2\n"
        f"{second}"
    )


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
