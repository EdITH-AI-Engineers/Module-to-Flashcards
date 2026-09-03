import csv
from dataclasses import replace
import io
from pathlib import Path

import pytest

from flashcard_csv import CSV_COLUMNS, render_module, write_module_output
from flashcard_types import ModuleIdentity
from tests.factories import valid_clusters, with_question


def split_blocks(text):
    first_label, remainder = text.split("\n", 1)
    first_csv, second_csv = remainder.split("\n\nModule 1.2\n", 1)
    return first_label, first_csv, second_csv


def test_render_module_has_two_complete_fifty_row_blocks():
    text = render_module(ModuleIdentity("CPE0021", "1"), valid_clusters())

    first_label, first_csv, second_csv = split_blocks(text)
    first_rows = list(csv.reader(io.StringIO(first_csv)))
    second_rows = list(csv.reader(io.StringIO(second_csv)))

    assert first_label == "Module 1.1"
    assert tuple(first_rows[0]) == CSV_COLUMNS
    assert tuple(second_rows[0]) == CSV_COLUMNS
    assert len(first_rows) == 51
    assert len(second_rows) == 51
    assert len({row[10] for row in first_rows[1:]}) == 10
    assert len({row[10] for row in second_rows[1:]}) == 10
    assert all(len(row) == 13 for row in first_rows + second_rows)


def test_csv_escapes_commas_and_quotes():
    clusters = with_question(
        valid_clusters(),
        0,
        0,
        'Which "classification," applies to topic1 alpha1 beta1 revision0?',
    )

    text = render_module(ModuleIdentity("CPE,0021", "1"), clusters)

    assert '"CPE,0021"' in text
    assert '""classification,""' in text


def test_true_false_rows_use_zero_or_one_and_empty_options():
    text = render_module(ModuleIdentity("CPE0021", "1"), valid_clusters())
    _, first_csv, _ = split_blocks(text)
    rows = list(csv.DictReader(io.StringIO(first_csv)))
    true_false = next(row for row in rows if row["Type"] == "true-false")

    assert true_false["Correct Option"] == ""
    assert true_false["Wrong Option 1"] == ""
    assert true_false["is_true"] in {"0", "1"}


def test_render_module_rejects_unvalidated_cluster_count():
    with pytest.raises(ValueError, match="exactly 20 clusters"):
        render_module(ModuleIdentity("CPE0021", "1"), valid_clusters()[:19])


def test_write_module_output_replaces_target_without_temp_files(tmp_path: Path):
    target = tmp_path / "nested" / "module.txt"
    target.parent.mkdir()
    target.write_text("old", encoding="utf-8")

    write_module_output(target, "new")

    assert target.read_text(encoding="utf-8") == "new"
    assert list(target.parent.glob("*.tmp")) == []
