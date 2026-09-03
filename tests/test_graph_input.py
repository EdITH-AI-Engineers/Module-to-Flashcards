import json
from pathlib import Path

import pytest

from graph_input import (
    GraphInputError,
    extract_graph_facts,
    load_graph,
    resolve_identity,
)


def graph(metadata=None):
    return {
        "metadata": metadata or {},
        "nodes": [{"id": "n1", "label": "Binary", "degree": 2}],
        "edges": [
            {
                "id": "e1",
                "source": "n1",
                "target": "n2",
                "subject": "binary",
                "relation": "uses",
                "object": "base 2",
                "evidence": [
                    {"chunk_id": 1, "slides": [3], "text": "do not expose"}
                ],
            }
        ],
    }


def test_identity_precedence_preserves_exact_strings():
    identity = resolve_identity(
        graph({"course_code": "GRAPH101", "module_number": "01"}),
        course_code="CLI-202",
        module_number="2",
    )

    assert identity.course_code == "CLI-202"
    assert identity.module_number == "01"


def test_identity_reports_only_missing_course_code():
    with pytest.raises(GraphInputError, match="course code is required"):
        resolve_identity(graph({"module_number": "1"}), None, None)


def test_identity_reports_only_missing_module_number():
    with pytest.raises(GraphInputError, match="module number is required"):
        resolve_identity(graph({"course_code": "CPE0021"}), None, None)


def test_extract_facts_excludes_provenance_text():
    facts = extract_graph_facts(graph())

    assert facts[0].statement == "binary | uses | base 2"
    assert "do not expose" not in facts[0].statement


def test_extract_facts_rejects_graph_without_usable_relationships():
    value = graph()
    value["edges"] = [{"id": "e1", "subject": "binary", "relation": "uses"}]

    with pytest.raises(GraphInputError, match="no usable relationship facts"):
        extract_graph_facts(value)


def test_load_graph_rejects_missing_edges(tmp_path: Path):
    path = tmp_path / "graph.json"
    path.write_text(json.dumps({"metadata": {}, "nodes": []}), encoding="utf-8")

    with pytest.raises(GraphInputError, match="edges"):
        load_graph(path)


def test_load_graph_rejects_invalid_json(tmp_path: Path):
    path = tmp_path / "graph.json"
    path.write_text("not json", encoding="utf-8")

    with pytest.raises(GraphInputError, match="valid JSON"):
        load_graph(path)


def test_load_graph_reports_missing_file(tmp_path: Path):
    with pytest.raises(GraphInputError, match="not found"):
        load_graph(tmp_path / "missing.json")
