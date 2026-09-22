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


def test_extract_facts_prefers_normalized_lesson_facts_with_context():
    value = graph()
    value["facts"] = [
        {
            "id": "f1",
            "statement": "A project is a temporary endeavor.",
            "slides": [4],
            "kind": "definition",
            "topic": "Project Foundations",
        }
    ]

    facts = extract_graph_facts(value)

    assert len(facts) == 1
    assert facts[0].fact_id == "f1"
    assert facts[0].statement == "A project is a temporary endeavor."
    assert facts[0].slides == (4,)
    assert facts[0].topic == "Project Foundations"


def test_extract_facts_collapses_existing_cumulative_fact_ids():
    value = graph()
    prefix = "Norman's seven principles are: "
    principles = (
        "use knowledge in the world and in the head",
        "simplify task structures",
        "make things visible",
        "get mappings right",
        "exploit constraints",
        "design for error",
        "standardize when all else fails",
    )
    value["facts"] = [
        {
            "id": "f81",
            "statement": prefix + "; ".join(principles[:-1]),
            "slides": [30],
        },
        {
            "id": "f84",
            "statement": prefix + "; ".join(principles),
            "slides": [31],
        },
    ]

    facts = extract_graph_facts(value)

    assert len(facts) == 1
    assert facts[0].fact_id == "f84"
    assert facts[0].slides == (30, 31)


def test_extract_facts_drops_only_slide_self_referential_statements():
    value = graph()
    value["facts"] = [
        {
            "id": "f1",
            "statement": "The title of the slide is DIGITAL READING PORTFOLIO.",
            "slides": [1],
        },
        {
            "id": "f2",
            "statement": "Creating a Digital Reading Portfolio is the topic of this slide.",
            "slides": [2],
        },
        {
            "id": "f48",
            "statement": "The topic of the slide is creating a digital reading portfolio.",
            "slides": [12],
        },
        {
            "id": "f3",
            "statement": "A digital portfolio is a document containing selected work.",
            "slides": [3],
        },
        {
            "id": "f4",
            "statement": "A portfolio page can present a learner's reflection.",
            "slides": [4],
        },
    ]

    facts = extract_graph_facts(value)

    assert [fact.fact_id for fact in facts] == ["f3", "f4"]
    assert [fact.statement for fact in facts] == [
        "A digital portfolio is a document containing selected work.",
        "A portfolio page can present a learner's reflection.",
    ]


def test_extract_facts_filters_noise_from_existing_graph_without_a_minimum():
    value = graph()
    value["facts"] = [
        {
            "id": "f1",
            "statement": "Energy Transformations",
            "slides": [1],
            "kind": "content",
            "topic": "Energy Transformations",
        },
        {
            "id": "f2",
            "statement": "The student should be able to understand energy transfer.",
            "slides": [2],
            "kind": "knowledge_statement",
            "topic": "Objectives",
        },
        {
            "id": "f3",
            "statement": "https://example.edu/energy-reference",
            "slides": [3],
            "kind": "content",
            "topic": "References",
        },
        {
            "id": "f4",
            "statement": "Energy cannot be created or destroyed.",
            "slides": [4],
            "kind": "knowledge_statement",
            "topic": "Conservation of Energy",
        },
    ]

    facts = extract_graph_facts(value)

    assert [fact.fact_id for fact in facts] == ["f4"]
    assert facts[0].statement == "Energy cannot be created or destroyed."


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
