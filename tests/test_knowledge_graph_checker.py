import json

import pytest

from knowledge_graph_checker import (
    KnowledgeGraphCheckError,
    check_knowledge_graph,
    is_checked_graph,
)


class Responses:
    def __init__(self, *values):
        self.values = iter(values)
        self.calls = []

    def complete(self, system, user, *, max_tokens, schema=None):
        self.calls.append((system, user, max_tokens, schema))
        return next(self.values)


def sample_graph():
    return {
        "metadata": {
            "course_code": "GEN101",
            "module_number": "1",
            "module_title": "Foundations",
        },
        "facts": [
            {"id": "f1", "statement": "A load creates stress in a material."},
            {"id": "f2", "statement": "This module discusses important topics."},
        ],
        "nodes": [
            {"id": "n1", "label": "load"},
            {"id": "n2", "label": "stress"},
            {"id": "n3", "label": "slide"},
            {"id": "n4", "label": "next"},
        ],
        "edges": [
            {
                "id": "e1",
                "source": "n1",
                "target": "n2",
                "subject": "load",
                "relation": "creates",
                "object": "stress",
            },
            {
                "id": "e2",
                "source": "n3",
                "target": "n4",
                "subject": "slide",
                "relation": "directs",
                "object": "click next",
            },
        ],
    }


def test_qwen_checker_prunes_only_selected_items_and_updates_graph_counts():
    backend = Responses('{"remove_ids":["f2"]}', '{"remove_ids":["e2"]}')

    checked = check_knowledge_graph(sample_graph(), backend)

    assert [fact["id"] for fact in checked["facts"]] == ["f1"]
    assert [edge["id"] for edge in checked["edges"]] == ["e1"]
    assert [node["id"] for node in checked["nodes"]] == ["n1", "n2"]
    assert checked["metadata"]["fact_count"] == 1
    assert checked["metadata"]["edge_count"] == 1
    assert checked["metadata"]["node_count"] == 2
    assert is_checked_graph(checked)
    assert len(backend.calls) == 2
    assert all(call[3] is not None for call in backend.calls)


def test_qwen_checker_retries_unknown_ids_then_accepts_valid_json():
    backend = Responses(
        '{"remove_ids":["invented"]}',
        '{"remove_ids":[]}',
        '{"remove_ids":[]}',
    )

    checked = check_knowledge_graph(sample_graph(), backend, max_retries=2)

    assert len(checked["facts"]) == 2
    assert len(backend.calls) == 3


def test_qwen_checker_refuses_to_publish_a_graph_with_no_lesson_facts():
    backend = Responses('{"remove_ids":["f1","f2"]}', '{"remove_ids":[]}')

    with pytest.raises(KnowledgeGraphCheckError, match="every lesson fact"):
        check_knowledge_graph(sample_graph(), backend)


def test_checker_prompt_is_domain_neutral():
    backend = Responses('{"remove_ids":[]}', '{"remove_ids":[]}')

    check_knowledge_graph(sample_graph(), backend)

    system = backend.calls[0][0]
    assert "engineering" in system
    assert "humanities" in system
    assert "multimedia arts" in system
    assert "HCI" not in system
