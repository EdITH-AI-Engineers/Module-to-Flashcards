import json

import pytest

from flashcard_types import CompletionTruncatedError
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
    backend = Responses('{"remove":[false,true]}', '{"remove":[false,true]}')

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


def test_qwen_checker_retries_wrong_decision_count_then_accepts_valid_json():
    backend = Responses(
        '{"remove":[false]}',
        '{"remove":[false,false]}',
        '{"remove":[false,false]}',
    )

    checked = check_knowledge_graph(sample_graph(), backend, max_retries=2)

    assert len(checked["facts"]) == 2
    assert len(backend.calls) == 3


def test_qwen_checker_refuses_to_publish_a_graph_with_no_lesson_facts():
    backend = Responses('{"remove":[true,true]}', '{"remove":[false,false]}')

    with pytest.raises(KnowledgeGraphCheckError, match="every lesson fact"):
        check_knowledge_graph(sample_graph(), backend)


def test_checker_prompt_is_domain_neutral():
    backend = Responses('{"remove":[false,false]}', '{"remove":[false,false]}')

    check_knowledge_graph(sample_graph(), backend)

    system = backend.calls[0][0]
    assert "engineering" in system
    assert "humanities" in system
    assert "multimedia arts" in system
    assert "HCI" not in system


def test_checker_schema_bounds_output_to_one_boolean_per_item():
    backend = Responses('{"remove":[false,true]}', '{"remove":[false,false]}')

    check_knowledge_graph(sample_graph(), backend)

    for _system, user, max_tokens, schema in backend.calls:
        decision_schema = schema["properties"]["remove"]
        assert decision_schema["minItems"] == 2
        assert decision_schema["maxItems"] == 2
        assert max_tokens < 512
        assert "exactly 2 booleans" in user


def test_checker_retries_a_length_truncation_with_bounded_larger_budget():
    class TruncateOnce:
        def __init__(self):
            self.calls = []
            self.truncated = False

        def complete(self, system, user, *, max_tokens, schema=None):
            self.calls.append((max_tokens, schema))
            if not self.truncated:
                self.truncated = True
                raise CompletionTruncatedError("truncated")
            return '{"remove":[false,false]}'

    backend = TruncateOnce()

    checked = check_knowledge_graph(sample_graph(), backend)

    assert len(checked["facts"]) == 2
    assert [call[0] for call in backend.calls[:2]] == [80, 160]
    assert all(
        call[1]["properties"]["remove"]["maxItems"] == 2
        for call in backend.calls
    )
