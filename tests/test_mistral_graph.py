import json

import mistral_graph
import text_extractor
from tests.batch_helpers import structured_content


def test_mistral_extracts_relationships_with_source_evidence_and_preserves_facts(tmp_path):
    source = tmp_path / "module.txt"
    source.write_text(structured_content(), encoding="utf-8")
    args = text_extractor.parse_args([str(source), "--output-dir", str(tmp_path / "graph"),
        "--course-code", "CPE0021", "--module-number", "01"])
    class Backend:
        model_name = "test-mistral"
        def complete(self, system, user, **kwargs):
            assert "processor" in user
            return json.dumps({"triples": [
                {"subject": "processor", "relation": "defines", "object": "executes instructions"},
                {"subject": "invented", "relation": "is_a", "object": "hallucination"}]})
    graph_path, csv_path = mistral_graph.run(args, Backend())
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    assert graph["metadata"]["model"] == "test-mistral"
    assert graph["metadata"]["course_code"] == "CPE0021"
    assert graph["facts"] and len(graph["edges"]) == 1
    assert "ALU" in graph["edges"][0]["evidence"][0]["text"]
    assert csv_path.is_file()


def test_relationship_extraction_repairs_unfinished_reply(tmp_path):
    from flashcard_types import IncompleteCompletionError
    source = tmp_path / "module.txt"
    source.write_text(structured_content(), encoding="utf-8")
    args = text_extractor.parse_args([str(source), "--output-dir", str(tmp_path / "graph"),
        "--course-code", "CPE0021", "--module-number", "01"])
    calls = []
    class Backend:
        model_name = "test-mistral"
        def complete(self, system, user, **kwargs):
            calls.append(system)
            if len(calls) == 1:
                raise IncompleteCompletionError("Unfinished response (finish_reason: error).", finish_reason="error")
            return '{"triples":[]}'
    graph_path, _ = mistral_graph.run(args, Backend())
    assert len(calls) == 2 and "finish_reason: error" in calls[1]
    assert json.loads(graph_path.read_text(encoding="utf-8"))["facts"]
