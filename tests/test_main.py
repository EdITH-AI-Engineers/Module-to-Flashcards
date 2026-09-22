import json
from pathlib import Path
import json

import pytest

import main
from tests.factories import plan_json


def test_course_corpus_round_trip_is_atomic_and_deduplicated(tmp_path):
    main.append_course_corpus(tmp_path, ("Concept A",), ("Question A?",))
    main.append_course_corpus(
        tmp_path,
        ("Concept A", "Concept B"),
        ("Question A?", "Question B?"),
    )

    assert main.load_course_corpus(tmp_path) == (
        ["Concept A", "Concept B"],
        ["Question A?", "Question B?"],
    )
    assert not list(tmp_path.glob(".course_corpus.json.*.tmp"))


def test_load_course_corpus_tolerates_missing_and_corrupt_files(tmp_path):
    assert main.load_course_corpus(tmp_path) == ([], [])
    (tmp_path / "course_corpus.json").write_text("not json", encoding="utf-8")
    assert main.load_course_corpus(tmp_path) == ([], [])


def test_parse_args_defaults_to_8k_context():
    assert main.parse_args(["graph.json"]).n_ctx == 8192


def test_parse_args_preserves_identity_strings():
    args = main.parse_args(
        [
            "output/knowledge_graph.json",
            "--course-code",
            "CPE0021",
            "--module-number",
            "01",
            "--skip-final-review",
        ]
    )

    assert args.course_code == "CPE0021"
    assert args.module_number == "01"
    assert args.final_review is False


def test_run_validates_graph_before_model_download(tmp_path, monkeypatch):
    missing = tmp_path / "missing.json"

    def fail_if_called(path):
        raise AssertionError("model download must not run for an invalid graph")

    monkeypatch.setattr(main, "ensure_model", fail_if_called)
    args = main.parse_args(
        [str(missing), "--course-code", "C", "--module-number", "1"]
    )

    with pytest.raises(Exception, match="not found"):
        main.run(args)


def test_run_writes_pipeline_result(tmp_path, monkeypatch, valid_clusters):
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        '{"metadata": {}, "nodes": [], "edges": '
        '[{"id":"e1","subject":"a","relation":"is","object":"b"}]}',
        encoding="utf-8",
    )
    output_path = tmp_path / "cards.txt"
    monkeypatch.setattr(main, "ensure_model", lambda path: tmp_path / "model.gguf")
    monkeypatch.setattr(main, "LocalQwenBackend", lambda *args, **kwargs: object())

    class FakePipeline:
        def __init__(self, backend, config, **kwargs):
            pass

        def run(self, identity, facts, **kwargs):
            return valid_clusters

    monkeypatch.setattr(main, "FlashcardPipeline", FakePipeline)
    args = main.parse_args(
        [
            str(graph_path),
            "--course-code",
            "CPE0021",
            "--module-number",
            "1",
            "--output",
            str(output_path),
        ]
    )

    result = main.run(args)

    assert result == output_path
    assert output_path.read_text(encoding="utf-8").startswith("Module 1.1\n")


def test_exhausted_bad_card_counts_do_not_create_output(tmp_path):
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "metadata": {},
                "nodes": [],
                "edges": [
                    {
                        "id": f"e{index}",
                        "subject": f"subject{index}",
                        "relation": "relates to",
                        "object": f"object{index}",
                    }
                    for index in range(1, 21)
                ],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "cards.txt"

    class UnderfullBackend:
        def __init__(self):
            self.responses = iter((plan_json(), '{"cards": []}', '{"cards": []}', '{"cards": []}'))

        def complete(self, system, user, *, max_tokens):
            return next(self.responses)

    args = main.parse_args(
        [
            str(graph_path),
            "--course-code", "CPE0021",
            "--module-number", "1",
            "--output", str(output_path),
            "--skip-final-review",
        ]
    )

    with pytest.raises(main.GenerationError, match="expected exactly 5 cards"):
        main.run(args, backend=UnderfullBackend())

    assert not output_path.exists()


def test_run_reuses_injected_backend(tmp_path, monkeypatch, valid_clusters):
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        '{"metadata": {}, "nodes": [], "edges": '
        '[{"id":"e1","subject":"a","relation":"is","object":"b"}]}',
        encoding="utf-8",
    )
    output_path = tmp_path / "cards.txt"
    shared_backend = object()
    monkeypatch.setattr(
        main,
        "ensure_model",
        lambda path: pytest.fail("an injected backend must bypass model loading"),
    )

    class FakePipeline:
        def __init__(self, backend, config, **kwargs):
            assert backend is shared_backend

        def run(self, identity, facts, **kwargs):
            return valid_clusters

    monkeypatch.setattr(main, "FlashcardPipeline", FakePipeline)
    args = main.parse_args(
        [
            str(graph_path),
            "--course-code", "CPE0021",
            "--module-number", "1",
            "--output", str(output_path),
        ]
    )

    assert main.run(args, backend=shared_backend) == output_path


def test_run_threads_and_updates_course_corpus(tmp_path, monkeypatch, valid_clusters):
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        '{"metadata": {}, "nodes": [], "edges": '
        '[{"id":"e1","subject":"a","relation":"is","object":"b"}]}',
        encoding="utf-8",
    )
    course_dir = tmp_path / "CPE0021"
    course_dir.mkdir()
    corpus_path = course_dir / "course_corpus.json"
    corpus_path.write_text(
        json.dumps(
            {
                "concept_names": ["Earlier concept"],
                "questions": ["What was asked earlier?"],
            }
        ),
        encoding="utf-8",
    )
    output_path = course_dir / "module-2" / "flashcards.txt"
    captured = {}

    class FakePipeline:
        def __init__(self, backend, config, **kwargs):
            pass

        def run(self, identity, facts, **kwargs):
            captured.update(kwargs)
            return valid_clusters

    monkeypatch.setattr(main, "FlashcardPipeline", FakePipeline)
    args = main.parse_args(
        [
            str(graph_path),
            "--course-code",
            "CPE0021",
            "--module-number",
            "2",
            "--output",
            str(output_path),
            "--course-corpus",
            str(corpus_path),
        ]
    )

    main.run(args, backend=object())

    assert captured == {
        "prior_concept_names": ["Earlier concept"],
        "prior_questions": ["What was asked earlier?"],
    }
    concepts, questions = main.load_course_corpus(course_dir)
    assert concepts[0] == "Earlier concept"
    assert questions[0] == "What was asked earlier?"
    assert len(concepts) == 21
    assert len(questions) == 101


def test_failed_output_write_does_not_update_course_corpus(
    tmp_path, monkeypatch, valid_clusters
):
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        '{"metadata": {}, "nodes": [], "edges": '
        '[{"id":"e1","subject":"a","relation":"is","object":"b"}]}',
        encoding="utf-8",
    )
    course_dir = tmp_path / "course"
    corpus_path = course_dir / "course_corpus.json"

    class FakePipeline:
        def __init__(self, backend, config, **kwargs):
            pass

        def run(self, identity, facts, **kwargs):
            return valid_clusters

    monkeypatch.setattr(main, "FlashcardPipeline", FakePipeline)
    monkeypatch.setattr(
        main,
        "write_module_output",
        lambda output, content: (_ for _ in ()).throw(OSError("write failed")),
    )
    args = main.parse_args(
        [
            str(graph_path),
            "--course-code",
            "CPE0021",
            "--module-number",
            "1",
            "--output",
            str(course_dir / "module-1" / "flashcards.txt"),
            "--course-corpus",
            str(corpus_path),
        ]
    )

    with pytest.raises(OSError, match="write failed"):
        main.run(args, backend=object())

    assert not corpus_path.exists()


def test_default_output_uses_resolved_graph_module(tmp_path, monkeypatch, valid_clusters):
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        '{"metadata": {"module_number": "01"}, "nodes": [], "edges": '
        '[{"id":"e1","subject":"a","relation":"is","object":"b"}]}',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "ensure_model", lambda path: tmp_path / "model.gguf")
    monkeypatch.setattr(main, "LocalQwenBackend", lambda *args, **kwargs: object())

    class FakePipeline:
        def __init__(self, backend, config, **kwargs):
            pass

        def run(self, identity, facts, **kwargs):
            return valid_clusters

    monkeypatch.setattr(main, "FlashcardPipeline", FakePipeline)
    args = main.parse_args([str(graph_path), "--course-code", "CPE0021"])

    result = main.run(args)

    assert result == Path("flashcards") / "module_01.txt"
    assert result.is_file()


def test_smoke_test_checks_for_exact_json_status(tmp_path, monkeypatch):
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        '{"metadata": {}, "nodes": [], "edges": '
        '[{"id":"e1","subject":"a","relation":"is","object":"b"}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(main, "ensure_model", lambda path: tmp_path / "model.gguf")

    class FakeBackend:
        def __init__(self, *args, **kwargs):
            pass

        def complete(self, system, user, *, max_tokens):
            return '{"status":"ok"}'

    monkeypatch.setattr(main, "LocalQwenBackend", FakeBackend)
    args = main.parse_args(
        [
            str(graph_path),
            "--course-code",
            "CPE0021",
            "--module-number",
            "1",
            "--smoke-test",
        ]
    )

    assert main.run(args) is None


def test_main_returns_one_and_prints_focused_error(capsys):
    exit_code = main.main(
        ["does-not-exist.json", "--course-code", "CPE0021", "--module-number", "1"]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.startswith("Error: knowledge graph not found:")
