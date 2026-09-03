from pathlib import Path

import pytest

import main


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

        def run(self, identity, facts):
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

        def run(self, identity, facts):
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
