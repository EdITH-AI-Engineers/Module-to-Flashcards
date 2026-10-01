import json

import pytest

from benchmarks.metrics import summarize_metrics
from benchmarks.run import load_cases, parse_args, summarize_cluster_passes


def test_summary_reports_per_task_tokens_failures_and_pass_rate():
    calls = [
        {
            "task": "cluster", "prompt_tokens": 100,
            "completion_tokens": 20, "finish_reason": "stop",
            "exception": None, "max_tokens": 64, "elapsed_seconds": 1.0,
            "output_chars": 80, "contains_think": False,
        },
        {
            "task": "cluster", "prompt_tokens": 200,
            "completion_tokens": 64, "finish_reason": "length",
            "exception": "CompletionTruncatedError", "max_tokens": 64,
            "elapsed_seconds": 2.0,
            "output_chars": 256, "contains_think": True,
        },
        {
            "task": "retry", "prompt_tokens": None,
            "completion_tokens": None, "finish_reason": "context_error",
            "exception": "ContextWindowExceededError", "max_tokens": 64,
            "elapsed_seconds": 0.1,
        },
    ]
    task_counts = {
        "cluster": {
            "first_attempts": 2,
            "first_attempt_passes": 1,
            "retries": 1,
        }
    }

    result = summarize_metrics(calls, task_counts)

    assert result["tasks"]["cluster"]["prompt_tokens"] == {"mean": 150, "p95": 200}
    assert result["tasks"]["cluster"]["completion_tokens"] == {"mean": 42, "p95": 64}
    assert result["tasks"]["cluster"]["length_finishes"] == 1
    assert result["tasks"]["cluster"]["first_attempt_pass_rate"] == 0.5
    assert result["tasks"]["cluster"]["retry_count"] == 1
    assert result["tasks"]["cluster"]["chars_per_completion_token"] == 4.0
    assert result["tasks"]["cluster"]["visible_think_markers"] == 1
    assert result["tasks"]["retry"]["context_errors"] == 1
    assert result["overall"]["length_finishes"] == 1
    assert result["overall"]["context_errors"] == 1
    assert result["overall"]["retry_count"] == 1


def test_manifest_selects_three_modules_by_default_without_embedding_graph_content(tmp_path):
    graphs = []
    for number in range(1, 4):
        graph = tmp_path / f"graph-{number}.json"
        graph.write_text("{}", encoding="utf-8")
        graphs.append(graph)
    manifest = tmp_path / "modules.json"
    manifest.write_text(
        json.dumps(
            {
                "modules": [
                    {
                        "graph": graph.name,
                        "course_code": "GED0081",
                        "module_number": str(number),
                    }
                    for number, graph in enumerate(graphs, start=1)
                ]
            }
        ),
        encoding="utf-8",
    )

    args = parse_args([])
    cases = load_cases(manifest, args.modules)

    assert args.modules == 3
    assert [case.graph for case in cases] == graphs
    assert [case.identity.module_number for case in cases] == ["1", "2", "3"]
    with pytest.raises(ValueError, match="at least 4 modules"):
        load_cases(manifest, 4)


def test_benchmark_accepts_optional_cluster_batch_size():
    assert parse_args([]).clusters_per_call == 1
    assert parse_args(["--clusters-per-call", "2"]).clusters_per_call == 2
    assert parse_args(["--clusters-per-call", "5"]).clusters_per_call == 5
    with pytest.raises(SystemExit):
        parse_args(["--clusters-per-call", "3"])


def test_benchmark_uses_per_cluster_first_attempt_rate_not_batch_call_rate():
    modules = [
        {"batch_stats": {"first_attempt_clusters": 20, "first_attempt_passes": 19}},
        {"batch_stats": {"first_attempt_clusters": 10, "first_attempt_passes": 7}},
    ]

    assert summarize_cluster_passes(modules) == {
        "first_attempt_clusters": 30,
        "first_attempt_passes": 26,
        "first_attempt_cluster_pass_rate": 26 / 30,
    }
