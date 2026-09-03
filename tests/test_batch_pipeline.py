from collections import Counter
from contextlib import contextmanager
import time

import pytest

import batch_pipeline
from batch_pipeline import BatchDependencies, BatchResult, run_batch
from tests.batch_helpers import (
    counting_dependencies,
    fake_dependencies,
    make_items,
    materialize,
)


def test_batch_groups_stages_and_reuses_each_runtime(tmp_path):
    items = make_items(tmp_path, "one.pdf", "two.pdf")
    events = []

    @contextmanager
    def qwen_loader(args):
        phase_backend = object()
        events.append(("qwen-load", phase_backend))
        yield phase_backend
        events.append(("qwen-release", phase_backend))

    @contextmanager
    def rebel_loader(args):
        runtime = object()
        events.append(("rebel-load", runtime))
        yield runtime
        events.append(("rebel-release", runtime))

    result = run_batch(
        items,
        dependencies=fake_dependencies(
            events,
            qwen_loader=qwen_loader,
            rebel_loader=rebel_loader,
        ),
    )

    assert len(result.outputs) == 2
    assert not result.errors
    assert [event[0] for event in events] == [
        "qwen-load",
        "normalize",
        "normalize",
        "qwen-release",
        "rebel-load",
        "graph",
        "graph",
        "rebel-release",
        "qwen-load",
        "flashcards",
        "flashcards",
        "qwen-release",
    ]
    assert events[1][2] is events[2][2]
    assert events[5][2] is events[6][2]
    assert events[9][2] is events[10][2]


def test_batch_reuses_valid_artifacts_without_loading_models(tmp_path):
    items = make_items(tmp_path, "one.pdf", "two.pdf")
    for item in items:
        for stage in ("normalize", "graph", "flashcards"):
            materialize(item, stage)

    @contextmanager
    def forbidden_loader(args):
        raise AssertionError("valid artifacts must not load a model")
        yield

    dependencies = fake_dependencies([], forbidden_loader, forbidden_loader)

    result = run_batch(items, dependencies=dependencies)

    assert result.outputs == tuple(item.paths.flashcards for item in items)
    assert not result.errors


def test_failed_module_does_not_stop_other_module(tmp_path):
    items = make_items(tmp_path, "one.pdf", "two.pdf")
    events = []

    @contextmanager
    def loader(args):
        yield object()

    result = run_batch(
        items,
        dependencies=fake_dependencies(
            events,
            loader,
            loader,
            fail_normalize="one.pdf",
        ),
    )

    assert result.outputs == (items[1].paths.flashcards,)
    assert result.errors == (
        {"pdf": "one.pdf", "error": "normalization failed"},
    )
    assert ("flashcards", "two.pdf") in [event[:2] for event in events]
    assert ("graph", "one.pdf") not in [event[:2] for event in events]


def test_timeout_counts_only_each_modules_active_work(tmp_path):
    items = make_items(tmp_path, "one.pdf", "two.pdf")
    events = []
    clock_values = iter([0, 4, 100, 101, 200, 202, 300, 301, 400, 401])

    @contextmanager
    def loader(args):
        yield object()

    dependencies = fake_dependencies(
        events,
        loader,
        loader,
        monotonic=lambda: next(clock_values),
    )

    result = run_batch(items, dependencies=dependencies, timeout_seconds=5)

    assert result.outputs == (items[1].paths.flashcards,)
    assert result.errors == (
        {
            "pdf": "one.pdf",
            "error": "module exceeded 5-second active-processing timeout",
        },
    )
    assert ("flashcards", "one.pdf") not in [event[:2] for event in events]


def test_upstream_recomputation_forces_downstream_recomputation(tmp_path):
    items = make_items(tmp_path, "one.pdf")
    item = items[0]
    for stage in ("normalize", "graph", "flashcards"):
        materialize(item, stage)
    item.paths.structured_text.write_text("invalid", encoding="utf-8")
    counters = Counter()

    result = run_batch(items, dependencies=counting_dependencies(counters))

    assert result.outputs == (item.paths.flashcards,)
    assert counters == {
        "qwen_load": 2,
        "normalize": 1,
        "rebel_load": 1,
        "graph": 1,
        "flashcards": 1,
    }


def test_invalid_stage_artifact_removes_only_that_item(tmp_path):
    items = make_items(tmp_path, "one.pdf", "two.pdf")

    @contextmanager
    def loader(args):
        yield object()

    def normalize(item, backend):
        if item.filename == "two.pdf":
            materialize(item, "normalize")

    dependencies = BatchDependencies(
        qwen_loader=loader,
        rebel_loader=loader,
        normalize_stage=normalize,
        graph_stage=lambda item, runtime: materialize(item, "graph"),
        flashcard_stage=lambda item, backend: materialize(item, "flashcards"),
        monotonic=time.monotonic,
    )

    result = run_batch(items, dependencies=dependencies)

    assert result.outputs == (items[1].paths.flashcards,)
    assert result.errors[0]["pdf"] == "one.pdf"
    assert "normalize stage did not create a valid artifact" in result.errors[0]["error"]


def test_mixed_runtime_configuration_is_rejected_before_loading(tmp_path):
    items = make_items(tmp_path, "one.pdf", "two.pdf")
    items[1].args.n_ctx = 4096

    @contextmanager
    def forbidden_loader(args):
        raise AssertionError("incompatible configuration must not load a model")
        yield

    result = run_batch(
        items,
        dependencies=fake_dependencies([], forbidden_loader, forbidden_loader),
    )

    error = "batch items must share runtime configuration: --n-ctx"
    assert result == BatchResult(
        outputs=(),
        errors=(
            {"pdf": "one.pdf", "error": error},
            {"pdf": "two.pdf", "error": error},
        ),
    )


def test_production_loaders_release_their_owned_runtimes(monkeypatch, tmp_path):
    items = make_items(tmp_path, "one.pdf")
    backend = type("Backend", (), {"closed": False})()
    backend.close = lambda: setattr(backend, "closed", True)
    runtime = object()
    released = []

    monkeypatch.setattr(batch_pipeline, "ensure_model", lambda model_dir: tmp_path / "qwen.gguf")
    monkeypatch.setattr(batch_pipeline, "LocalQwenBackend", lambda *a, **k: backend)
    monkeypatch.setattr(batch_pipeline, "load_runtime", lambda *a, **k: runtime)
    monkeypatch.setattr(batch_pipeline, "release_runtime", released.append)

    with pytest.raises(RuntimeError, match="stage failed"):
        with batch_pipeline.PRODUCTION_DEPENDENCIES.qwen_loader(items[0].args):
            raise RuntimeError("stage failed")
    with pytest.raises(RuntimeError, match="stage failed"):
        with batch_pipeline.PRODUCTION_DEPENDENCIES.rebel_loader(items[0].args):
            raise RuntimeError("stage failed")

    assert backend.closed is True
    assert released == [runtime]


@pytest.mark.parametrize(
    ("failing_loader", "failure_point"),
    (
        ("qwen", "entry"),
        ("qwen", "exit"),
        ("rebel", "entry"),
        ("rebel", "exit"),
    ),
)
def test_loader_failure_aborts_all_remaining_heavy_stages(
    tmp_path, failing_loader, failure_point
):
    items = make_items(tmp_path, "one.pdf", "two.pdf")
    events = []
    if failing_loader == "qwen":
        materialize(items[1], "normalize")
    else:
        for item in items:
            materialize(item, "normalize")
        materialize(items[1], "graph")

    @contextmanager
    def qwen_loader(args):
        events.append("qwen-entry")
        if failing_loader == "qwen" and failure_point == "entry":
            raise RuntimeError("qwen entry failed")
        yield object()
        events.append("qwen-exit")
        if failing_loader == "qwen" and failure_point == "exit":
            raise RuntimeError("qwen exit failed")

    @contextmanager
    def rebel_loader(args):
        events.append("rebel-entry")
        if failing_loader == "rebel" and failure_point == "entry":
            raise RuntimeError("rebel entry failed")
        yield object()
        events.append("rebel-exit")
        if failing_loader == "rebel" and failure_point == "exit":
            raise RuntimeError("rebel exit failed")

    result = run_batch(
        items,
        dependencies=fake_dependencies(events, qwen_loader, rebel_loader),
    )

    loader_events = [event for event in events if isinstance(event, str)]
    expected_events = [f"{failing_loader}-entry"]
    if failure_point == "exit":
        expected_events.append(f"{failing_loader}-exit")
    error = f"{failing_loader} {failure_point} failed"
    assert loader_events == expected_events
    assert result == BatchResult(
        outputs=(),
        errors=(
            {"pdf": "one.pdf", "error": error},
            {"pdf": "two.pdf", "error": error},
        ),
    )


def test_errors_are_returned_in_input_order_across_stages(tmp_path):
    items = make_items(tmp_path, "one.pdf", "two.pdf")

    @contextmanager
    def loader(args):
        yield object()

    def normalize(item, backend):
        if item.filename == "two.pdf":
            raise RuntimeError("second failed normalization")
        materialize(item, "normalize")

    def graph(item, runtime):
        raise RuntimeError("first failed graph generation")

    dependencies = BatchDependencies(
        qwen_loader=loader,
        rebel_loader=loader,
        normalize_stage=normalize,
        graph_stage=graph,
        flashcard_stage=lambda item, backend: materialize(item, "flashcards"),
        monotonic=time.monotonic,
    )

    result = run_batch(items, dependencies=dependencies)

    assert result.errors == (
        {"pdf": "one.pdf", "error": "first failed graph generation"},
        {"pdf": "two.pdf", "error": "second failed normalization"},
    )


def test_artifact_validation_time_is_excluded_from_active_budget(
    tmp_path, monkeypatch
):
    items = make_items(tmp_path, "one.pdf")
    now = [0.0]
    original_validator = batch_pipeline._valid_structured_text

    def slow_validator(path):
        result = original_validator(path)
        if path.is_file():
            now[0] += 10
        return result

    @contextmanager
    def loader(args):
        yield object()

    dependencies = fake_dependencies(
        [],
        loader,
        loader,
        monotonic=lambda: now[0],
    )
    original_normalize = dependencies.normalize_stage

    def normalize(item, backend):
        now[0] += 4
        original_normalize(item, backend)

    dependencies = BatchDependencies(
        qwen_loader=dependencies.qwen_loader,
        rebel_loader=dependencies.rebel_loader,
        normalize_stage=normalize,
        graph_stage=dependencies.graph_stage,
        flashcard_stage=dependencies.flashcard_stage,
        monotonic=dependencies.monotonic,
    )
    monkeypatch.setattr(batch_pipeline, "_valid_structured_text", slow_validator)

    result = run_batch(items, dependencies=dependencies, timeout_seconds=5)

    assert result.outputs == (items[0].paths.flashcards,)
    assert not result.errors


def test_exact_timeout_boundary_stops_before_the_next_stage(tmp_path):
    items = make_items(tmp_path, "one.pdf")
    events = []
    clock_values = iter([0, 5])

    @contextmanager
    def qwen_loader(args):
        events.append("qwen")
        yield object()

    @contextmanager
    def rebel_loader(args):
        events.append("rebel")
        yield object()

    result = run_batch(
        items,
        dependencies=fake_dependencies(
            events,
            qwen_loader,
            rebel_loader,
            monotonic=lambda: next(clock_values),
        ),
        timeout_seconds=5,
    )

    assert events[0] == "qwen"
    assert "rebel" not in events
    assert result.errors == (
        {
            "pdf": "one.pdf",
            "error": "module exceeded 5-second active-processing timeout",
        },
    )


def test_empty_batch_returns_empty_result_without_loading_models():
    @contextmanager
    def forbidden_loader(args):
        raise AssertionError("an empty batch must not load a model")
        yield

    result = run_batch(
        (),
        dependencies=fake_dependencies([], forbidden_loader, forbidden_loader),
    )

    assert result == BatchResult(outputs=(), errors=())


def test_production_normalize_adapter_translates_output_and_backend(
    tmp_path, monkeypatch
):
    item = make_items(tmp_path, "one.pdf")[0]
    backend = object()
    captured = {}

    def fake_run(args, *, backend):
        captured.update(args=vars(args), backend=backend)
        return args.output

    monkeypatch.setattr(batch_pipeline.slides_pdf_to_txt, "run", fake_run)

    output = batch_pipeline.PRODUCTION_DEPENDENCIES.normalize_stage(item, backend)

    assert output == item.paths.structured_text
    assert captured["backend"] is backend
    assert captured["args"]["pdf"] == item.args.pdf
    assert captured["args"]["output"] == item.paths.structured_text
    assert captured["args"]["max_tokens"] == 2048


def test_production_graph_adapter_translates_fast_settings(tmp_path, monkeypatch):
    item = make_items(tmp_path, "one.pdf")[0]
    runtime = object()
    captured = {}

    def fake_run(args, *, runtime):
        captured.update(args=vars(args), runtime=runtime)
        return args.output_dir / "knowledge_graph.json", args.output_dir / "triples.csv"

    monkeypatch.setattr(batch_pipeline.text_extractor, "run", fake_run)

    batch_pipeline.PRODUCTION_DEPENDENCIES.graph_stage(item, runtime)

    assert captured["runtime"] is runtime
    assert captured["args"]["input"] == item.paths.structured_text
    assert captured["args"]["output_dir"] == item.paths.graph_dir
    assert captured["args"]["device"] == "auto"
    assert captured["args"]["batch_size"] == 1
    assert captured["args"]["num_beams"] == 1


def test_production_flashcard_adapter_translates_review_settings(
    tmp_path, monkeypatch
):
    item = make_items(tmp_path, "one.pdf")[0]
    backend = object()
    captured = {}

    def fake_run(args, *, backend):
        captured.update(args=vars(args), backend=backend)
        return args.output

    monkeypatch.setattr(batch_pipeline.flashcard_generator, "run", fake_run)

    output = batch_pipeline.PRODUCTION_DEPENDENCIES.flashcard_stage(item, backend)

    assert output == item.paths.flashcards
    assert captured["backend"] is backend
    assert captured["args"]["graph"] == item.paths.graph_json
    assert captured["args"]["output"] == item.paths.flashcards
    assert captured["args"]["max_retries"] == item.args.attempts
    assert captured["args"]["final_review"] is False
    assert captured["args"]["smoke_test"] is False
