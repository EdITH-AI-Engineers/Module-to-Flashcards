from collections import Counter
from contextlib import contextmanager
import json
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
from flashcard_csv import render_module
from flashcard_types import ModuleIdentity
from tests.factories import valid_clusters


def test_batch_groups_stages_and_reuses_each_runtime(tmp_path):
    items = make_items(tmp_path, "one.txt", "two.txt")
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

    assert result.outputs == tuple(
        path for item in items for path in item.paths.flashcard_parts
    )
    assert not result.errors
    assert [event[0] for event in events] == [
        "ingest",
        "ingest",
        "rebel-load",
        "graph",
        "graph",
        "rebel-release",
        "qwen-load",
        "flashcards",
        "flashcards",
        "qwen-release",
    ]
    assert events[0][2] is None and events[1][2] is None
    assert events[3][2] is events[4][2]
    assert events[7][2] is events[8][2]


def test_batch_reports_module_progress_transitions(tmp_path):
    item = make_items(tmp_path, "one.txt")[0]
    progress = []

    result = run_batch(
        (item,),
        dependencies=counting_dependencies(Counter()),
        progress=progress.append,
    )

    assert result.outputs == item.paths.flashcard_parts
    assert progress[0].state == "queued"
    assert progress[0].stage == "ingest"
    assert any(
        event.state == "processing" and event.stage == "graph"
        for event in progress
    )
    assert any(
        event.state == "processing" and event.stage == "flashcards"
        for event in progress
    )
    assert progress[-1].state == "completed"
    assert progress[-1].stage == "complete"
    assert progress[-1].progress_percent == 100


def test_reused_batch_reports_immediate_completion(tmp_path):
    item = make_items(tmp_path, "one.txt")[0]
    for stage in ("ingest", "graph", "flashcards"):
        materialize(item, stage)
    batch_pipeline._write_manifest(item)
    progress = []

    result = run_batch(
        (item,),
        dependencies=fake_dependencies([], None, None),
        progress=progress.append,
    )

    assert result.outputs == item.paths.flashcard_parts
    assert len(progress) == 1
    assert progress[0].state == "completed"
    assert progress[0].message == "Reused completed module artifacts."


def test_batch_reuses_valid_artifacts_without_loading_models(tmp_path):
    items = make_items(tmp_path, "one.txt", "two.txt")
    for item in items:
        for stage in ("ingest", "graph", "flashcards"):
            materialize(item, stage)
        batch_pipeline._write_manifest(item)

    @contextmanager
    def forbidden_loader(args):
        raise AssertionError("valid artifacts must not load a model")
        yield

    dependencies = fake_dependencies([], forbidden_loader, forbidden_loader)

    result = run_batch(items, dependencies=dependencies)

    assert result.outputs == tuple(path for item in items for path in item.paths.flashcard_parts)
    assert not result.errors


def test_failed_module_does_not_stop_other_module(tmp_path):
    items = make_items(tmp_path, "one.txt", "two.txt")
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
            fail_ingest="one.txt",
        ),
    )

    assert result.outputs == items[1].paths.flashcard_parts
    assert result.errors == (
        {"file": "one.txt", "error": "ingest failed"},
    )
    assert ("flashcards", "two.txt") in [event[:2] for event in events]
    assert ("graph", "one.txt") not in [event[:2] for event in events]


def test_timeout_counts_only_each_modules_active_work(tmp_path):
    items = make_items(tmp_path, "one.txt", "two.txt")
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

    assert result.outputs == items[1].paths.flashcard_parts
    assert result.errors == (
        {
            "file": "one.txt",
            "error": "module exceeded 5-second active-processing timeout",
        },
    )
    assert ("flashcards", "one.txt") in [event[:2] for event in events]


def test_upstream_recomputation_forces_downstream_recomputation(tmp_path):
    items = make_items(tmp_path, "one.txt")
    item = items[0]
    for stage in ("ingest", "graph", "flashcards"):
        materialize(item, stage)
    item.paths.structured_text.write_text("invalid", encoding="utf-8")
    counters = Counter()

    result = run_batch(items, dependencies=counting_dependencies(counters))

    assert result.outputs == item.paths.flashcard_parts
    assert counters == {
        "qwen_load": 1,
        "ingest": 1,
        "rebel_load": 1,
        "graph": 1,
        "flashcards": 1,
    }


def test_invalid_stage_artifact_removes_only_that_item(tmp_path):
    items = make_items(tmp_path, "one.txt", "two.txt")

    @contextmanager
    def loader(args):
        yield object()

    def ingest(item, backend):
        if item.filename == "two.txt":
            materialize(item, "ingest")

    dependencies = BatchDependencies(
        qwen_loader=loader,
        rebel_loader=loader,
        ingest_stage=ingest,
        graph_stage=lambda item, runtime: materialize(item, "graph"),
        flashcard_stage=lambda item, backend: materialize(item, "flashcards"),
        monotonic=time.monotonic,
    )

    result = run_batch(items, dependencies=dependencies)

    assert result.outputs == items[1].paths.flashcard_parts
    assert result.errors[0]["file"] == "one.txt"
    assert "ingest stage did not create a valid artifact" in result.errors[0]["error"]


def test_mixed_runtime_configuration_is_rejected_before_loading(tmp_path):
    items = make_items(tmp_path, "one.txt", "two.txt")
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
            {"file": "one.txt", "error": error},
            {"file": "two.txt", "error": error},
        ),
    )


def test_production_loaders_release_their_owned_runtimes(monkeypatch, tmp_path):
    items = make_items(tmp_path, "one.txt")
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


def test_production_loaders_disable_network_for_portable_items(monkeypatch, tmp_path):
    item = make_items(tmp_path, "one.txt")[0]
    item.args.portable = True
    item.args.rebel_model = tmp_path / "models" / "rebel-large"
    qwen_calls = []
    rebel_calls = []
    backend = type("Backend", (), {"close": lambda self: None})()
    runtime = object()

    monkeypatch.setattr(
        batch_pipeline,
        "ensure_model",
        lambda model_dir, **kwargs: qwen_calls.append((model_dir, kwargs))
        or tmp_path / "qwen.gguf",
    )
    monkeypatch.setattr(batch_pipeline, "LocalQwenBackend", lambda *a, **k: backend)
    monkeypatch.setattr(
        batch_pipeline,
        "load_runtime",
        lambda model, device, **kwargs: rebel_calls.append((model, device, kwargs))
        or runtime,
    )
    monkeypatch.setattr(batch_pipeline, "release_runtime", lambda value: None)

    with batch_pipeline.PRODUCTION_DEPENDENCIES.qwen_loader(item.args):
        pass
    with batch_pipeline.PRODUCTION_DEPENDENCIES.rebel_loader(item.args):
        pass

    assert qwen_calls == [(item.args.model_dir, {"allow_download": False})]
    assert rebel_calls == [
        (item.args.rebel_model, item.args.kg_device, {"local_files_only": True})
    ]


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
    items = make_items(tmp_path, "one.txt", "two.txt")
    events = []
    if failing_loader == "qwen":
        for item in items:
            materialize(item, "ingest")
            materialize(item, "graph")
    else:
        for item in items:
            materialize(item, "ingest")
        materialize(items[1], "graph")
    for item in items:
        batch_pipeline._write_manifest(item)

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
            {"file": "one.txt", "error": error},
            {"file": "two.txt", "error": error},
        ),
    )


def test_errors_are_returned_in_input_order_across_stages(tmp_path):
    items = make_items(tmp_path, "one.txt", "two.txt")

    @contextmanager
    def loader(args):
        yield object()

    def ingest(item, backend):
        if item.filename == "two.txt":
            raise RuntimeError("second failed ingest")
        materialize(item, "ingest")

    def graph(item, runtime):
        raise RuntimeError("first failed graph generation")

    dependencies = BatchDependencies(
        qwen_loader=loader,
        rebel_loader=loader,
        ingest_stage=ingest,
        graph_stage=graph,
        flashcard_stage=lambda item, backend: materialize(item, "flashcards"),
        monotonic=time.monotonic,
    )

    result = run_batch(items, dependencies=dependencies)

    assert result.errors == (
        {"file": "one.txt", "error": "first failed graph generation"},
        {"file": "two.txt", "error": "second failed ingest"},
    )


def test_artifact_validation_time_is_excluded_from_active_budget(
    tmp_path, monkeypatch
):
    items = make_items(tmp_path, "one.txt")
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
    original_ingest = dependencies.ingest_stage

    def ingest(item, backend):
        now[0] += 4
        original_ingest(item, backend)

    dependencies = BatchDependencies(
        qwen_loader=dependencies.qwen_loader,
        rebel_loader=dependencies.rebel_loader,
        ingest_stage=ingest,
        graph_stage=dependencies.graph_stage,
        flashcard_stage=dependencies.flashcard_stage,
        monotonic=dependencies.monotonic,
    )
    monkeypatch.setattr(batch_pipeline, "_valid_structured_text", slow_validator)

    result = run_batch(items, dependencies=dependencies, timeout_seconds=5)

    assert result.outputs == items[0].paths.flashcard_parts
    assert not result.errors


def test_exact_timeout_boundary_stops_before_the_next_stage(tmp_path):
    items = make_items(tmp_path, "one.txt")
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

    assert "rebel" in events
    assert "qwen" not in events
    assert result.errors == (
        {
            "file": "one.txt",
            "error": "module exceeded 5-second active-processing timeout",
        },
    )


def test_zero_batch_timeout_is_disabled_without_reading_the_clock(tmp_path):
    items = make_items(tmp_path, "one.txt")

    @contextmanager
    def loader(args):
        yield object()

    result = run_batch(
        items,
        dependencies=fake_dependencies(
            [],
            loader,
            loader,
            monotonic=lambda: pytest.fail("zero timeout must not set a deadline"),
        ),
        timeout_seconds=0,
    )

    assert result.outputs == items[0].paths.flashcard_parts
    assert not result.errors


def test_negative_batch_timeout_fails_every_item_before_loading_models(tmp_path):
    items = make_items(tmp_path, "one.txt", "two.txt")

    @contextmanager
    def forbidden_loader(args):
        pytest.fail("negative timeout must fail before model loading")
        yield object()

    result = run_batch(
        items,
        dependencies=fake_dependencies([], forbidden_loader, forbidden_loader),
        timeout_seconds=-1,
    )

    assert result.errors == (
        {"file": "one.txt", "error": "module timeout must not be negative"},
        {"file": "two.txt", "error": "module timeout must not be negative"},
    )


def test_final_flashcard_stage_exhausting_the_budget_is_not_successful(tmp_path):
    items = make_items(tmp_path, "one.txt")
    events = []
    clock_values = iter((0, 0, 0, 5))

    @contextmanager
    def loader(args):
        yield object()

    result = run_batch(
        items,
        dependencies=fake_dependencies(
            events,
            loader,
            loader,
            monotonic=lambda: next(clock_values),
        ),
        timeout_seconds=5,
    )

    assert ("flashcards", "one.txt") in [event[:2] for event in events]
    assert result.outputs == ()
    assert result.errors == (
        {
            "file": "one.txt",
            "error": "module exceeded 5-second active-processing timeout",
        },
    )


def test_force_partial_failure_removes_stale_downstream_artifacts_for_resume(tmp_path):
    item = make_items(tmp_path, "one.txt")[0]
    for stage in ("ingest", "graph", "flashcards"):
        materialize(item, stage)
    manifest = item.paths.workspace / "batch_reuse_manifest.json"
    manifest.write_text("old manifest", encoding="utf-8")
    item.args.force = True

    @contextmanager
    def loader(args):
        yield object()

    first = run_batch(
        (item,),
        dependencies=BatchDependencies(
            qwen_loader=loader,
            rebel_loader=loader,
            ingest_stage=lambda current, backend: materialize(current, "ingest"),
            graph_stage=lambda current, runtime: (_ for _ in ()).throw(
                RuntimeError("graph failed")
            ),
            flashcard_stage=lambda current, backend: materialize(current, "flashcards"),
            monotonic=time.monotonic,
        ),
    )

    assert first.errors == ({"file": "one.txt", "error": "graph failed"},)
    assert not item.paths.graph_json.exists()
    assert not item.paths.triples_csv.exists()
    assert not any(path.exists() for path in item.paths.flashcard_parts)
    assert not item.paths.flashcard_receipt.exists()
    assert not manifest.exists()

    item.args.force = False
    counters = Counter()
    resumed = run_batch((item,), dependencies=counting_dependencies(counters))

    assert resumed.outputs == item.paths.flashcard_parts
    assert counters["ingest"] == counters["graph"] == counters["flashcards"] == 1


def test_matching_batch_reuse_manifest_skips_all_model_work(tmp_path):
    item = make_items(tmp_path, "one.txt")[0]
    first_counters = Counter()

    first = run_batch((item,), dependencies=counting_dependencies(first_counters))

    assert first.outputs == item.paths.flashcard_parts
    assert (item.paths.workspace / "batch_reuse_manifest.json").is_file()

    @contextmanager
    def forbidden_loader(args):
        pytest.fail("matching manifest and artifacts must be reused")
        yield object()

    resumed = run_batch(
        (item,), dependencies=fake_dependencies([], forbidden_loader, forbidden_loader)
    )

    assert resumed.outputs == item.paths.flashcard_parts


def _legacy_item(tmp_path):
    item = make_items(tmp_path, "one.txt")[0]
    for stage in ("ingest", "graph", "flashcards"):
        materialize(item, stage)
    for path in (*item.paths.flashcard_parts, item.paths.flashcard_receipt):
        path.unlink()
    item.paths.flashcards.write_text(
        render_module(ModuleIdentity("CPE0021", "1"), valid_clusters()),
        encoding="utf-8-sig",
    )
    item.args.cluster_workers = 1
    batch_pipeline._write_manifest(item)
    manifest_path = item.paths.workspace / "batch_reuse_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("output_format")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    item.args.cluster_workers = "auto"
    return item


def test_batch_converts_matching_legacy_output_without_qwen(tmp_path):
    item = _legacy_item(tmp_path)

    result = run_batch(
        (item,),
        dependencies=fake_dependencies(
            [],
            lambda args: pytest.fail("Qwen must not load"),
            lambda args: pytest.fail("REBEL must not load"),
        ),
    )

    assert result.outputs == item.paths.flashcard_parts
    assert not result.errors
    assert item.paths.flashcards.is_file()


def test_batch_changed_source_does_not_migrate_legacy_output(tmp_path):
    item = _legacy_item(tmp_path)
    item.args.input.write_text(
        item.args.input.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    counters = Counter()

    result = run_batch((item,), dependencies=counting_dependencies(counters))

    assert result.outputs == item.paths.flashcard_parts
    assert counters["qwen_load"] == 1


def test_corrupt_legacy_output_regenerates(tmp_path):
    item = _legacy_item(tmp_path)
    item.paths.flashcards.write_text("bad old data", encoding="utf-8")
    counters = Counter()

    result = run_batch((item,), dependencies=counting_dependencies(counters))

    assert result.outputs == item.paths.flashcard_parts
    assert counters["qwen_load"] == 1


def test_retained_legacy_backup_never_replaces_newer_batch_pair(tmp_path):
    item = _legacy_item(tmp_path)
    assert run_batch((item,), dependencies=fake_dependencies([], None, None)).outputs
    item.args.input.write_text(
        item.args.input.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    assert run_batch((item,), dependencies=counting_dependencies(Counter())).outputs
    item.paths.flashcard_parts[1].unlink()
    counters = Counter()

    result = run_batch((item,), dependencies=counting_dependencies(counters))

    assert result.outputs == item.paths.flashcard_parts
    assert counters["qwen_load"] == 1


def test_explicit_worker_change_prevents_legacy_migration(tmp_path):
    item = _legacy_item(tmp_path)
    item.args.cluster_workers = 12
    counters = Counter()

    result = run_batch((item,), dependencies=counting_dependencies(counters))

    assert result.outputs == item.paths.flashcard_parts
    assert counters["qwen_load"] == 1


@pytest.mark.parametrize(
    "change",
    (
        lambda item: item.args.input.write_text(
            item.args.input.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        ),
        lambda item: setattr(item.args, "seed", 99),
    ),
    ids=("replaced-source", "changed-setting"),
)
def test_changed_batch_reuse_identity_or_source_forces_all_stages(tmp_path, change):
    item = make_items(tmp_path, "one.txt")[0]
    assert run_batch((item,), dependencies=counting_dependencies(Counter())).outputs
    change(item)
    counters = Counter()

    result = run_batch((item,), dependencies=counting_dependencies(counters))

    assert result.outputs == item.paths.flashcard_parts
    assert counters["ingest"] == counters["graph"] == counters["flashcards"] == 1


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


def test_production_ingest_adapter_stages_validated_structured_text(tmp_path):
    item = make_items(tmp_path, "one.txt")[0]
    output = batch_pipeline.PRODUCTION_DEPENDENCIES.ingest_stage(item, None)

    assert output == item.paths.structured_text
    assert output.read_text(encoding="utf-8") == item.args.input.read_text(
        encoding="utf-8"
    )


def test_production_graph_adapter_translates_fast_settings(tmp_path, monkeypatch):
    item = make_items(tmp_path, "one.txt")[0]
    runtime = object()
    captured = {}

    def fake_run(args, *, runtime):
        captured.update(args=vars(args), runtime=runtime)
        return args.output_dir / "knowledge_graph.json", args.output_dir / "triples.csv"

    monkeypatch.setattr(batch_pipeline.text_extractor, "run", fake_run)

    batch_pipeline.PRODUCTION_DEPENDENCIES.graph_stage(item, runtime)

    assert captured["runtime"] is runtime
    assert captured["args"]["input"] == item.paths.structured_text
    assert captured["args"]["output_dir"] == item.paths.unchecked_graph_dir
    assert captured["args"]["device"] == "auto"
    assert captured["args"]["batch_size"] == 1
    assert captured["args"]["num_beams"] == 1


def test_production_flashcard_adapter_translates_review_settings(
    tmp_path, monkeypatch
):
    item = make_items(tmp_path, "one.txt")[0]
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
    assert captured["args"]["unchecked_graph"] == item.paths.unchecked_graph_json
    assert captured["args"]["output"] == item.paths.flashcards
    assert captured["args"]["course_corpus"] == (
        item.paths.flashcards.parent / "course_corpus.json"
    )
    assert captured["args"]["max_retries"] == item.args.attempts
    assert captured["args"]["final_review"] is False
    assert captured["args"]["smoke_test"] is False
