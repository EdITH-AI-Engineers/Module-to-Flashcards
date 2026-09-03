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
