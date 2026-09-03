from collections import Counter
from contextlib import contextmanager
from time import perf_counter, sleep

from batch_pipeline import BatchDependencies, run_batch
from tests.batch_helpers import counting_dependencies, make_items, materialize


def test_three_file_batch_loads_qwen_twice_and_rebel_once(tmp_path):
    counters = Counter()
    items = make_items(tmp_path, "one.pdf", "two.pdf", "three.pdf")

    result = run_batch(items, dependencies=counting_dependencies(counters))

    assert len(result.outputs) == 3
    assert counters["qwen_load"] == 2
    assert counters["rebel_load"] == 1
    assert counters["normalize"] == 3
    assert counters["graph"] == 3
    assert counters["flashcards"] == 3


def test_three_file_batch_reduces_synthetic_dependency_load_time(tmp_path):
    """Synthetic loader-delay comparison; this does not measure model inference."""
    delay_seconds = 0.01
    staged_counters = Counter()
    legacy_counters = Counter()
    items = make_items(tmp_path, "one.pdf", "two.pdf", "three.pdf")

    result = run_batch(
        items,
        dependencies=_delayed_dependencies(staged_counters, delay_seconds),
    )

    _run_legacy_nine_load_simulation(
        items, legacy_counters, delay_seconds
    )

    assert len(result.outputs) == 3
    assert (staged_counters["qwen_load"], staged_counters["rebel_load"]) == (2, 1)
    assert (legacy_counters["qwen_load"], legacy_counters["rebel_load"]) == (6, 3)
    assert staged_counters["load_seconds"] < legacy_counters["load_seconds"]


def _delayed_dependencies(counters, delay_seconds):
    @contextmanager
    def qwen_loader(args):
        counters["qwen_load"] += 1
        started = perf_counter()
        sleep(delay_seconds)
        counters["load_seconds"] += perf_counter() - started
        yield object()

    @contextmanager
    def rebel_loader(args):
        counters["rebel_load"] += 1
        started = perf_counter()
        sleep(delay_seconds)
        counters["load_seconds"] += perf_counter() - started
        yield object()

    def normalize(item, backend):
        counters["normalize"] += 1
        materialize(item, "normalize")

    def graph(item, runtime):
        counters["graph"] += 1
        materialize(item, "graph")

    def flashcards(item, backend):
        counters["flashcards"] += 1
        materialize(item, "flashcards")

    return BatchDependencies(
        qwen_loader=qwen_loader,
        rebel_loader=rebel_loader,
        normalize_stage=normalize,
        graph_stage=graph,
        flashcard_stage=flashcards,
        monotonic=perf_counter,
    )


def _run_legacy_nine_load_simulation(items, counters, delay_seconds):
    """Simulate the previous per-file Qwen/REBEL/Qwen dependency lifecycle."""
    dependencies = _delayed_dependencies(counters, delay_seconds)
    for item in items:
        with dependencies.qwen_loader(item.args) as backend:
            dependencies.normalize_stage(item, backend)
        with dependencies.rebel_loader(item.args) as runtime:
            dependencies.graph_stage(item, runtime)
        with dependencies.qwen_loader(item.args) as backend:
            dependencies.flashcard_stage(item, backend)
