from collections import Counter

from batch_pipeline import run_batch
from tests.batch_helpers import counting_dependencies, make_items


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


def test_three_file_batch_reuses_staged_runtimes_instead_of_legacy_loads(tmp_path):
    staged_counters = Counter()
    legacy_counters = Counter()
    items = make_items(tmp_path, "one.pdf", "two.pdf", "three.pdf")

    result = run_batch(items, dependencies=counting_dependencies(staged_counters))
    _run_legacy_nine_load_simulation(items, legacy_counters)

    assert len(result.outputs) == 3
    assert (staged_counters["qwen_load"], staged_counters["rebel_load"]) == (2, 1)
    assert (legacy_counters["qwen_load"], legacy_counters["rebel_load"]) == (6, 3)


def _run_legacy_nine_load_simulation(items, counters):
    """Simulate the previous per-file Qwen/REBEL/Qwen dependency lifecycle."""
    dependencies = counting_dependencies(counters)
    for item in items:
        with dependencies.qwen_loader(item.args) as backend:
            dependencies.normalize_stage(item, backend)
        with dependencies.rebel_loader(item.args) as runtime:
            dependencies.graph_stage(item, runtime)
        with dependencies.qwen_loader(item.args) as backend:
            dependencies.flashcard_stage(item, backend)
