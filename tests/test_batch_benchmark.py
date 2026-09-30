from collections import Counter

from batch_pipeline import run_batch
from tests.batch_helpers import counting_dependencies, make_items


def test_three_file_batch_loads_each_model_once(tmp_path):
    counters = Counter()
    items = make_items(tmp_path, "one.txt", "two.txt", "three.txt")

    result = run_batch(items, dependencies=counting_dependencies(counters))

    assert len(result.outputs) == 3
    assert counters["qwen_load"] == 1
    assert counters["rebel_load"] == 1
    assert counters["ingest"] == 3
    assert counters["graph"] == 3
    assert counters["flashcards"] == 3


def test_three_file_batch_reuses_staged_runtimes_instead_of_legacy_loads(tmp_path):
    staged_counters = Counter()
    legacy_counters = Counter()
    items = make_items(tmp_path, "one.txt", "two.txt", "three.txt")

    result = run_batch(items, dependencies=counting_dependencies(staged_counters))
    _run_legacy_nine_load_simulation(items, legacy_counters)

    assert len(result.outputs) == 3
    assert (staged_counters["qwen_load"], staged_counters["rebel_load"]) == (1, 1)
    assert (legacy_counters["qwen_load"], legacy_counters["rebel_load"]) == (3, 3)


def _run_legacy_nine_load_simulation(items, counters):
    """Simulate loading each inference runtime once per file."""
    dependencies = counting_dependencies(counters)
    for item in items:
        dependencies.ingest_stage(item, None)
        with dependencies.rebel_loader(item.args) as runtime:
            dependencies.graph_stage(item, runtime)
        with dependencies.qwen_loader(item.args) as backend:
            dependencies.flashcard_stage(item, backend)
