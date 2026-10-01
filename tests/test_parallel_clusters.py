import json
import re
import threading
from dataclasses import asdict

import pytest

from flashcard_pipeline import FlashcardPipeline, GenerationError, PipelineConfig
from flashcard_types import ModuleIdentity
from tests.factories import cluster_json, graph_facts, make_cards, plan_json


class ConcurrentBackend:
    def __init__(self, shared=None):
        self.shared = shared or {
            "lock": threading.Lock(),
            "barrier": threading.Barrier(5),
            "active": 0,
            "peak": 0,
            "forks": [],
        }
        self.closed = False

    def fork(self, worker_index):
        del worker_index
        child = ConcurrentBackend(self.shared)
        self.shared["forks"].append(child)
        return child

    def close(self):
        self.closed = True

    def complete(self, system, user, *, max_tokens, schema=None):
        del system, max_tokens, schema
        if "Generate exactly 5 cards" not in user:
            return plan_json()
        topic = json.loads(user.split("INPUT JSON:\n", 1)[1])["topic"]
        position = int(re.search(r"Concept (\d+)", topic).group(1))
        with self.shared["lock"]:
            self.shared["active"] += 1
            self.shared["peak"] = max(
                self.shared["peak"], self.shared["active"]
            )
        try:
            self.shared["barrier"].wait(timeout=5)
            return cluster_json(position)
        finally:
            with self.shared["lock"]:
                self.shared["active"] -= 1


def test_five_clusters_run_concurrently_and_return_in_plan_order():
    backend = ConcurrentBackend()
    pipeline = FlashcardPipeline(
        backend,
        PipelineConfig(cluster_workers=5, final_review=False),
        progress=lambda _message: None,
    )

    clusters = pipeline.run(ModuleIdentity("CPE0021", "1"), graph_facts())

    assert [cluster.concept.name for cluster in clusters] == [
        f"Concept {i} topic{i} alpha{i} beta{i}" for i in range(1, 21)
    ]
    assert backend.shared["peak"] == 5
    assert len(backend.shared["forks"]) == 4
    assert all(child.closed for child in backend.shared["forks"])
    assert backend.closed is False
    assert pipeline.rejection_stats["attempts"] == 21


def test_cluster_worker_count_must_be_between_one_and_twenty():
    with pytest.raises(ValueError, match="cluster_workers"):
        PipelineConfig(cluster_workers=0)
    PipelineConfig(cluster_workers=6)
    with pytest.raises(ValueError, match="cluster_workers"):
        PipelineConfig(cluster_workers=21)


def test_parallel_progress_reports_completed_count():
    backend = ConcurrentBackend()
    messages = []
    pipeline = FlashcardPipeline(
        backend,
        PipelineConfig(cluster_workers=5, final_review=False),
        progress=messages.append,
    )

    pipeline.run(ModuleIdentity("CPE0021", "1"), graph_facts())

    completed = [message for message in messages if message.startswith("Completed cluster ")]
    assert len(completed) == 20
    assert completed[0].startswith("Completed cluster 1/20:")
    assert completed[-1].startswith("Completed cluster 20/20:")


def test_parallel_generation_reports_measured_cluster_rate(monkeypatch):
    moments = iter((0.0, 240.0))
    monkeypatch.setattr("flashcard_pipeline.monotonic", lambda: next(moments))
    messages = []
    pipeline = FlashcardPipeline(
        ConcurrentBackend(),
        PipelineConfig(cluster_workers=5, final_review=False),
        progress=messages.append,
    )

    pipeline.run(ModuleIdentity("CPE0021", "1"), graph_facts())

    assert "Cluster generation: 20 clusters in 240.0s (5.00/min, 5 workers)." in messages


def test_failed_parallel_generation_closes_every_fork():
    class FailingBackend(ConcurrentBackend):
        def fork(self, worker_index):
            del worker_index
            child = FailingBackend(self.shared)
            self.shared["forks"].append(child)
            return child

        def complete(self, system, user, *, max_tokens, schema=None):
            del system, max_tokens, schema
            match = re.search(r"Concept (\d+) topic", user)
            if match is None:
                return plan_json()
            if match.group(1) == "1":
                return '{"cards": []}'
            return cluster_json(int(match.group(1)))

    backend = FailingBackend()
    pipeline = FlashcardPipeline(
        backend,
        PipelineConfig(cluster_workers=5, final_review=False),
        progress=lambda _message: None,
    )

    with pytest.raises(GenerationError, match="concept 1"):
        pipeline.run(ModuleIdentity("CPE0021", "1"), graph_facts())

    assert len(backend.shared["forks"]) == 4
    assert all(child.closed for child in backend.shared["forks"])
    assert backend.closed is False


def test_model_load_failure_names_the_worker_and_closes_loaded_forks():
    class FailingForkBackend(ConcurrentBackend):
        def fork(self, worker_index):
            if worker_index == 2:
                raise MemoryError("not enough memory")
            return super().fork(worker_index)

    backend = FailingForkBackend()
    pipeline = FlashcardPipeline(
        backend,
        PipelineConfig(cluster_workers=5, final_review=False),
        progress=lambda _message: None,
    )

    with pytest.raises(GenerationError, match="worker 3/5.*memory"):
        pipeline.run(ModuleIdentity("CPE0021", "1"), graph_facts())

    assert len(backend.shared["forks"]) == 1
    assert backend.shared["forks"][0].closed is True


def test_parallel_clusters_still_repair_cross_cluster_duplicates():
    class DuplicateBackend(ConcurrentBackend):
        def fork(self, worker_index):
            del worker_index
            child = DuplicateBackend(self.shared)
            self.shared["forks"].append(child)
            return child

        def complete(self, system, user, *, max_tokens, schema=None):
            if user.startswith("Paraphrase exactly one flagged flashcard"):
                return json.dumps({"cards": [asdict(make_cards(2, revision=1)[0])]})
            response = super().complete(
                system, user, max_tokens=max_tokens, schema=schema
            )
            if "Generate exactly 5 cards" in user and '"topic": "Concept 2 ' in user:
                payload = json.loads(response)
                payload["cards"][0]["question"] = make_cards(1)[0].question
                return json.dumps(payload)
            return response

    backend = DuplicateBackend()
    pipeline = FlashcardPipeline(
        backend,
        PipelineConfig(cluster_workers=5, final_review=False),
        progress=lambda _message: None,
    )

    clusters = pipeline.run(ModuleIdentity("CPE0021", "1"), graph_facts())

    assert clusters[1].cards[0].question == make_cards(2, revision=1)[0].question
    assert clusters[1].cards[1:] == make_cards(2)[1:]
