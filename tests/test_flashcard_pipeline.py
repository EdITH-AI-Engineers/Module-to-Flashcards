from collections import deque
import json

import pytest

from flashcard_pipeline import FlashcardPipeline, GenerationError, PipelineConfig
from flashcard_types import ModuleIdentity
from tests.factories import cluster_json, graph_facts, plan_json


class FakeBackend:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.calls = []

    def complete(self, system, user, *, max_tokens):
        self.calls.append((system, user, max_tokens))
        response = self.responses.popleft()
        if callable(response):
            return response(system, user, max_tokens)
        return response


def empty_review():
    return json.dumps({"issues": []})


def flag_first_cluster(system, user, max_tokens):
    payload = json.loads(user.split("INPUT JSON:\n", 1)[1])
    first_uuid = payload["clusters"][0]["cluster"]
    return json.dumps(
        {
            "issues": [
                {
                    "cluster": first_uuid,
                    "reasons": ["semantic duplication"],
                }
            ]
        }
    )


def test_pipeline_generates_twenty_valid_clusters_without_review():
    backend = FakeBackend(
        [plan_json()] + [cluster_json(index) for index in range(1, 21)]
    )
    pipeline = FlashcardPipeline(backend, PipelineConfig(final_review=False))

    clusters = pipeline.run(ModuleIdentity("CPE0021", "1"), graph_facts())

    assert len(clusters) == 20
    assert len({cluster.cluster for cluster in clusters}) == 20
    assert all(len(cluster.cards) == 5 for cluster in clusters)
    assert len(backend.calls) == 21


def test_pipeline_defaults_use_practical_local_token_budgets():
    config = PipelineConfig()

    assert config.plan_max_tokens == 3072
    assert config.cluster_max_tokens == 1536
    assert config.review_max_tokens == 1024


def test_pipeline_reports_major_generation_stages():
    backend = FakeBackend(
        [plan_json()] + [cluster_json(index) for index in range(1, 21)]
    )
    messages = []
    pipeline = FlashcardPipeline(
        backend,
        PipelineConfig(final_review=False),
        progress=messages.append,
    )

    pipeline.run(ModuleIdentity("CPE0021", "1"), graph_facts())

    assert messages[0] == "Planning 20 concepts..."
    assert "Generating cluster 1/20: Concept 1 topic1 alpha1 beta1" in messages
    assert "Generating cluster 20/20: Concept 20 topic20 alpha20 beta20" in messages
    assert messages[-1] == "Generation complete."


def test_invalid_cluster_is_retried_with_validator_feedback():
    responses = [plan_json(), '{"cards": []}', cluster_json(1)]
    responses.extend(cluster_json(index) for index in range(2, 21))
    backend = FakeBackend(responses)
    pipeline = FlashcardPipeline(
        backend,
        PipelineConfig(max_retries=3, final_review=False),
    )

    pipeline.run(ModuleIdentity("CPE0021", "1"), graph_facts())

    assert "expected exactly 5 cards" in backend.calls[2][1]
    assert "complete replacement" in backend.calls[2][1].lower()


def test_invalid_plan_retry_omits_rejected_bulk_response():
    invalid = json.dumps({"concepts": []})
    backend = FakeBackend(
        [invalid, plan_json()] + [cluster_json(index) for index in range(1, 21)]
    )
    pipeline = FlashcardPipeline(
        backend,
        PipelineConfig(max_retries=3, final_review=False),
    )

    pipeline.run(ModuleIdentity("CPE0021", "1"), graph_facts())

    retry_prompt = backend.calls[1][1]
    assert "expected exactly 20 concepts" in retry_prompt
    assert '"rejected_candidate"' not in retry_prompt


def test_exhausted_retries_do_not_return_partial_results():
    backend = FakeBackend(
        [plan_json(), '{"cards": []}', '{"cards": []}', '{"cards": []}']
    )
    pipeline = FlashcardPipeline(
        backend,
        PipelineConfig(max_retries=3, final_review=False),
    )

    with pytest.raises(GenerationError, match="concept 1"):
        pipeline.run(ModuleIdentity("CPE0021", "1"), graph_facts())

    assert len(backend.calls) == 4


def test_explicit_insufficient_content_stops_without_retries():
    backend = FakeBackend(
        [json.dumps({"insufficient_content": "only ten concepts are supported"})]
    )
    pipeline = FlashcardPipeline(backend, PipelineConfig(final_review=False))

    with pytest.raises(GenerationError, match="more content is required"):
        pipeline.run(ModuleIdentity("CPE0021", "1"), graph_facts())

    assert len(backend.calls) == 1


def test_final_review_uses_five_groups_and_one_global_pass():
    responses = [plan_json()]
    responses.extend(cluster_json(index) for index in range(1, 21))
    responses.extend(empty_review() for _ in range(6))
    backend = FakeBackend(responses)
    pipeline = FlashcardPipeline(backend, PipelineConfig(final_review=True))

    pipeline.run(ModuleIdentity("CPE0021", "1"), graph_facts())

    review_calls = backend.calls[21:]
    assert len(review_calls) == 6
    for _, prompt, _ in review_calls[:5]:
        payload = json.loads(prompt.split("INPUT JSON:\n", 1)[1])
        assert len(payload["clusters"]) == 4
        assert "cards" in payload["clusters"][0]
    global_payload = json.loads(review_calls[-1][1].split("INPUT JSON:\n", 1)[1])
    assert len(global_payload["clusters"]) == 20
    assert "questions" in global_payload["clusters"][0]
    assert "cards" not in global_payload["clusters"][0]


def test_final_review_regenerates_flagged_cluster_and_preserves_uuid():
    responses = [plan_json()]
    responses.extend(cluster_json(index) for index in range(1, 21))
    responses.extend(empty_review() for _ in range(5))
    responses.append(flag_first_cluster)
    responses.append(cluster_json(1, revision=1))
    backend = FakeBackend(responses)
    pipeline = FlashcardPipeline(backend, PipelineConfig(final_review=True))

    clusters = pipeline.run(ModuleIdentity("CPE0021", "1"), graph_facts())

    assert "revision1" in clusters[0].cards[0].question
    assert clusters[0].cluster in backend.calls[-1][1]
    assert "semantic duplication" in backend.calls[-1][1]
