from collections import deque
from dataclasses import asdict, replace
import json

import pytest

from flashcard_pipeline import (
    FlashcardPipeline,
    GenerationError,
    PipelineConfig,
    _repair_grounding_errors,
)
from flashcard_types import (
    CompletionTruncatedError,
    FlashcardCluster,
    FlashcardDraft,
    GraphFact,
    ModuleIdentity,
)
from flashcard_validator import parse_cards, validate_cluster
from tests.factories import (
    cluster_json,
    graph_facts,
    make_cards,
    make_concept,
    plan_json,
)


class FakeBackend:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.calls = []
        self.schemas = []

    def complete(self, system, user, *, max_tokens, schema=None):
        self.calls.append((system, user, max_tokens))
        self.schemas.append(schema)
        response = self.responses.popleft()
        if callable(response):
            return response(system, user, max_tokens)
        return response


def empty_review():
    return json.dumps({"issues": []})


def single_card_json(card: FlashcardDraft) -> str:
    return json.dumps({"cards": [asdict(card)]})


def review_payload(user: str) -> dict[str, object]:
    payload_text = user.split("INPUT JSON:\n", 1)[1].lstrip()
    payload, _ = json.JSONDecoder().raw_decode(payload_text)
    return payload


def flag_second_cluster_duplicate(system, user, max_tokens):
    payload = review_payload(user)
    first_uuid = payload["clusters"][0]["cluster"]
    second_uuid = payload["clusters"][1]["cluster"]
    return json.dumps(
        {
            "issues": [
                {
                    "cluster": second_uuid,
                    "reasons": [
                        f"card 1 duplicates cluster {first_uuid} card 1"
                    ],
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
    assert backend.schemas[0]["oneOf"][0]["properties"]["concepts"]["minItems"] == 20
    card_schema = backend.schemas[1]["properties"]["cards"]
    assert card_schema["minItems"] == card_schema["maxItems"] == 5
    card_item = card_schema["items"]
    assert set(card_item["required"]) == {
        "type",
        "question",
        "correct_option",
        "wrong_option_1",
        "wrong_option_2",
        "wrong_option_3",
        "is_true",
        "expalanation",
        "hint",
        "difficulty",
        "assessment_approach",
    }
    assert card_item["additionalProperties"] is False


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

    assert messages[0] == "Planning 20 concepts from 20 grounded lesson facts..."
    assert "Generating cluster 1/20: Concept 1 topic1 alpha1 beta1" in messages
    assert "Generating cluster 20/20: Concept 20 topic20 alpha20 beta20" in messages
    assert "Generation quality: 21 model responses, 0 rejected" in messages[-2]
    assert messages[-1] == "100 flashcards generated (50 + 50)."


def test_pipeline_balances_large_fact_set_across_slides_and_bounds_prompt():
    facts = tuple(
        GraphFact(
            f"e{index}",
            f"Grounded lesson fact {index} with enough detail.",
            slides=((index - 1) % 3 + 1,),
            topic=f"Topic {(index - 1) % 3 + 1}",
        )
        for index in range(1, 61)
    )
    backend = FakeBackend(
        [plan_json()] + [cluster_json(index) for index in range(1, 21)]
    )
    messages = []
    pipeline = FlashcardPipeline(
        backend,
        PipelineConfig(final_review=False),
        progress=messages.append,
    )

    pipeline.run(ModuleIdentity("CPE0021", "1"), facts)

    payload = json.loads(backend.calls[0][1].split("INPUT JSON:\n", 1)[1])
    assert len(payload["graph_facts"]) == 50
    assert {item["slides"][0] for item in payload["graph_facts"]} == {1, 2, 3}
    assert messages[0] == "Planning 20 concepts from 50 grounded lesson facts..."


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
    assert "REJECTED JSON TO CORRECT" not in backend.calls[2][1]
    stats = pipeline.rejection_stats
    assert stats["attempts"] == 22
    assert stats["rejected_attempts"] == 1
    assert stats["rejection_rate"] == 1 / 22
    assert stats["categories"]["schema/structure"] == 1


def test_truncated_cluster_retries_without_embedding_partial_output():
    def truncated(system, user, max_tokens):
        raise CompletionTruncatedError(
            "length limit",
            partial_content="PARTIAL_SENTINEL",
            prompt_tokens=6400,
            completion_tokens=1792,
        )

    responses = [plan_json(), truncated, cluster_json(1)]
    responses.extend(cluster_json(index) for index in range(2, 21))
    backend = FakeBackend(responses)
    pipeline = FlashcardPipeline(
        backend,
        PipelineConfig(max_retries=3, final_review=False),
    )

    clusters = pipeline.run(ModuleIdentity("CPE0021", "1"), graph_facts())

    retry_prompt = backend.calls[2][1]
    assert len(clusters) == 20
    assert "response was truncated before completing the JSON" in retry_prompt
    assert "PARTIAL_SENTINEL" not in retry_prompt
    assert "ORIGINAL REQUEST" not in retry_prompt


def test_context_first_multiple_choice_keeps_relevant_external_distractor():
    cards = list(make_cards(1))
    cards[0] = replace(
        cards[0],
        question=(
            "A learner groups a value by its numerical base. "
            "This example demonstrates what classification?"
        ),
        wrong_option_3="Hexadecimal classification",
        hint="Classification 1",
    )
    facts = graph_facts()

    errors = validate_cluster(tuple(cards), make_concept(1), facts)
    repaired = _repair_grounding_errors(
        tuple(cards), errors, facts, phrase_facts=facts[1:]
    )

    assert errors == ("card 1 hint reveals the correct answer",)
    assert repaired is not None
    assert repaired[0].wrong_option_3 == "Hexadecimal classification"
    assert validate_cluster(repaired, make_concept(1), facts) == ()


def test_overfull_cluster_is_retried_and_never_reaches_pipeline_result():
    overfull = json.loads(cluster_json(1))
    overfull["cards"].append(dict(overfull["cards"][0]))
    responses = [plan_json(), json.dumps(overfull), cluster_json(1)]
    responses.extend(cluster_json(index) for index in range(2, 21))
    backend = FakeBackend(responses)
    pipeline = FlashcardPipeline(
        backend,
        PipelineConfig(max_retries=3, final_review=False),
    )

    clusters = pipeline.run(ModuleIdentity("CPE0021", "1"), graph_facts())

    assert "expected exactly 5 cards, received 6" in backend.calls[2][1]
    assert sum(len(cluster.cards) for cluster in clusters) == 100


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
    assert "expected at least 20 concepts" in retry_prompt
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


def test_fewer_than_twenty_facts_still_asks_planner_for_insufficient_content():
    backend = FakeBackend(
        [json.dumps({"insufficient_content": "only nineteen concepts are supported"})]
    )


def flag_duplicate_without_card_location(system, user, max_tokens):
    payload = review_payload(user)
    return json.dumps(
        {
            "issues": [
                {
                    "cluster": payload["clusters"][1]["cluster"],
                    "reasons": ["semantic duplication"],
                }
            ]
        }
    )
    pipeline = FlashcardPipeline(backend, PipelineConfig(final_review=False))
    facts = graph_facts()[:19]

    with pytest.raises(GenerationError, match="more content is required"):
        pipeline.run(ModuleIdentity("CPE0021", "1"), facts)

    assert len(backend.calls) == 1


def test_prior_question_is_checked_without_current_module_clusters():
    card = FlashcardDraft(
        type="identification",
        question="What term names the instruction cycle?",
        correct_option="Instruction cycle",
        wrong_option_1="",
        wrong_option_2="",
        wrong_option_3="",
        is_true=None,
        expalanation="The instruction cycle is the named process.",
        hint="Think of the processor's repeated sequence.",
        difficulty=1,
        assessment_approach="recall",
    )

    errors = FlashcardPipeline._duplicate_errors(
        (card,),
        (),
        ("What term names the instruction cycle?",),
    )

    assert errors == (
        "card 1 duplicates a question from a previously generated module in this course",
    )


def test_duplicate_question_against_an_earlier_cluster_is_deferred_to_review():
    earlier = FlashcardCluster(
        cluster="00000000-0000-4000-8000-000000000001",
        concept=make_concept(1),
        cards=make_cards(1),
    )
    candidate = list(make_cards(2))
    candidate[0] = replace(candidate[0], question=earlier.cards[0].question)

    errors = FlashcardPipeline._duplicate_errors(tuple(candidate), (earlier,))

    assert not any("from concept" in error for error in errors)


def test_pipeline_threads_prior_concepts_and_retries_prior_question_duplicate():
    responses = [plan_json(), cluster_json(1), cluster_json(1, revision=1)]
    responses.extend(cluster_json(index) for index in range(2, 21))
    backend = FakeBackend(responses)
    pipeline = FlashcardPipeline(
        backend,
        PipelineConfig(max_retries=3, final_review=False),
    )
    duplicate_question = json.loads(cluster_json(1))["cards"][0]["question"]

    pipeline.run(
        ModuleIdentity("CPE0021", "2"),
        graph_facts(),
        prior_concept_names=("Earlier concept",),
        prior_questions=(duplicate_question,),
    )

    plan_payload = json.loads(backend.calls[0][1].split("INPUT JSON:\n", 1)[1])
    assert plan_payload["previously_covered_concepts"] == ["Earlier concept"]
    assert "already_covered_subjects" in backend.calls[1][1]
    assert "previously generated module" in backend.calls[2][1]
    assert "already_covered_subjects" in backend.calls[2][1]


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
    assert all(
        schema["properties"]["issues"]["type"] == "array"
        for schema in backend.schemas[21:]
    )
    global_payload = json.loads(review_calls[-1][1].split("INPUT JSON:\n", 1)[1])
    assert len(global_payload["clusters"]) == 20
    assert "questions" in global_payload["clusters"][0]
    assert "cards" not in global_payload["clusters"][0]


def test_final_review_repairs_only_flagged_card_and_preserves_uuid():
    responses = [plan_json()]
    responses.extend(cluster_json(index) for index in range(1, 21))
    responses.extend(empty_review() for _ in range(5))
    responses.append(flag_second_cluster_duplicate)
    responses.append(single_card_json(make_cards(2, revision=1)[0]))
    backend = FakeBackend(responses)
    pipeline = FlashcardPipeline(backend, PipelineConfig(final_review=True))

    clusters = pipeline.run(ModuleIdentity("CPE0021", "1"), graph_facts())

    assert "revision1" in clusters[1].cards[0].question
    assert clusters[1].cards[1:] == parse_cards(cluster_json(2))[1:]
    assert clusters[1].cluster in backend.calls[-1][1]
    assert "card 1 duplicates cluster" in backend.calls[-1][1]
    repair_schema = backend.schemas[-1]["properties"]["cards"]
    assert repair_schema["minItems"] == repair_schema["maxItems"] == 1


def test_global_duplicate_review_retries_when_card_location_is_missing():
    responses = [plan_json()]
    responses.extend(cluster_json(index) for index in range(1, 21))
    responses.extend(empty_review() for _ in range(5))
    responses.extend(
        (
            flag_duplicate_without_card_location,
            flag_second_cluster_duplicate,
            single_card_json(make_cards(2, revision=1)[0]),
        )
    )
    backend = FakeBackend(responses)
    pipeline = FlashcardPipeline(
        backend,
        PipelineConfig(max_retries=2, final_review=True),
        progress=lambda message: None,
    )

    clusters = pipeline.run(ModuleIdentity("CPE0021", "1"), graph_facts())

    assert "duplicate reason must use" in backend.calls[-2][1]
    assert "revision1" in clusters[1].cards[0].question


def test_final_duplicate_validation_paraphrases_instead_of_failing():
    first_cluster = json.loads(cluster_json(1))
    duplicate_cluster = json.loads(cluster_json(2))
    duplicate_cluster["cards"][0]["question"] = first_cluster["cards"][0]["question"]

    original_card = parse_cards(json.dumps(duplicate_cluster))[0]
    repaired_card = replace(
        original_card,
        question=make_cards(2, revision=1)[0].question,
        correct_option="A semantically equivalent reworded answer",
        wrong_option_1="Zephyr",
        wrong_option_2="Quasar",
        wrong_option_3="Nebula",
    )
    responses = [
        plan_json(),
        json.dumps(first_cluster),
        json.dumps(duplicate_cluster),
        *(cluster_json(index) for index in range(3, 21)),
        single_card_json(repaired_card),
    ]
    backend = FakeBackend(responses)
    pipeline = FlashcardPipeline(
        backend,
        PipelineConfig(final_review=False),
        progress=lambda message: None,
    )

    clusters = pipeline.run(ModuleIdentity("CPE0021", "1"), graph_facts())

    assert "revision1" in clusters[1].cards[0].question
    assert clusters[1].cards[0].correct_option == (
        "A semantically equivalent reworded answer"
    )
    assert clusters[1].cards[0].wrong_option_1 == "Zephyr"
    assert clusters[1].cards[1:] == parse_cards(cluster_json(2))[1:]
    assert "near-duplicate questions at cluster 1 card 1" in backend.calls[-1][1]


def test_duplicate_stack_combines_conflicts_for_one_card_location():
    first_cluster = json.loads(cluster_json(1))
    second_cluster = json.loads(cluster_json(2))
    third_cluster = json.loads(cluster_json(3))
    duplicate_question = first_cluster["cards"][0]["question"]
    second_cluster["cards"][0]["question"] = duplicate_question
    third_cluster["cards"][0]["question"] = duplicate_question
    responses = [
        plan_json(),
        json.dumps(first_cluster),
        json.dumps(second_cluster),
        json.dumps(third_cluster),
        *(cluster_json(index) for index in range(4, 21)),
        single_card_json(make_cards(3, revision=1)[0]),
        single_card_json(make_cards(2, revision=1)[0]),
    ]
    backend = FakeBackend(responses)
    pipeline = FlashcardPipeline(
        backend,
        PipelineConfig(final_review=False),
        progress=lambda message: None,
    )

    clusters = pipeline.run(ModuleIdentity("CPE0021", "1"), graph_facts())

    assert len(backend.calls) == 23
    first_repair_payload = json.loads(
        backend.calls[21][1].split("INPUT JSON:\n", 1)[1]
    )
    assert first_repair_payload["json_location"] == {
        "cluster": 3,
        "card": 1,
    }
    assert len(first_repair_payload["conflicting_questions"]) == 2
    assert clusters[1].cards[1:] == parse_cards(cluster_json(2))[1:]
    assert clusters[2].cards[1:] == parse_cards(cluster_json(3))[1:]


def test_duplicate_card_retry_includes_rejected_card_and_exact_conflict():
    first_cluster = json.loads(cluster_json(1))
    second_cluster = json.loads(cluster_json(2))
    duplicate_question = first_cluster["cards"][0]["question"]
    second_cluster["cards"][0]["question"] = duplicate_question
    rejected_card = replace(make_cards(2)[0], question=duplicate_question)
    accepted_card = make_cards(2, revision=1)[0]
    responses = [
        plan_json(),
        json.dumps(first_cluster),
        json.dumps(second_cluster),
        *(cluster_json(index) for index in range(3, 21)),
        single_card_json(rejected_card),
        single_card_json(accepted_card),
    ]
    backend = FakeBackend(responses)
    pipeline = FlashcardPipeline(
        backend,
        PipelineConfig(max_retries=2, final_review=False),
        progress=lambda message: None,
    )

    clusters = pipeline.run(ModuleIdentity("CPE0021", "1"), graph_facts())

    retry_prompt = backend.calls[-1][1]
    assert "REJECTED JSON:" in retry_prompt
    assert duplicate_question in retry_prompt
    assert clusters[1].cards[0] == parse_cards(single_card_json(accepted_card))[0]
    assert clusters[1].cards[1:] == parse_cards(cluster_json(2))[1:]


def test_true_false_duplicate_repair_restores_locked_prose_outside_schema():
    first_cluster = json.loads(cluster_json(1))
    second_cluster = json.loads(cluster_json(2))
    second_cluster["cards"][4]["question"] = first_cluster["cards"][4]["question"]
    second_cluster["cards"][4]["expalanation"] = (
        'Donald Arthur "Don" Norman is best known for his books on design, '
        "especially The Design of Everyday Things."
    )
    original_card = parse_cards(json.dumps(second_cluster))[4]
    interrogative = replace(
        original_card,
        question="Does placing the relationship in reverse order always keep it intact?",
    )
    changed_metadata = replace(
        original_card,
        question=(
            "The directional relationship always remains intact after its "
            "elements are placed in reverse order."
        ),
        expalanation="This replacement explanation should not be allowed to change.",
        hint="This replacement hint should remain locked during duplicate repair.",
    )
    responses = [
        plan_json(),
        json.dumps(first_cluster),
        json.dumps(second_cluster),
        *(cluster_json(index) for index in range(3, 21)),
        single_card_json(interrogative),
        single_card_json(changed_metadata),
    ]
    backend = FakeBackend(responses)
    pipeline = FlashcardPipeline(
        backend,
        PipelineConfig(max_retries=2, final_review=False),
        progress=lambda message: None,
    )

    clusters = pipeline.run(ModuleIdentity("CPE0021", "1"), graph_facts())

    repaired = clusters[1].cards[4]
    assert repaired.question == changed_metadata.question
    assert repaired.expalanation == original_card.expalanation
    assert repaired.hint == original_card.hint
    assert repaired.assessment_approach == original_card.assessment_approach
    assert "TRUE-FALSE DECLARATIVE REQUIREMENT" in backend.calls[-1][1]
    card_schema = backend.schemas[-1]["properties"]["cards"]["items"]
    assert card_schema["properties"]["assessment_approach"]["enum"] == [
        original_card.assessment_approach
    ]
    assert card_schema["properties"]["expalanation"] == {"type": "string"}
    assert card_schema["properties"]["hint"] == {"type": "string"}


def test_cluster_accepts_relevant_distractor_without_full_module_token_match():
    cards = list(make_cards(1))
    cards[0] = replace(cards[0], wrong_option_3="Related external alternative")
    response = json.dumps({"cards": [card.__dict__ for card in cards]})
    backend = FakeBackend([response])
    pipeline = FlashcardPipeline(
        backend,
        PipelineConfig(final_review=False),
        progress=lambda message: None,
    )
    concept_fact = GraphFact("e1", "subject1 relates to object1")
    generated = pipeline._generate_cards(
        ModuleIdentity("CPE0021", "1"),
        make_concept(1),
        (),
        label="full-module grounding",
        concept_facts=(concept_fact,),
        distractor_facts=(),
        module_facts=(concept_fact,),
    )

    assert generated[0].wrong_option_3 == "Related external alternative"
    assert pipeline.rejection_stats["rejected_attempts"] == 0


def test_provenance_wrappers_are_stripped_before_local_hint_repair():
    cards = list(make_cards(1))
    cards[0] = replace(
        cards[0],
        question=(
            "Which classification discussed in the module applies to "
            "topic1 alpha1 beta1 revision0?"
        ),
    )
    cards[2] = replace(
        cards[2],
        question=(
            "The module states that item topic1 alpha1 beta1 revision0 has "
            "its stated relationship."
        ),
    )
    cards[3] = replace(cards[3], hint=cards[3].correct_option)
    backend = FakeBackend(
        [json.dumps({"cards": [asdict(card) for card in cards]})]
    )
    pipeline = FlashcardPipeline(
        backend,
        PipelineConfig(final_review=False),
        progress=lambda message: None,
    )

    generated = pipeline._generate_cards(
        ModuleIdentity("CPE0021", "1"),
        make_concept(1),
        (),
        label="provenance cleanup",
    )

    assert generated[0].question == (
        "Which classification applies to topic1 alpha1 beta1 revision0?"
    )
    assert generated[2].question == (
        "Item topic1 alpha1 beta1 revision0 has its stated relationship."
    )
    assert generated[3].hint == (
        "Consider the relationship or distinction needed to answer."
    )
    assert len(backend.calls) == 1
