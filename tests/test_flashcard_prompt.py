import json

from flashcard_prompt import (
    SYSTEM_PROMPT,
    build_cluster_prompt,
    build_concept_plan_prompt,
    build_duplicate_review_prompt,
    build_grounding_review_prompt,
    build_retry_prompt,
)
from flashcard_types import (
    ConceptPlan,
    FlashcardCluster,
    FlashcardDraft,
    GraphFact,
    ModuleIdentity,
)


def concept():
    return ConceptPlan(
        "Binary base",
        ("e1",),
        ("binary | uses | base 2",),
        (
            "recall",
            "comparison",
            "application",
            "misconception detection",
            "reversed reasoning",
        ),
    )


def card(question="Which relationship is valid?"):
    return FlashcardDraft(
        type="multiple-choice",
        question=question,
        correct_option="Binary uses base 2",
        wrong_option_1="Binary uses base 8",
        wrong_option_2="Binary uses base 10",
        wrong_option_3="Binary uses base 16",
        is_true=None,
        expalanation="Binary is related to base 2.",
        hint="Focus on the stated base.",
        difficulty=1,
        assessment_approach="recall",
    )


def cluster():
    return FlashcardCluster(
        cluster="123e4567-e89b-42d3-a456-426614174000",
        concept=concept(),
        cards=(card(),),
    )


def test_system_prompt_requires_distinct_assessment_approaches():
    lowered = SYSTEM_PROMPT.casefold()

    assert "only the supplied graph facts" in lowered
    assert "return json only" in lowered
    assert all(
        name in SYSTEM_PROMPT
        for name in ("multiple-choice", "identification", "true-false")
    )
    assert "expalanation" in SYSTEM_PROMPT
    assert "meaningfully different assessment approaches" in lowered
    assert "assessment approach labels may repeat" not in lowered


def test_plan_prompt_serializes_relationships_without_provenance():
    prompt = build_concept_plan_prompt(
        ModuleIdentity("CPE0021", "1"),
        (GraphFact("e1", "binary | uses | base 2"),),
    )

    payload = json.loads(prompt.split("INPUT JSON:\n", 1)[1])
    assert payload == {
        "course_code": "CPE0021",
        "module_number": "1",
        "graph_facts": [{"fact_id": "e1", "statement": "binary | uses | base 2"}],
    }
    assert "slide" not in prompt.casefold()
    assert "array must contain exactly 20 concept objects" in prompt.casefold()
    assert 'key "fact_ids" literally' in prompt
    assert "choose exactly 5 distinct approaches" in prompt.casefold()
    assert "array of exactly 5 distinct allowed approaches" in prompt.casefold()
    assert "labels may repeat" not in prompt.casefold()


def test_plan_prompt_includes_prior_concepts_when_supplied():
    prompt = build_concept_plan_prompt(
        ModuleIdentity("CPE0021", "2"),
        (GraphFact("e1", "binary | uses | base 2"),),
        ("Binary base",),
    )

    payload = json.loads(prompt.split("INPUT JSON:\n", 1)[1])
    assert payload["previously_covered_concepts"] == ["Binary base"]
    assert "same underlying learning point" in prompt


def test_cluster_prompt_assigns_each_planned_approach_by_card_position():
    prompt = build_cluster_prompt(ModuleIdentity("CPE0021", "1"), concept())

    payload = json.loads(prompt.split("INPUT JSON:\n", 1)[1])
    assert payload["concept"]["facts"] == ["binary | uses | base 2"]
    assert payload["concept"]["assessment_approaches"] == list(
        concept().assessment_approaches
    )
    for position, approach in enumerate(concept().assessment_approaches, start=1):
        assert (
            f'Card {position}: assessment_approach must be exactly "{approach}"'
            in prompt
        )
    assert "REPEATED CRITICAL RULES" in prompt
    assert 'must literally be ""' in prompt
    assert "no wrong_option may repeat" in prompt
    assert "Never write knowledge graph" in prompt


def test_retry_prompt_targets_named_cards_and_repeats_critical_checks():
    prompt = build_retry_prompt(
        "ORIGINAL",
        '{"cards": []}',
        ["card 2 identification wrong options must be empty"],
    )

    assert "complete replacement" in prompt.casefold()
    assert "correct that exact card" in prompt.casefold()
    assert "card 2 identification wrong options must be empty" in prompt
    assert "assessment_approach exactly match" in prompt
    assert 'all exactly ""' in prompt
    assert "four different strings" in prompt
    assert "ORIGINAL" in prompt


def test_retry_prompt_can_omit_large_rejected_candidate():
    prompt = build_retry_prompt("ORIGINAL", None, ["unknown fact id"])

    assert "ORIGINAL" in prompt
    assert "unknown fact id" in prompt
    assert "rejected_candidate" not in prompt


def test_grounding_review_includes_evidence_and_card_content():
    prompt = build_grounding_review_prompt((cluster(),))

    assert "binary | uses | base 2" in prompt
    assert "Binary uses base 2" in prompt
    assert '"issues"' in prompt


def test_duplicate_review_excludes_answers_and_evidence():
    prompt = build_duplicate_review_prompt((cluster(),))

    assert "Which relationship is valid?" in prompt
    assert "Binary uses base 2" not in prompt
    assert "binary | uses | base 2" not in prompt
    assert '"issues"' in prompt
