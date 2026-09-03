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


def test_system_prompt_defines_internal_json_contract_and_all_types():
    lowered = SYSTEM_PROMPT.lower()

    assert "only the supplied graph facts" in lowered
    assert "return json only" in lowered
    assert all(name in SYSTEM_PROMPT for name in ("multiple-choice", "identification", "true-false"))
    assert "expalanation" in SYSTEM_PROMPT


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
    assert "slide" not in prompt.lower()
    assert '"concepts"' in prompt
    assert "array must contain exactly 20 concept objects" in prompt.lower()
    assert "concise reason" not in prompt.lower()


def test_cluster_prompt_contains_only_the_selected_concept_evidence():
    prompt = build_cluster_prompt(ModuleIdentity("CPE0021", "1"), concept())

    payload = json.loads(prompt.split("INPUT JSON:\n", 1)[1])
    assert payload["concept"]["facts"] == ["binary | uses | base 2"]
    assert payload["concept"]["assessment_approaches"] == list(concept().assessment_approaches)
    assert "exactly 5" in prompt
    assert '"cards"' in prompt


def test_retry_prompt_requests_complete_replacement_with_errors():
    prompt = build_retry_prompt("ORIGINAL", '{"cards": []}', ["expected exactly 5 cards"])

    assert "complete replacement" in prompt.lower()
    assert "expected exactly 5 cards" in prompt
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
