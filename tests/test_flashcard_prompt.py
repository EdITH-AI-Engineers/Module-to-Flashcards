import json

from flashcard_prompt import (
    SYSTEM_PROMPT,
    build_cluster_prompt,
    build_cluster_retry_prompt,
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


def test_system_prompt_requires_approach_labels_to_match_actual_reasoning():
    lowered = SYSTEM_PROMPT.casefold()

    assert "only the supplied graph facts" in lowered
    assert "return json only" in lowered
    assert all(
        name in SYSTEM_PROMPT
        for name in ("multiple-choice", "identification", "true-false")
    )
    assert "expalanation" in SYSTEM_PROMPT
    assert "approaches may repeat" in lowered
    assert "no planned approach is required to appear" in lowered
    assert "the assessment_approach label must describe the reasoning" in lowered
    assert '"what term refers to..." is recall or classification' in lowered


def test_cluster_prompt_avoids_banned_provenance_language_for_distractors():
    prompt = build_cluster_prompt(
        ModuleIdentity("CPE0021", "1"),
        concept(),
        (GraphFact("e1", "binary | uses | base 2"),),
        (GraphFact("e2", "octal | uses | base 8"),),
    )

    assert "provided module content" not in prompt.casefold()


def test_prompts_treat_scenario_analysis_as_an_approach_not_a_card_type():
    scenario_concept = ConceptPlan(
        "Binary base",
        ("e1",),
        ("binary | uses | base 2",),
        (
            "scenario analysis",
            "comparison",
            "application",
            "misconception detection",
            "reversed reasoning",
        ),
    )

    prompt = build_cluster_prompt(
        ModuleIdentity("CPE0021", "1"),
        scenario_concept,
        (GraphFact("e1", "binary | uses | base 2"),),
        (GraphFact("e2", "octal | uses | base 8"),),
    )

    assert "Use only multiple-choice, identification, and true-false." in SYSTEM_PROMPT
    assert '"type":"scenario analysis"' not in prompt
    scenario_examples = [
        json.loads(line.strip())
        for line in prompt.splitlines()
        if '"assessment_approach":"scenario analysis"' in line
    ]
    assert {example["type"] for example in scenario_examples} == {
        "multiple-choice",
        "identification",
        "true-false",
    }


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


def test_plan_prompt_does_not_require_exclusive_fact_ownership():
    prompt = build_concept_plan_prompt(
        ModuleIdentity("CPE0021", "1"),
        tuple(GraphFact(f"e{index}", f"fact {index}") for index in range(1, 21)),
    )

    assert "EVIDENCE OWNERSHIP" not in prompt
    assert "each fact_id may appear at most once" not in prompt.casefold()


def test_plan_prompt_includes_prior_concepts_when_supplied():
    prompt = build_concept_plan_prompt(
        ModuleIdentity("CPE0021", "2"),
        (GraphFact("e1", "binary | uses | base 2"),),
        ("Binary base",),
    )

    payload = json.loads(prompt.split("INPUT JSON:\n", 1)[1])
    assert payload["previously_covered_concepts"] == ["Binary base"]
    assert "same underlying learning point" in prompt


def test_cluster_prompt_allows_planned_approaches_in_any_card_order():
    prompt = build_cluster_prompt(
        ModuleIdentity("CPE0021", "1"),
        concept(),
        (GraphFact("e1", "binary | uses | base 2"),),
        (GraphFact("e2", "octal | uses | base 8"),),
    )

    payload = json.loads(prompt.split("INPUT JSON:\n", 1)[1])
    assert "facts" not in payload["concept"]
    assert payload["concept_facts"] == [
        {"fact_id": "e1", "statement": "binary | uses | base 2"}
    ]
    assert "distractor_pool" not in payload
    assert "grounded_vocabulary" not in payload
    assert "allowed_wrong_option_terms" in payload
    assert "binary" in payload["allowed_wrong_option_terms"]
    assert "octal" in payload["allowed_wrong_option_terms"]
    assert "at least one exact" in prompt
    assert payload["concept"]["assessment_approaches"] == list(
        concept().assessment_approaches
    )
    for approach in concept().assessment_approaches:
        assert f'- "{approach}"' in prompt
    normalized_prompt = " ".join(prompt.split())
    assert "Approaches may repeat" in prompt
    assert "a listed approach does not have to appear" in normalized_prompt
    assert "no approach is tied to a numbered card position" in normalized_prompt
    assert "POSITIONAL ASSESSMENT APPROACHES" not in prompt
    assert "Card 1: assessment_approach" not in prompt
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
    assert "one of the planned values" in prompt
    assert "Approaches may repeat" in prompt
    assert 'all exactly ""' in prompt
    assert "four different strings" in prompt
    assert "ORIGINAL" in prompt


def test_retry_prompt_can_omit_large_rejected_candidate():
    prompt = build_retry_prompt("ORIGINAL", None, ["unknown fact id"])

    assert "ORIGINAL" in prompt
    assert "unknown fact id" in prompt
    assert "rejected_candidate" not in prompt


def test_cluster_retry_is_compact_and_does_not_nest_original_prompt():
    large_concept = ConceptPlan(
        "Design Rules",
        tuple(f"f{index}" for index in range(1, 20)),
        tuple(f"Design rule fact {index} with supporting detail." for index in range(1, 20)),
        ("recall", "comparison", "classification", "application", "scenario analysis"),
    )
    concept_facts = tuple(
        GraphFact(
            f"f{index}",
            f"Design rule fact {index} with enough supporting detail for assessment.",
        )
        for index in range(1, 20)
    )
    distractor_facts = tuple(
        GraphFact(
            f"d{index}",
            f"Related distractor fact {index} with grounded terminology.",
        )
        for index in range(1, 13)
    )
    prior = tuple(
        f"Previously covered subject number {index} -> prior answer"
        for index in range(1, 21)
    )

    prompt = build_cluster_retry_prompt(
        ModuleIdentity("CCS0005", "1"),
        large_concept,
        concept_facts,
        distractor_facts,
        prior,
        "PARTIAL_CANDIDATE",
        ["response must contain a JSON object only"],
    )

    assert "ORIGINAL REQUEST" not in prompt
    assert prompt.count("Design rule fact 1 with enough supporting detail") == 1
    assert "PARTIAL_CANDIDATE" in prompt
    assert "Approaches may repeat" in prompt
    assert "positional assessment_approach order" not in prompt
    assert "A direct definition" in prompt
    assert "not scenario analysis" in prompt
    assert len(SYSTEM_PROMPT) + len(prompt) < 30_000


def test_cluster_retry_omits_unusable_partial_output_after_truncation():
    prompt = build_cluster_retry_prompt(
        ModuleIdentity("CPE0021", "1"),
        concept(),
        (GraphFact("e1", "binary | uses | base 2"),),
        (GraphFact("e2", "octal | uses | base 8"),),
        (),
        None,
        ["response was truncated before completing the JSON"],
    )

    assert "REJECTED JSON TO CORRECT" not in prompt
    assert "truncated before completing" in prompt


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
