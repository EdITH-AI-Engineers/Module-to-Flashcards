import json
from dataclasses import replace

import pytest

from flashcard_types import ConceptPlan, FlashcardDraft, GraphFact
from flashcard_validator import (
    InsufficientContentError,
    ValidationError,
    parse_cards,
    parse_concept_plan,
    validate_cluster,
)


APPROACHES = (
    "recall",
    "comparison",
    "application",
    "misconception detection",
    "reversed reasoning",
)


def valid_concept():
    return ConceptPlan(
        "Binary base",
        ("e1",),
        ("binary | uses | base 2",),
        APPROACHES,
    )


def valid_cards():
    return (
        FlashcardDraft(
            type="multiple-choice",
            question="Which base is associated with the binary relationship?",
            correct_option="Base 2",
            wrong_option_1="Base 8",
            wrong_option_2="Base 10",
            wrong_option_3="Base 16",
            is_true=None,
            expalanation="The binary relationship identifies base 2.",
            hint="Focus on the numerical base in the relationship.",
            difficulty=1,
            assessment_approach="recall",
        ),
        FlashcardDraft(
            type="identification",
            question="What term names the numeral system in the relationship?",
            correct_option="Binary",
            wrong_option_1="",
            wrong_option_2="",
            wrong_option_3="",
            is_true=None,
            expalanation="Binary is the named numeral system.",
            hint="Recall the system associated with two symbols.",
            difficulty=1,
            assessment_approach="comparison",
        ),
        FlashcardDraft(
            type="true-false",
            question="The binary relationship uses base 2.",
            correct_option="",
            wrong_option_1="",
            wrong_option_2="",
            wrong_option_3="",
            is_true=1,
            expalanation="The relationship explicitly connects binary with base 2.",
            hint="Check the direction and value in the relationship.",
            difficulty=1,
            assessment_approach="misconception detection",
        ),
        FlashcardDraft(
            type="multiple-choice",
            question="How should a binary value be classified by its base?",
            correct_option="As base 2",
            wrong_option_1="As base 8",
            wrong_option_2="As base 10",
            wrong_option_3="As base 16",
            is_true=None,
            expalanation="A binary value belongs to the base-2 system.",
            hint="Use the relationship between the system and its base.",
            difficulty=2,
            assessment_approach="application",
        ),
        FlashcardDraft(
            type="true-false",
            question="Reversing base 2 to base 10 preserves the binary relationship.",
            correct_option="",
            wrong_option_1="",
            wrong_option_2="",
            wrong_option_3="",
            is_true=0,
            expalanation="Base 10 does not preserve the stated binary relationship.",
            hint="Compare the two base values.",
            difficulty=2,
            assessment_approach="reversed reasoning",
        ),
    )


def cards_json(cards=None):
    values = cards or valid_cards()
    return json.dumps({"cards": [card.__dict__ for card in values]})


def plan_json(count=20):
    concepts = []
    known = []
    for index in range(1, count + 1):
        fact_id = f"e{index}"
        statement = f"subject {index} | relates to | object {index}"
        known.append(GraphFact(fact_id, statement))
        concepts.append(
            {
                "name": f"Concept {index}",
                "fact_ids": [fact_id],
                "assessment_approaches": list(APPROACHES),
            }
        )
    return json.dumps({"concepts": concepts}), tuple(known)


def test_parser_rejects_markdown_around_json():
    with pytest.raises(ValidationError, match="JSON object only"):
        parse_cards('```json\n{"cards": []}\n```')


def test_cards_parser_creates_typed_records():
    assert parse_cards(cards_json()) == valid_cards()


def test_cards_parser_rejects_boolean_true_false_value():
    values = list(valid_cards())
    values[2] = replace(values[2], is_true=True)
    with pytest.raises(ValidationError, match="integer 0, integer 1, or null"):
        parse_cards(cards_json(tuple(values)))


def test_concept_plan_requires_twenty_supported_concepts():
    raw, known = plan_json(count=1)
    with pytest.raises(ValidationError, match="exactly 20 concepts"):
        parse_concept_plan(raw, known)


def test_concept_plan_attaches_exact_fact_text_from_known_ids():
    raw, known = plan_json()

    concepts = parse_concept_plan(raw, known)

    assert concepts[0].fact_ids == ("e1",)
    assert concepts[0].facts == ("subject 1 | relates to | object 1",)


def test_concept_plan_rejects_unknown_fact_id():
    raw, known = plan_json()
    payload = json.loads(raw)
    payload["concepts"][0]["fact_ids"] = ["missing"]

    with pytest.raises(ValidationError, match="unknown fact id"):
        parse_concept_plan(json.dumps(payload), known)


def test_concept_plan_reports_explicit_insufficient_content():
    with pytest.raises(InsufficientContentError, match="only ten concepts"):
        parse_concept_plan(
            json.dumps({"insufficient_content": "only ten concepts are supported"}),
            (),
        )


def test_concept_plan_rejects_placeholder_insufficient_reason():
    with pytest.raises(ValidationError, match="specific limitation") as exc_info:
        parse_concept_plan(
            json.dumps({"insufficient_content": "concise reason"}),
            (),
        )

    assert type(exc_info.value) is ValidationError


def test_valid_cluster_has_no_errors():
    assert validate_cluster(valid_cards(), valid_concept()) == ()


@pytest.mark.parametrize(
    ("position", "changes", "message"),
    [
        (0, {"is_true": 1}, "multiple-choice is_true must be empty"),
        (0, {"wrong_option_3": ""}, "three non-empty wrong options"),
        (0, {"wrong_option_2": "Base 8"}, "options must be distinct"),
        (1, {"wrong_option_1": "Wrong"}, "identification wrong options must be empty"),
        (1, {"correct_option": "It is the term that names this complete relationship."}, "concise phrase"),
        (2, {"correct_option": "True"}, "true-false answer options must be empty"),
        (2, {"is_true": None}, "true-false is_true must be 0 or 1"),
        (2, {"question": "True or False? The binary relationship uses base 2."}, "declarative statement only"),
        (3, {"difficulty": 4}, "difficulty must be 1, 2, or 3"),
        (3, {"question": "According to the graph, which base applies?"}, "banned framing"),
        (3, {"question": "The binary value belongs where?"}, "direct question stem"),
        (3, {"hint": "The answer is As base 2."}, "hint reveals the correct answer"),
        (4, {"question": "The source says base 10 replaces base 2."}, "provenance metadata"),
    ],
)
def test_cluster_rejects_type_and_text_rule_violations(position, changes, message):
    values = list(valid_cards())
    values[position] = replace(values[position], **changes)

    errors = validate_cluster(tuple(values), valid_concept())

    assert any(message in error for error in errors)


def test_cluster_requires_five_distinct_assessment_approaches():
    values = list(valid_cards())
    values[-1] = replace(values[-1], assessment_approach="recall")

    errors = validate_cluster(tuple(values), valid_concept())

    assert any("five distinct assessment approaches" in error for error in errors)


def test_cluster_requires_all_three_question_types():
    values = tuple(replace(card, type="multiple-choice", is_true=None) for card in valid_cards())

    errors = validate_cluster(values, valid_concept())

    assert any("at least one multiple-choice, identification, and true-false" in error for error in errors)
