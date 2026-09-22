import json
from dataclasses import replace

import pytest

from flashcard_types import ConceptPlan, FlashcardDraft, GraphFact
from flashcard_validator import (
    InsufficientContentError,
    ValidationError,
    parse_cards,
    parse_concept_plan,
    parse_review_issues,
    validate_cluster,
)


APPROACHES = (
    "recall",
    "comparison",
    "misconception detection",
    "application",
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


def test_cards_parser_rejects_using_distractor_pool_as_answer_authority():
    values = list(valid_cards())
    values[0] = replace(
        values[0],
        expalanation="This answer is stated in the distractor pool.",
    )

    with pytest.raises(
        ValidationError,
        match="treats internal distractor data as answer authority",
    ):
        parse_cards(cards_json(tuple(values)))


@pytest.mark.parametrize(
    ("question", "expected"),
    (
        (
            "Which measure applies according to the module?",
            "Which measure applies?",
        ),
        (
            "How do the methods differ based on the supplied material?",
            "How do the methods differ?",
        ),
        (
            "What principle follows as described in the lesson?",
            "What principle follows?",
        ),
        (
            "According to the document, which outcome is supported?",
            "Which outcome is supported?",
        ),
        (
            "Which outcome is supported according to the source material?",
            "Which outcome is supported?",
        ),
        (
            "Which property, according to the module, determines the outcome?",
            "Which property determines the outcome?",
        ),
    ),
)
def test_cards_parser_removes_generic_provenance_wrappers(question, expected):
    payload = json.loads(cards_json())
    payload["cards"][0]["question"] = question

    cards = parse_cards(json.dumps(payload))

    assert cards[0].question == expected
    assert "card 1 exposes provenance metadata" not in validate_cluster(
        cards, valid_concept()
    )


def test_cards_parser_keeps_non_provenance_according_to_clause():
    payload = json.loads(cards_json())
    question = "Which setting changes according to user preference?"
    payload["cards"][0]["question"] = question

    cards = parse_cards(json.dumps(payload))

    assert cards[0].question == question


def test_cards_parser_keeps_compound_source_phrase_intact():
    payload = json.loads(cards_json())
    question = "Which value is explicitly stated in the information table?"
    payload["cards"][0]["question"] = question

    cards = parse_cards(json.dumps(payload))

    assert cards[0].question == question


@pytest.mark.parametrize(
    "question",
    (
        "Which outcome appears in the supplied material?",
        "Which conclusion follows from the provided information?",
        "Which concept is defined in the text?",
        "Which conclusion follows from the given facts?",
        "Which topic appears in the slides?",
        "Which claim appears in the given source?",
        "Which result follows from the facts?",
    ),
)
def test_cluster_rejects_unwrapped_provenance_language(question):
    payload = json.loads(cards_json())
    payload["cards"][0]["question"] = question
    cards = parse_cards(json.dumps(payload))

    errors = validate_cluster(cards, valid_concept())

    assert "card 1 exposes provenance metadata" in errors


def test_cards_parser_rejects_boolean_true_false_value():
    values = list(valid_cards())
    values[2] = replace(values[2], is_true=True)
    with pytest.raises(ValidationError, match="integer 0, integer 1, or null"):
        parse_cards(cards_json(tuple(values)))


def test_cards_parser_supplies_omitted_empty_identification_options():
    payload = json.loads(cards_json())
    identification = payload["cards"][1]
    identification.pop("wrong_option_1")
    identification.pop("wrong_option_2")
    identification.pop("wrong_option_3")

    cards = parse_cards(json.dumps(payload))

    assert cards[1].wrong_option_1 == ""
    assert cards[1].wrong_option_2 == ""
    assert cards[1].wrong_option_3 == ""


def test_cards_parser_clears_conventional_true_false_option_labels():
    payload = json.loads(cards_json())
    true_false = payload["cards"][2]
    true_false["wrong_option_1"] = "True"
    true_false["wrong_option_2"] = "False"

    cards = parse_cards(json.dumps(payload))

    assert cards[2].correct_option == ""
    assert cards[2].wrong_option_1 == ""
    assert cards[2].wrong_option_2 == ""
    assert cards[2].wrong_option_3 == ""


def test_cards_parser_supplies_omitted_true_false_answer_options():
    payload = json.loads(cards_json())
    true_false = payload["cards"][2]
    true_false.pop("correct_option")
    true_false.pop("wrong_option_1")
    true_false.pop("wrong_option_2")
    true_false.pop("wrong_option_3")

    cards = parse_cards(json.dumps(payload))

    assert cards[2].correct_option == ""
    assert cards[2].wrong_option_1 == ""
    assert cards[2].wrong_option_2 == ""
    assert cards[2].wrong_option_3 == ""


def test_concept_plan_requires_twenty_supported_concepts():
    raw, known = plan_json(count=1)
    with pytest.raises(ValidationError, match="at least 20 concepts"):
        parse_concept_plan(raw, known)


def test_concept_plan_trims_supported_overshoot_to_twenty():
    raw, known = plan_json(count=24)

    concepts = parse_concept_plan(raw, known)

    assert len(concepts) == 20
    assert concepts[-1].name == "Concept 20"


def test_concept_plan_attaches_exact_fact_text_from_known_ids():
    raw, known = plan_json()

    concepts = parse_concept_plan(raw, known)

    assert concepts[0].fact_ids == ("e1",)
    assert concepts[0].facts == ("subject 1 | relates to | object 1",)


def test_concept_plan_accepts_comma_separated_token_fields():
    raw, known = plan_json()
    payload = json.loads(raw)
    payload["concepts"][0]["fact_ids"] = "e1,e2,e3"
    payload["concepts"][1]["fact_ids"] = ["e21"]
    payload["concepts"][2]["fact_ids"] = ["e22"]
    payload["concepts"][0]["assessment_approaches"] = (
        "recall,comparison,classification,application,scenario analysis"
    )
    known = (
        *known,
        GraphFact("e21", "subject 21 | relates to | object 21"),
        GraphFact("e22", "subject 22 | relates to | object 22"),
    )

    concepts = parse_concept_plan(json.dumps(payload), known)

    assert concepts[0].fact_ids == ("e1", "e2", "e3")
    assert concepts[0].assessment_approaches == (
        "recall",
        "comparison",
        "classification",
        "application",
        "scenario analysis",
    )


@pytest.mark.parametrize(
    "approaches",
    (
        ["guided review", "guided review"],
        list(APPROACHES[:-1]),
        [*APPROACHES, "classification"],
    ),
)
def test_concept_plan_requires_exactly_five_distinct_assessment_approaches(
    approaches,
):
    raw, known = plan_json()
    payload = json.loads(raw)
    payload["concepts"][0]["assessment_approaches"] = approaches

    with pytest.raises(ValidationError, match="exactly 5 distinct"):
        parse_concept_plan(json.dumps(payload), known)


def test_concept_plan_rejects_unsupported_assessment_approach():
    raw, known = plan_json()
    payload = json.loads(raw)
    payload["concepts"][0]["assessment_approaches"][-1] = "guided review"

    with pytest.raises(ValidationError, match="unsupported assessment approaches"):
        parse_concept_plan(json.dumps(payload), known)


@pytest.mark.parametrize("value", (123, None, ""))
def test_concept_plan_rejects_malformed_fact_id_scalars(value):
    raw, known = plan_json()
    payload = json.loads(raw)
    payload["concepts"][0]["fact_ids"] = value

    with pytest.raises(ValidationError, match="fact_ids must be a non-empty JSON array"):
        parse_concept_plan(json.dumps(payload), known)


def test_review_reason_with_a_comma_remains_one_freeform_string():
    cluster = "00000000-0000-4000-8000-000000000001"
    reason = "the option overlaps with another, and the explanation is unclear"

    issues = parse_review_issues(
        json.dumps({"issues": [{"cluster": cluster, "reasons": [reason]}]}),
        {cluster},
    )

    assert issues[0].reasons == (reason,)


def test_concept_plan_rejects_unknown_fact_id():
    raw, known = plan_json()
    payload = json.loads(raw)
    payload["concepts"][0]["fact_ids"] = ["missing"]

    with pytest.raises(ValidationError, match="unknown fact id"):
        parse_concept_plan(json.dumps(payload), known)


def test_concept_plan_rejects_fact_id_reused_by_another_concept():
    raw, known = plan_json()
    payload = json.loads(raw)
    payload["concepts"][6]["fact_ids"] = ["e2"]

    with pytest.raises(ValidationError) as exc_info:
        parse_concept_plan(json.dumps(payload), known)

    assert exc_info.value.errors == (
        "concept 7 reuses fact id 'e2' already assigned to concept 2; "
        "each fact id may support only one concept",
    )


def test_concept_plan_rejects_repeated_fact_id_within_one_concept():
    raw, known = plan_json()
    payload = json.loads(raw)
    payload["concepts"][0]["fact_ids"] = ["e1", "e1"]

    with pytest.raises(ValidationError) as exc_info:
        parse_concept_plan(json.dumps(payload), known)

    assert exc_info.value.errors == ("concept 1 repeats fact id 'e1'",)


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


def test_multiple_choice_accepts_a_context_first_question():
    values = list(valid_cards())
    values[0] = replace(
        values[0],
        question=(
            "A learner groups a value by its numerical base. "
            "This example demonstrates what classification?"
        ),
    )

    assert validate_cluster(tuple(values), valid_concept()) == ()


def test_multiple_choice_accepts_nonleading_according_to_wording():
    values = list(valid_cards())
    values[0] = replace(
        values[0],
        question=(
            "A setting changes according to user preference. "
            "This demonstrates what behavior?"
        ),
    )

    assert validate_cluster(tuple(values), valid_concept()) == ()


@pytest.mark.parametrize(
    "question",
    (
        "According to the design rules, which measure applies?",
        "Based on the design rules, which measure applies?",
    ),
)
def test_multiple_choice_rejects_leading_wrapper_phrases(question):
    values = list(valid_cards())
    values[0] = replace(values[0], question=question)

    errors = validate_cluster(tuple(values), valid_concept())

    assert "card 1 question contains banned framing" in errors


def test_context_first_multiple_choice_requires_a_question_mark():
    values = list(valid_cards())
    values[0] = replace(
        values[0],
        question=(
            "A learner groups a value by its numerical base. "
            "This example demonstrates a classification"
        ),
    )

    errors = validate_cluster(tuple(values), valid_concept())

    assert "card 1 direct question must end with a question mark" in errors


def test_cluster_rejects_scenario_analysis_as_a_card_type():
    values = list(valid_cards())
    values[0] = replace(
        values[0],
        type="scenario analysis",
        question=(
            "A learner classifies a numeral system by its base. "
            "Which base identifies the binary relationship?"
        ),
    )

    errors = validate_cluster(tuple(values), valid_concept())

    assert any("type must be one of" in error for error in errors)


@pytest.mark.parametrize("position", (0, 1, 2))
def test_each_card_type_accepts_scenario_analysis_as_its_approach(position):
    values = list(valid_cards())
    scenario_questions = (
        "A learner sees a numeral system using two symbols. Which base applies?",
        "What term names the system when a learner sees that it uses base 2?",
        "A learner observes base 2, so the numeral system has the binary relationship.",
    )
    approaches = list(APPROACHES)
    approaches[position] = "scenario analysis"
    values[position] = replace(
        values[position],
        question=scenario_questions[position],
        assessment_approach="scenario analysis",
    )
    concept = replace(valid_concept(), assessment_approaches=tuple(approaches))

    assert validate_cluster(tuple(values), concept) == ()


def test_scenario_analysis_rejects_a_definition_question_without_a_situation():
    values = list(valid_cards())
    approaches = list(APPROACHES)
    approaches[1] = "scenario analysis"
    values[1] = replace(
        values[1],
        question="What concept applies when designing an efficient interface?",
        correct_option="Design Rules",
        assessment_approach="scenario analysis",
    )
    concept = replace(valid_concept(), assessment_approaches=tuple(approaches))

    errors = validate_cluster(tuple(values), concept)

    assert any(
        "scenario analysis must present a concrete situation" in error
        for error in errors
    )


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
        (1, {"question": "The binary value belongs where?"}, "direct question stem"),
        (3, {"hint": "The answer is As base 2."}, "hint reveals the correct answer"),
        (4, {"question": "The source says base 10 replaces base 2."}, "provenance metadata"),
    ],
)
def test_cluster_rejects_type_and_text_rule_violations(position, changes, message):
    values = list(valid_cards())
    values[position] = replace(values[position], **changes)

    errors = validate_cluster(tuple(values), valid_concept())

    assert any(message in error for error in errors)


def test_cluster_allows_planned_assessment_approaches_to_repeat():
    values = tuple(replace(card, assessment_approach="recall") for card in valid_cards())

    errors = validate_cluster(values, valid_concept())

    assert not any("assessment_approach must be one of" in error for error in errors)


def test_cluster_accepts_planned_assessment_approaches_in_any_order():
    values = list(valid_cards())
    values[0] = replace(values[0], assessment_approach="comparison")
    values[1] = replace(values[1], assessment_approach="recall")

    errors = validate_cluster(tuple(values), valid_concept())

    assert not any("assessment approach" in error for error in errors)


def test_cluster_rejects_an_unplanned_assessment_approach():
    values = list(valid_cards())
    values[-1] = replace(values[-1], assessment_approach="guided review")

    errors = validate_cluster(tuple(values), valid_concept())

    assert any(
        "card 5 assessment_approach must be one of the planned approaches" in error
        for error in errors
    )


def test_grounding_accepts_common_derivational_word_forms():
    values = list(valid_cards())
    values[0] = replace(
        values[0],
        wrong_option_1="Safety",
        wrong_option_2="Comfort",
        wrong_option_3="Enjoyment",
    )
    facts = (
        GraphFact(
            "e1",
            "binary uses base 2 and should be safe comfortable and enjoyable",
        ),
    )

    errors = validate_cluster(tuple(values), valid_concept(), facts)

    assert not any("not grounded" in error for error in errors)


@pytest.mark.parametrize("phrase", ("concept fact", "distractor pool"))
def test_cluster_rejects_internal_evidence_labels(phrase):
    values = list(valid_cards())
    values[0] = replace(
        values[0],
        expalanation=f"The answer is supported by the {phrase}.",
    )

    errors = validate_cluster(tuple(values), valid_concept())

    assert "card 1 exposes provenance metadata" in errors


def test_cluster_rejects_blank_assessment_approach():
    values = list(valid_cards())
    values[-1] = replace(values[-1], assessment_approach="   ")

    errors = validate_cluster(tuple(values), valid_concept())

    assert any("assessment approach must not be empty" in error for error in errors)


def test_cluster_requires_all_three_question_types():
    values = tuple(replace(card, type="multiple-choice", is_true=None) for card in valid_cards())

    errors = validate_cluster(values, valid_concept())

    assert any("at least one multiple-choice, identification, and true-false" in error for error in errors)
