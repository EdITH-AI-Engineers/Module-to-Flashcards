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


def test_cards_parser_normalizes_smart_and_mojibake_apostrophes():
    payload = json.loads(cards_json())
    payload["cards"][0]["question"] = "Which of Norman\u2019s principles applies?"
    payload["cards"][0]["correct_option"] = (
        "Norman\u00e2\u20ac\u2122s visibility principle"
    )
    payload["cards"][0]["expalanation"] = (
        "It clearly follows Norman\u2018s established visibility guidance here."
    )

    card = parse_cards(json.dumps(payload, ensure_ascii=False))[0]

    assert card.question == "Which of Norman's principles applies?"
    assert card.correct_option == "Norman's visibility principle"
    assert card.expalanation == (
        "It clearly follows Norman's established visibility guidance here."
    )


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
        (
            "The module states that design rules are a set of technical specifications.",
            "Design rules are a set of technical specifications.",
        ),
        (
            "Which term refers to the original title of a design rule discussed in the module?",
            "Which term refers to the original title of a design rule?",
        ),
        (
            "What is the original title of the design rule mentioned in the module?",
            "What is the original title of the design rule?",
        ),
        (
            "The module explicitly states that design rules guide decisions.",
            "Design rules guide decisions.",
        ),
        (
            "Which design rule, discussed in the module, guides decisions?",
            "Which design rule guides decisions?",
        ),
        (
            "Which design rule that is mentioned in the module guides decisions?",
            "Which design rule guides decisions?",
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


@pytest.mark.parametrize(
    "question",
    (
        "The module states are synchronized across replicas.",
        "Which design rule was mentioned in the module?",
    ),
)
def test_cards_parser_does_not_erase_substantive_provenance_predicates(question):
    payload = json.loads(cards_json())
    payload["cards"][0]["question"] = question

    cards = parse_cards(json.dumps(payload))

    assert cards[0].question == question


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


def test_cards_parser_does_not_partially_strip_possessive_provenance_modifier():
    payload = json.loads(cards_json())
    question = "Which role is mentioned in the module's focus statement?"
    payload["cards"][0]["question"] = question

    cards = parse_cards(json.dumps(payload))

    assert cards[0].question == question
    assert "card 1 exposes provenance metadata" in validate_cluster(
        cards, valid_concept()
    )


@pytest.mark.parametrize(
    "hint",
    (
        "Focus on the specific role mentioned in the module's focus statement.",
        "Recall the main topic mentioned in the module's description.",
        "Refer to the provided fact about the history of HCI.",
        "This follows, as indicated by the provided facts.",
    ),
)
def test_cards_parser_falls_back_instead_of_damaging_provenance_hints(hint):
    payload = json.loads(cards_json())
    payload["cards"][0]["hint"] = hint

    card = parse_cards(json.dumps(payload))[0]

    assert card.hint == (
        "Consider the key relationship or distinction central to this topic."
    )


def test_cards_parser_does_not_leave_a_damaged_passive_provenance_hint():
    payload = json.loads(cards_json())
    payload["cards"][0]["hint"] = "The title is directly mentioned in the fact."

    card = parse_cards(json.dumps(payload))[0]

    assert card.hint == (
        "Consider the key relationship or distinction central to this topic."
    )
    assert card.hint != "The title is directly."


def test_cards_parser_removes_a_fact_id_wrapper_without_losing_the_explanation():
    payload = json.loads(cards_json())
    payload["cards"][0]["expalanation"] = (
        "Fact f14 confirms that binary uses base 2 rather than base 10."
    )

    card = parse_cards(json.dumps(payload))[0]

    assert card.expalanation == "Binary uses base 2 rather than base 10."


@pytest.mark.parametrize(
    "explanation",
    (
        "While HCI involves several disciplines, the provided fact emphasizes its interdisciplinary nature.",
        "The central role of perception in interaction is explicitly stated in the provided fact.",
    ),
)
def test_cards_parser_preserves_unsafe_explanations_for_validation_retry(
    explanation,
):
    payload = json.loads(cards_json())
    payload["cards"][0]["expalanation"] = explanation

    card = parse_cards(json.dumps(payload))[0]

    assert card.expalanation == explanation
    assert "card 1 exposes provenance metadata" in validate_cluster(
        (card, *valid_cards()[1:]), valid_concept()
    )
    assert "correct answer here" not in card.expalanation


@pytest.mark.parametrize(
    "explanation",
    (
        "Base 2 is the correct answer here.",
        "This statement is true.",
        "This statement is false.",
    ),
)
def test_cluster_rejects_generic_explanation_filler(explanation):
    values = list(valid_cards())
    values[0] = replace(values[0], expalanation=explanation)

    errors = validate_cluster(tuple(values), valid_concept())

    assert "card 1 expalanation must explain the answer" in errors


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


def test_concept_plan_rejects_a_numbered_module_title_as_a_concept():
    raw, known = plan_json()
    payload = json.loads(raw)
    payload["concepts"][0]["name"] = "Module 3 Title"

    with pytest.raises(ValidationError, match="presentation/provenance metadata"):
        parse_concept_plan(json.dumps(payload), known)


def test_concept_plan_allows_module_as_a_domain_term():
    raw, known = plan_json()
    payload = json.loads(raw)
    payload["concepts"][0]["name"] = "Software Module Interfaces"
    known = (
        GraphFact("e1", "A software module exposes a public interface."),
        *known[1:],
    )

    concepts = parse_concept_plan(json.dumps(payload), known)

    assert concepts[0].name == "Software Module Interfaces"


def test_concept_plan_accepts_comma_separated_token_fields():
    raw, known = plan_json()
    payload = json.loads(raw)
    known = (
        *known,
        GraphFact("e21", "additional subject | supports | first concept"),
        GraphFact("e22", "another subject | supports | first concept"),
    )
    payload["concepts"][0]["fact_ids"] = "e1,e21,e22"
    payload["concepts"][0]["assessment_approaches"] = (
        "recall,comparison,classification,application,scenario analysis"
    )

    concepts = parse_concept_plan(json.dumps(payload), known)

    assert concepts[0].fact_ids == ("e1", "e21", "e22")
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


def test_concept_plan_allows_shared_context_when_each_concept_has_unique_anchor():
    raw, known = plan_json()
    payload = json.loads(raw)
    known = (*known, GraphFact("e21", "shared context | supports | both concepts"))
    payload["concepts"][1]["fact_ids"] = ["e2", "e21"]
    payload["concepts"][6]["fact_ids"] = ["e7", "e21"]

    concepts = parse_concept_plan(json.dumps(payload), known)

    assert concepts[1].fact_ids == ("e2", "e21")
    assert concepts[6].fact_ids == ("e7", "e21")


def test_concept_plan_rejects_concept_with_only_reused_fact_ids():
    raw, known = plan_json()
    payload = json.loads(raw)
    payload["concepts"][6]["fact_ids"] = ["e2"]

    with pytest.raises(ValidationError, match="no uniquely assigned anchor fact_id"):
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


def test_scenario_analysis_label_does_not_require_a_scenario_template():
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

    assert not any("scenario analysis" in error for error in errors)


@pytest.mark.parametrize(
    "field_value",
    (
        "To document a program",
        "A software module exposes a public interface",
        "A file stores the records",
        "A slide mechanism controls the position",
    ),
)
def test_cluster_allows_domain_uses_of_words_that_can_also_name_sources(
    field_value,
):
    values = list(valid_cards())
    values[0] = replace(values[0], wrong_option_3=field_value)

    errors = validate_cluster(tuple(values), valid_concept())

    assert "card 1 exposes provenance metadata" not in errors


@pytest.mark.parametrize(
    "question",
    (
        "Which answer is listed in fact f13?",
        "Which answer follows from the concept facts?",
        "Which topic appears in the provided vocabulary?",
        "Which claim follows from the already covered subjects?",
        "Which option came from suggested wrong option terms?",
    ),
)
def test_cluster_rejects_internal_generation_metadata(question):
    values = list(valid_cards())
    values[0] = replace(values[0], question=question)

    errors = validate_cluster(tuple(values), valid_concept())

    assert "card 1 exposes provenance metadata" in errors


def test_scenario_analysis_accepts_researcher_observing_users():
    values = list(valid_cards())
    approaches = list(APPROACHES)
    approaches[0] = "scenario analysis"
    values[0] = replace(
        values[0],
        question=(
            "A researcher observes that users struggle with a new interface. "
            "This scenario demonstrates which aspect of human capabilities?"
        ),
        assessment_approach="scenario analysis",
    )
    concept = replace(valid_concept(), assessment_approaches=tuple(approaches))

    assert validate_cluster(tuple(values), concept) == ()


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


def test_cluster_allows_relevant_distractors_that_are_absent_from_source_facts():
    values = list(valid_cards())
    values[0] = replace(
        values[0],
        wrong_option_1="Octal representation",
        wrong_option_2="Decimal notation",
        wrong_option_3="Hexadecimal encoding",
    )
    facts = (
        GraphFact(
            "e1",
            "binary uses base 2",
        ),
    )

    errors = validate_cluster(tuple(values), valid_concept(), facts)

    assert not any("not grounded" in error for error in errors)


@pytest.mark.parametrize(
    "options",
    (
        ("C", "C++", "C#", "Assembly"),
        ("x+1", "x-1", "x*1", "x/1"),
        ("Na+", "Na-", "Cl-", "H+"),
        ("f/2.8", "f/4", "1/60 s", "ISO 400"),
        ("RGB", "RGBA", "CMYK", "HSL"),
    ),
)
def test_multiple_choice_distinctness_preserves_meaningful_notation(options):
    values = list(valid_cards())
    values[0] = replace(
        values[0],
        correct_option=options[0],
        wrong_option_1=options[1],
        wrong_option_2=options[2],
        wrong_option_3=options[3],
    )

    errors = validate_cluster(tuple(values), valid_concept())

    assert "card 1 multiple-choice options must be distinct" not in errors


@pytest.mark.parametrize(
    ("left", "right"),
    (
        ("C++", " c ++ "),
        ("f/2.8", "F / 2.8"),
        ("x-1", "x \u2212 1"),
        ("Na+", "na +"),
    ),
)
def test_multiple_choice_distinctness_still_rejects_formatting_variants(
    left, right
):
    values = list(valid_cards())
    values[0] = replace(
        values[0],
        correct_option=left,
        wrong_option_1=right,
        wrong_option_2="Alternative B",
        wrong_option_3="Alternative C",
    )

    errors = validate_cluster(tuple(values), valid_concept())

    assert "card 1 multiple-choice options must be distinct" in errors


def test_short_identifier_does_not_leak_through_letters_inside_words():
    values = list(valid_cards())
    values[1] = replace(
        values[1],
        question="Which language is commonly used for systems programming?",
        correct_option="C",
        hint="Consider compiled languages used for systems software.",
    )

    errors = validate_cluster(tuple(values), valid_concept())

    assert "card 2 question reveals the identification answer" not in errors
    assert "card 2 hint reveals the correct answer" not in errors


@pytest.mark.parametrize(
    ("answer", "question", "hint"),
    (
        ("C", "What is C?", "The answer is C."),
        ("C++", "Which language is C ++?", "Recall C++ syntax."),
        ("x+1", "What expression is x + 1?", "Use x+1."),
    ),
)
def test_answer_leakage_still_detects_complete_identifiers_and_expressions(
    answer, question, hint
):
    values = list(valid_cards())
    values[1] = replace(
        values[1],
        question=question,
        correct_option=answer,
        hint=hint,
    )

    errors = validate_cluster(tuple(values), valid_concept())

    assert "card 2 question reveals the identification answer" in errors
    assert "card 2 hint reveals the correct answer" in errors


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
