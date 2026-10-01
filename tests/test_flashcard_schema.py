from flashcard_schema import (
    build_card_cluster_schema,
    build_cluster_batch_schema,
    build_numbered_card_repair_schema,
    build_single_card_schema,
)
from tests.test_flashcard_validator import valid_cards


def test_card_schema_omits_only_structural_constants():
    schema = build_card_cluster_schema(("recall", "application"))
    cards = schema["properties"]["cards"]
    assert cards["minItems"] == cards["maxItems"] == 5
    variants = {
        item["properties"]["type"]["enum"][0]: item
        for item in cards["items"]["oneOf"]
    }
    assert set(variants) == {"multiple-choice", "identification", "true-false"}
    assert "is_true" not in variants["multiple-choice"]["properties"]
    assert "wrong_option_1" in variants["multiple-choice"]["required"]
    assert "is_true" not in variants["identification"]["properties"]
    assert "wrong_option_1" not in variants["identification"]["properties"]
    assert variants["true-false"]["properties"]["is_true"]["enum"] == [0, 1]
    assert "correct_option" not in variants["true-false"]["properties"]
    assert all("expalanation" in item["required"] for item in variants.values())


def test_single_card_schema_locks_repair_type_and_metadata():
    original = valid_cards()[2]
    schema = build_single_card_schema(original)
    cards = schema["properties"]["cards"]
    assert cards["minItems"] == cards["maxItems"] == 1
    properties = cards["items"]["properties"]
    assert properties["type"]["enum"] == ["true-false"]
    assert properties["is_true"]["enum"] == [1]
    assert properties["difficulty"]["enum"] == [original.difficulty]
    assert properties["assessment_approach"]["enum"] == [original.assessment_approach]


def test_numbered_repair_schema_requests_only_flagged_positions():
    schema = build_numbered_card_repair_schema(("recall",), (2, 4))
    cards = schema["properties"]["cards"]
    assert cards["minItems"] == cards["maxItems"] == 2
    assert cards["items"]["properties"]["card_number"]["enum"] == [2, 4]
    assert len(cards["items"]["properties"]["card"]["oneOf"]) == 3


def test_batch_schema_preserves_five_cards_and_concept_numbers():
    schema = build_cluster_batch_schema(((1, ("recall",)), (2, ("application",))))
    clusters = schema["properties"]["clusters"]

    assert clusters["minItems"] == clusters["maxItems"] == 2
    variants = clusters["items"]["oneOf"]
    assert [item["properties"]["number"]["enum"] for item in variants] == [[1], [2]]
    assert all(
        item["properties"]["cards"]["minItems"]
        == item["properties"]["cards"]["maxItems"] == 5
        for item in variants
    )
    assert variants[0]["properties"]["cards"]["items"]["oneOf"][0]["properties"]["assessment_approach"]["enum"] == ["recall"]
