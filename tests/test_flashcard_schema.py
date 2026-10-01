from flashcard_schema import build_card_cluster_schema, build_single_card_schema
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
