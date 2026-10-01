from __future__ import annotations

from collections.abc import Sequence

from flashcard_contract import CARDS_PER_CLUSTER, CONCEPTS_PER_MODULE
from flashcard_types import FlashcardDraft
from flashcard_validator import ALLOWED_APPROACHES


def _strict_object(
    properties: dict[str, object],
    required: Sequence[str],
) -> dict[str, object]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def build_concept_plan_schema(fact_ids: Sequence[str]) -> dict[str, object]:
    """Constrain planning output to either a complete plan or one refusal."""

    concept = _strict_object(
        {
            "name": {"type": "string", "minLength": 1},
            "fact_ids": {
                "type": "array",
                "items": {"type": "string", "enum": list(fact_ids)},
                "minItems": 1,
            },
            "assessment_approaches": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": sorted(ALLOWED_APPROACHES),
                },
                "minItems": CARDS_PER_CLUSTER,
                "maxItems": CARDS_PER_CLUSTER,
            },
        },
        ("name", "fact_ids", "assessment_approaches"),
    )
    plan = _strict_object(
        {
            "concepts": {
                "type": "array",
                "items": concept,
                "minItems": CONCEPTS_PER_MODULE,
                "maxItems": CONCEPTS_PER_MODULE,
            }
        },
        ("concepts",),
    )
    insufficient = _strict_object(
        {"insufficient_content": {"type": "string", "minLength": 5}},
        ("insufficient_content",),
    )
    return {"oneOf": [plan, insufficient]}


def build_card_cluster_schema(
    assessment_approaches: Sequence[str],
) -> dict[str, object]:
    """Require five complete card shapes while omitting structural constants."""

    common = {
        "question": {"type": "string"},
        "expalanation": {"type": "string"},
        "hint": {"type": "string"},
        "difficulty": {"type": "integer", "enum": [1, 2, 3]},
        "assessment_approach": {
            "type": "string",
            "enum": list(assessment_approaches),
        },
    }
    card_variants: list[dict[str, object]] = []
    for card_type, specific in (
        (
            "multiple-choice",
            {
                "correct_option": {"type": "string"},
                "wrong_option_1": {"type": "string"},
                "wrong_option_2": {"type": "string"},
                "wrong_option_3": {"type": "string"},
            },
        ),
        ("identification", {"correct_option": {"type": "string"}}),
        ("true-false", {"is_true": {"type": "integer", "enum": [0, 1]}}),
    ):
        properties = {"type": {"type": "string", "enum": [card_type]}, **common, **specific}
        card_variants.append(_strict_object(properties, tuple(properties)))
    card = {"oneOf": card_variants}
    return _strict_object(
        {
            "cards": {
                "type": "array",
                "items": card,
                "minItems": CARDS_PER_CLUSTER,
                "maxItems": CARDS_PER_CLUSTER,
            }
        },
        ("cards",),
    )


def build_single_card_schema(
    original_card: FlashcardDraft,
) -> dict[str, object]:
    """Require one repair card while grammar-locking only controlled values."""

    schema = build_card_cluster_schema((original_card.assessment_approach,))
    cards = schema["properties"]["cards"]
    cards["minItems"] = 1
    cards["maxItems"] = 1
    variants = cards["items"]["oneOf"]
    selected = next(
        variant
        for variant in variants
        if variant["properties"]["type"]["enum"] == [original_card.type]
    )
    cards["items"] = selected
    properties = selected["properties"]
    for field in (
        "type",
        "is_true",
        "difficulty",
        "assessment_approach",
    ):
        if field in properties:
            properties[field] = {"enum": [getattr(original_card, field)]}
    if original_card.type == "true-false":
        for field in (
            "correct_option",
            "wrong_option_1",
            "wrong_option_2",
            "wrong_option_3",
        ):
            if field in properties:
                properties[field] = {"enum": [getattr(original_card, field)]}
    elif original_card.type == "identification":
        for field in ("wrong_option_1", "wrong_option_2", "wrong_option_3"):
            if field in properties:
                properties[field] = {"enum": [getattr(original_card, field)]}
    return schema


def build_numbered_card_repair_schema(
    assessment_approaches: Sequence[str], card_numbers: Sequence[int]
) -> dict[str, object]:
    """Constrain a partial retry to numbered replacement cards only."""
    if not card_numbers or len(set(card_numbers)) != len(card_numbers):
        raise ValueError("card_numbers must be a non-empty unique sequence")
    card = build_card_cluster_schema(assessment_approaches)["properties"]["cards"]["items"]
    entry = _strict_object(
        {
            "card_number": {"type": "integer", "enum": sorted(card_numbers)},
            "card": card,
        },
        ("card_number", "card"),
    )
    return _strict_object(
        {
            "cards": {
                "type": "array",
                "items": entry,
                "minItems": len(card_numbers),
                "maxItems": len(card_numbers),
            }
        },
        ("cards",),
    )


def build_cluster_batch_schema(
    numbered_approaches: Sequence[tuple[int, Sequence[str]]],
) -> dict[str, object]:
    """Constrain batched output to numbered, independently shaped clusters."""
    if not numbered_approaches:
        raise ValueError("batch needs at least one concept")
    variants = []
    for number, approaches in numbered_approaches:
        cards = build_card_cluster_schema(approaches)["properties"]["cards"]
        variants.append(
            _strict_object(
                {
                    "number": {"type": "integer", "enum": [number]},
                    "cards": cards,
                },
                ("number", "cards"),
            )
        )
    return _strict_object(
        {
            "clusters": {
                "type": "array",
                "items": {"oneOf": variants},
                "minItems": len(numbered_approaches),
                "maxItems": len(numbered_approaches),
            }
        },
        ("clusters",),
    )


def build_review_schema(known_clusters: Sequence[str]) -> dict[str, object]:
    issue = _strict_object(
        {
            "cluster": {
                "type": "string",
                "enum": sorted(known_clusters),
            },
            "reasons": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
            },
        },
        ("cluster", "reasons"),
    )
    return _strict_object(
        {"issues": {"type": "array", "items": issue}},
        ("issues",),
    )
