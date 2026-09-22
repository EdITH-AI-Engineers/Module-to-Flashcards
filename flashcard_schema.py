from __future__ import annotations

from collections.abc import Sequence

from flashcard_contract import CARDS_PER_CLUSTER, CONCEPTS_PER_MODULE
from flashcard_validator import ALLOWED_APPROACHES, ALLOWED_TYPES


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
    """Require exactly five card objects with the exact eleven field names."""

    card = _strict_object(
        {
            "type": {"type": "string", "enum": sorted(ALLOWED_TYPES)},
            "question": {"type": "string"},
            "correct_option": {"type": "string"},
            "wrong_option_1": {"type": "string"},
            "wrong_option_2": {"type": "string"},
            "wrong_option_3": {"type": "string"},
            "is_true": {"type": ["integer", "null"]},
            "expalanation": {"type": "string"},
            "hint": {"type": "string"},
            "difficulty": {"type": "integer", "enum": [1, 2, 3]},
            "assessment_approach": {
                "type": "string",
                "enum": list(assessment_approaches),
            },
        },
        (
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
        ),
    )
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
    assessment_approaches: Sequence[str],
) -> dict[str, object]:
    """Require one complete card for a location-preserving repair."""

    schema = build_card_cluster_schema(assessment_approaches)
    cards = schema["properties"]["cards"]
    cards["minItems"] = 1
    cards["maxItems"] = 1
    return schema


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
