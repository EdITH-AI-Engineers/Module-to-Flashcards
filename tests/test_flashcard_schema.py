import json

from flashcard_schema import build_concept_plan_schema
from flashcard_types import GraphFact
from flashcard_validator import parse_concept_plan
from tests.test_flashcard_validator import APPROACHES


def test_concept_plan_schema_requires_five_supported_card_targets():
    schema = build_concept_plan_schema(tuple(f"f{index}" for index in range(1, 13)))
    concept = schema["oneOf"][0]["properties"]["concepts"]["items"]
    targets = concept["properties"]["card_targets"]
    fact_ids = targets["items"]["properties"]["fact_ids"]

    assert concept["required"] == ["name", "card_targets"]
    assert targets["minItems"] == targets["maxItems"] == 5
    assert fact_ids["minItems"] == 1
    assert "maxItems" not in fact_ids


def test_complete_twenty_concept_plan_accepts_more_than_eight_fact_ids():
    known_facts = tuple(
        (f"f{index}", f"Fact {index} states an explicit supported relationship.")
        for index in range(1, 13)
    )
    payload = {
        "concepts": [
            {
                "name": f"Concept {index}",
                "card_targets": [
                    {
                        "learning_point": f"Concept {index} learning point {position}",
                        "fact_ids": [
                            f"f{fact_index}" for fact_index in range(1, 13)
                        ],
                        "assessment_approach": approach,
                    }
                    for position, approach in enumerate(APPROACHES, start=1)
                ],
            }
            for index in range(1, 21)
        ]
    }

    concepts = parse_concept_plan(
        json.dumps(payload),
        tuple(GraphFact(fact_id, statement) for fact_id, statement in known_facts),
    )

    assert len(concepts) == 20
    assert all(len(concept.fact_ids) == 12 for concept in concepts)
    assert concepts[0].fact_ids == concepts[1].fact_ids
    assert len(concepts[0].card_targets) == 5
