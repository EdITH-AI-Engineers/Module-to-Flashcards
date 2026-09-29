import json

from flashcard_contract import MAX_FACT_IDS_PER_CONCEPT
from flashcard_schema import build_concept_plan_schema
from flashcard_types import GraphFact
from flashcard_validator import parse_concept_plan
from tests.test_flashcard_validator import APPROACHES


def test_concept_plan_schema_caps_fact_ids_without_exclusive_ownership():
    schema = build_concept_plan_schema(tuple(f"f{index}" for index in range(1, 21)))
    concept = schema["oneOf"][0]["properties"]["concepts"]["items"]
    fact_ids = concept["properties"]["fact_ids"]

    assert MAX_FACT_IDS_PER_CONCEPT == 8
    assert fact_ids["minItems"] == 1
    assert fact_ids["maxItems"] == 8


def test_complete_twenty_concept_plan_accepts_eight_fact_ids_per_concept():
    known_facts = tuple(
        (f"f{index}", f"Fact {index} states an explicit supported relationship.")
        for index in range(1, 21)
    )
    payload = {
        "concepts": [
            {
                "name": f"Concept {index}",
                "fact_ids": [f"f{fact_index}" for fact_index in range(1, 9)],
                "assessment_approaches": list(APPROACHES),
            }
            for index in range(1, 21)
        ]
    }

    concepts = parse_concept_plan(
        json.dumps(payload),
        tuple(GraphFact(fact_id, statement) for fact_id, statement in known_facts),
    )

    assert len(concepts) == 20
    assert all(len(concept.fact_ids) == 8 for concept in concepts)
    assert concepts[0].fact_ids == concepts[1].fact_ids


def test_concept_plan_parser_rejects_more_than_eight_fact_ids():
    facts = tuple(
        GraphFact(f"f{index}", f"Fact {index} states an explicit supported relationship.")
        for index in range(1, 10)
    )
    payload = {
        "concepts": [
            {
                "name": f"Concept {index}",
                "fact_ids": [fact.fact_id for fact in facts],
                "assessment_approaches": list(APPROACHES),
            }
            for index in range(1, 21)
        ]
    }

    try:
        parse_concept_plan(json.dumps(payload), facts)
    except Exception as exc:
        assert "at most 8" in str(exc)
    else:
        raise AssertionError("parser accepted nine fact IDs for one concept")
