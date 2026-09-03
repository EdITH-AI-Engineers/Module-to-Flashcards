import json
from dataclasses import asdict, replace

from flashcard_types import (
    ConceptPlan,
    FlashcardCluster,
    FlashcardDraft,
    GraphFact,
)


APPROACHES = (
    "recall",
    "comparison",
    "application",
    "misconception detection",
    "reversed reasoning",
)


def make_concept(index: int) -> ConceptPlan:
    return ConceptPlan(
        name=f"Concept {index} topic{index} alpha{index} beta{index}",
        fact_ids=(f"e{index}",),
        facts=(f"subject{index} | relates to | object{index}",),
        assessment_approaches=APPROACHES,
    )


def make_cards(index: int, revision: int = 0) -> tuple[FlashcardDraft, ...]:
    tag = f"topic{index} alpha{index} beta{index} revision{revision}"
    return (
        FlashcardDraft(
            type="multiple-choice",
            question=f"Which classification applies to {tag}?",
            correct_option=f"Classification {index}",
            wrong_option_1=f"Alternative {index}A",
            wrong_option_2=f"Alternative {index}B",
            wrong_option_3=f"Alternative {index}C",
            is_true=None,
            expalanation=f"The relationship supports classification {index}.",
            hint=f"Compare the available classifications for item {index}.",
            difficulty=1,
            assessment_approach="recall",
        ),
        FlashcardDraft(
            type="identification",
            question=f"What term names item {tag}?",
            correct_option=f"Term {index}",
            wrong_option_1="",
            wrong_option_2="",
            wrong_option_3="",
            is_true=None,
            expalanation=f"Term {index} names the item.",
            hint=f"Recall the name paired with item {index}.",
            difficulty=1,
            assessment_approach="comparison",
        ),
        FlashcardDraft(
            type="true-false",
            question=f"Item {tag} has its stated relationship.",
            correct_option="",
            wrong_option_1="",
            wrong_option_2="",
            wrong_option_3="",
            is_true=1,
            expalanation=f"The relationship for item {index} is stated.",
            hint=f"Check how item {index} is connected.",
            difficulty=1,
            assessment_approach="misconception detection",
        ),
        FlashcardDraft(
            type="multiple-choice",
            question=f"How should {tag} be applied in the stated relationship?",
            correct_option=f"Apply relationship {index}",
            wrong_option_1=f"Reverse relationship {index}",
            wrong_option_2=f"Discard relationship {index}",
            wrong_option_3=f"Replace relationship {index}",
            is_true=None,
            expalanation=f"Applying relationship {index} preserves the stated connection.",
            hint=f"Follow the connection assigned to item {index}.",
            difficulty=2,
            assessment_approach="application",
        ),
        FlashcardDraft(
            type="true-false",
            question=f"Reversing {tag} always preserves its relationship.",
            correct_option="",
            wrong_option_1="",
            wrong_option_2="",
            wrong_option_3="",
            is_true=0,
            expalanation=f"The relationship for item {index} is directional.",
            hint=f"Consider the direction assigned to item {index}.",
            difficulty=2,
            assessment_approach="reversed reasoning",
        ),
    )


def valid_clusters() -> tuple[FlashcardCluster, ...]:
    return tuple(
        FlashcardCluster(
            cluster=f"00000000-0000-4000-8000-{index:012d}",
            concept=make_concept(index),
            cards=make_cards(index),
        )
        for index in range(1, 21)
    )


def graph_facts() -> tuple[GraphFact, ...]:
    return tuple(
        GraphFact(f"e{index}", f"subject{index} | relates to | object{index}")
        for index in range(1, 21)
    )


def plan_json() -> str:
    return json.dumps(
        {
            "concepts": [
                {
                    "name": make_concept(index).name,
                    "fact_ids": list(make_concept(index).fact_ids),
                    "assessment_approaches": list(APPROACHES),
                }
                for index in range(1, 21)
            ]
        }
    )


def cluster_json(index: int, revision: int = 0) -> str:
    return json.dumps({"cards": [asdict(card) for card in make_cards(index, revision)]})


def with_question(
    clusters: tuple[FlashcardCluster, ...],
    cluster_index: int,
    card_index: int,
    question: str,
) -> tuple[FlashcardCluster, ...]:
    changed = list(clusters)
    target = changed[cluster_index]
    cards = list(target.cards)
    cards[card_index] = replace(cards[card_index], question=question)
    changed[cluster_index] = replace(target, cards=tuple(cards))
    return tuple(changed)
