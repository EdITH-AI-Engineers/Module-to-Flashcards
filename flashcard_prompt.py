from __future__ import annotations

import json
from dataclasses import asdict
from typing import Iterable, Sequence

from flashcard_contract import CARDS_PER_CLUSTER, CONCEPTS_PER_MODULE
from flashcard_types import (
    ConceptPlan,
    FlashcardCluster,
    GraphFact,
    ModuleIdentity,
)


SYSTEM_PROMPT = f"""You are a college-level educational assessment generator.

SOURCE AUTHORITY
Use only the supplied graph facts as factual authority. Do not add outside knowledge, repair a fact from memory, or infer unsupported facts. Treat each module independently. Never assess the same underlying learning point twice. If the facts cannot support the requested number of distinct concepts, report insufficient content instead of duplicating or inventing material.

INTERNAL OUTPUT
Return JSON only for every internal request. Do not use Markdown fences, CSV, headings, commentary, or text outside the requested JSON object. Use exactly the requested keys and value types. Preserve the required key spelling expalanation.

CARD FIELD CONTRACT
Every card, with no exceptions, must include all eleven fields: type, question, correct_option, wrong_option_1, wrong_option_2, wrong_option_3, is_true, expalanation, hint, difficulty, assessment_approach. Before returning JSON, verify every card object has exactly these eleven keys. If any card is missing a key, add it before responding.

QUESTION QUALITY
Write clear, authentic college-level assessment items. Assess terminology, distinctions, relationships, mechanisms, processes, causes, effects, classifications, applications, implications, conditions, limitations, or technical reasoning only when the supplied facts support them. Difficulty must come from the required thinking, never confusing wording. Do not mechanically convert a fact into a stem or reveal an answer through its full definition. Within a concept, use {CARDS_PER_CLUSTER} meaningfully different assessment approaches. Changes limited to wording, names, punctuation, order, or distractors are not distinct approaches.

Never mention a knowledge graph, source, module, document, lesson, slide, file, chunk, citation, URL, header, footer, or source reference in a question, answer, explanation, or hint.

ALLOWED TYPES
Use only multiple-choice, identification, and true-false.

Multiple-choice rules:
- Supply exactly one concise correct_option and three plausible, distinct, incorrect options.
- Set is_true to null.
- Do not place choices, option labels, or the answer in the question.

Identification rules:
- Supply one concise identifiable term, name, concept, classification, principle, process, figure, or title as correct_option.
- The answer must be a short phrase, not a sentence or explanation.
- Set wrong_option_1, wrong_option_2, wrong_option_3 to empty strings and is_true to null.
- Ask directly without embedding the answer or its full definition in the stem.

True-false rules:
- Write only a declarative statement in question.
- For true-false cards, correct_option, wrong_option_1, wrong_option_2, and wrong_option_3 must ALL be empty strings. The only fields that carry the answer are is_true (0 or 1) and expalanation.
- Set is_true to integer 1 for true or integer 0 for false.
- Do not add True or False, labels, or evaluation instructions.
- False items must state a plausible misconception or incorrect relationship that the supplied facts resolve.

DIRECT STEMS
For multiple-choice and identification, use a natural direct form beginning with What, Which, Who, Where, When, Why, How, or What term. Do not use wrappers such as According to, Based on, The material states, The following claim, Consider this statement, Evaluate this statement, Identify the concept associated with, or equivalents. The framings "Which of the following", "Which best describes", and "Which most accurately" are allowed when they produce a clear, answerable question. Avoid only vague or subjective wording that cannot be resolved from the supplied facts.

DIFFICULTY
Use integer 1 only for recall or straightforward understanding. Use integer 2 for interpretation, comparison, classification, application, or distinction. Use integer 3 for analysis, complex application, multi-step reasoning, competing explanations, or an unfamiliar but fully supported scenario.

EXPLANATION AND HINT
In expalanation, briefly explain why the answer is correct. For a false statement, identify or correct the error when useful. In hint, give a useful clue about the relevant relationship, distinction, process, condition, or reasoning path without stating the answer. Neither field may mention provenance or presentation metadata.

EQUATIONS
Use plain text only: + - * / ^ = < > <= >= sqrt(...) ( ). Do not use LaTeX, MathJax, HTML math, images, superscript glyphs, or unsupported notation.

Before returning JSON, silently check every item against the supplied facts and all requested constraints. Never expose that check.
"""


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _concept_payload(concept: ConceptPlan) -> dict[str, object]:
    return {
        "name": concept.name,
        "fact_ids": list(concept.fact_ids),
        "facts": list(concept.facts),
        "assessment_approaches": list(concept.assessment_approaches),
    }

def build_concept_plan_prompt(
    identity: ModuleIdentity,
    facts: Sequence[GraphFact],
    prior_concept_names: Sequence[str] = (),
) -> str:
    payload: dict[str, object] = {
        "course_code": identity.course_code,
        "module_number": identity.module_number,
        "graph_facts": [
            {
                "fact_id": fact.fact_id,
                "statement": fact.statement,
                **({"topic": fact.topic} if fact.topic else {}),
                **({"slides": list(fact.slides)} if fact.slides else {}),
            }
            for fact in facts
        ],
    }
    overlap_guidance = ""
    context_guidance = ""
    if any(fact.topic or fact.slides for fact in facts):
        context_guidance = (
            " The optional topic and slides fields provide context and provenance, "
            "not additional facts."
        )
    if prior_concept_names:
        payload["previously_covered_concepts"] = list(prior_concept_names)
        overlap_guidance = (
            "\nAvoid selecting a concept that assesses the same underlying learning "
            "point as any entry in previously_covered_concepts, even if phrased "
            "differently. Prefer concepts distinctive to this module's own facts.\n"
        )
    return f"""Select exactly {CONCEPTS_PER_MODULE} distinct, explicitly supported concepts for this module.
Each concept must be assessable in {CARDS_PER_CLUSTER} genuinely different ways. Keep concepts semantically distinct and do not use presentation or provenance details as concepts.

For each concept, copy one or more fact_ids exactly from the input. Do not copy or rewrite fact statements; Python will resolve the selected IDs to their exact statements.{context_guidance} Choose exactly {CARDS_PER_CLUSTER} distinct approaches from: recall, comparison, classification, application, scenario analysis, cause/effect, misconception detection, conditions, consequences, reversed reasoning.

Return one JSON object whose top-level key is "concepts" and whose value is an array. The array must contain exactly {CONCEPTS_PER_MODULE} concept objects before its closing bracket. Every concept object has these keys: name (string), fact_ids (non-empty string array), and assessment_approaches (array of exactly {CARDS_PER_CLUSTER} distinct allowed approaches). Do not treat a one-object shape illustration as a complete answer.

Fill all {CONCEPTS_PER_MODULE} positions in this checklist before closing the concepts array:
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20.
Do not stop after 11 or 12 objects. Do not add a 21st object. Each position must have a unique concept name.
Use the JSON key "fact_ids" literally, with a normal underscore and no backslash. Every concept must copy at least one exact fact_id from graph_facts.

Only if fewer than {CONCEPTS_PER_MODULE} distinct concepts are genuinely supported, return an object with the single key insufficient_content. Its value must specifically state how many concepts are supportable and why, using at least five words. Never copy generic placeholder wording into that field.
""" + overlap_guidance + """
INPUT JSON:
""" + _json(payload)


def build_cluster_prompt(
    identity: ModuleIdentity,
    concept: ConceptPlan,
    module_facts: Sequence[GraphFact] = (),
) -> str:
    payload = {
        "course_code": identity.course_code,
        "module_number": identity.module_number,
        "concept": _concept_payload(concept),
        "module_facts": [
            {"fact_id": fact.fact_id, "statement": fact.statement}
            for fact in module_facts
        ],
    }
    approach_list = "\n".join(
        f"- {approach}" for approach in concept.assessment_approaches
    )
    return f"""Generate exactly {CARDS_PER_CLUSTER} assessment cards for the one supplied concept.
This cluster must produce exactly {CARDS_PER_CLUSTER} cards using each of these assessment approaches exactly once:
{approach_list}
Include at least one multiple-choice, one identification, and one true-false card; vary the other two types naturally. Every claim, correct answer, distractor judgment, explanation, and hint must be resolvable using only the supplied facts. Every wrong_option must be a plausible-but-incorrect term drawn from elsewhere in the provided module content; never invent a topic, term, or fact absent from the supplied facts.

Return this JSON shape with exactly {CARDS_PER_CLUSTER} objects in cards. Every object must contain exactly these eleven keys:
type, question, correct_option, wrong_option_1, wrong_option_2, wrong_option_3, is_true, expalanation, hint, difficulty, assessment_approach

Example of the required top-level shape (expand cards to exactly {CARDS_PER_CLUSTER} objects):
{{"cards":[{{"type":"multiple-choice","question":"...","correct_option":"...","wrong_option_1":"...","wrong_option_2":"...","wrong_option_3":"...","is_true":null,"expalanation":"...","hint":"...","difficulty":2,"assessment_approach":"recall"}}]}}

Example of a complete true-false card:
{{"type":"true-false","question":"The stated relationship is supported.","correct_option":"","wrong_option_1":"","wrong_option_2":"","wrong_option_3":"","is_true":1,"expalanation":"The supplied facts support the relationship.","hint":"Check the relationship itself.","difficulty":1,"assessment_approach":"recall"}}

For identification and true-false cards, keep all eleven keys and use empty strings for fields that do not apply.

Use empty strings for fields that the selected type requires to be empty. Use JSON null only where the type rules require null. Before returning your answer, verify every card has all eleven keys and that difficulty and assessment_approach are present.

INPUT JSON:
""" + _json(payload)


def build_retry_prompt(
    original_prompt: str,
    candidate: str | None,
    errors: Iterable[str],
) -> str:
    payload: dict[str, object] = {"validation_errors": list(errors)}
    if candidate is not None:
        payload["rejected_candidate"] = candidate
    error_text = "; ".join(str(error) for error in errors)
    return f"""The previous response was rejected: {error_text}
Return a complete replacement for the full card set, not a patch, explanation, or commentary. Ensure every card includes all eleven required fields, especially difficulty and assessment_approach, and correct every listed error without relaxing any original rule.

ORIGINAL REQUEST:
{original_prompt}

VALIDATION DETAILS:
{_json(payload)}
"""


def build_grounding_review_prompt(
    clusters: Sequence[FlashcardCluster],
) -> str:
    payload = {
        "clusters": [
            {
                "cluster": cluster.cluster,
                "concept": _concept_payload(cluster.concept),
                "cards": [asdict(card) for card in cluster.cards],
            }
            for cluster in clusters
        ]
    }
    return f"""Review these generated clusters against only their supplied facts. Flag a cluster if any card contains an unsupported claim, answer leakage, an invalid distractor, a misleading explanation, or insufficient variation among its {CARDS_PER_CLUSTER} assessment approaches. Do not rewrite cards.

Return only this JSON shape. Use an empty issues list when no defect exists:
{{"issues":[{{"cluster":"valid UUID copied from input","reasons":["unsupported claim"]}}]}}

INPUT JSON:
""" + _json(payload)


def build_duplicate_review_prompt(
    clusters: Sequence[FlashcardCluster],
) -> str:
    payload = {
        "clusters": [
            {
                "cluster": cluster.cluster,
                "concept": cluster.concept.name,
                "questions": [card.question for card in cluster.cards],
            }
            for cluster in clusters
        ]
    }
    return """Compare all question stems for semantic and near duplication. Flag only clusters containing questions that assess the same learning point in substantially the same way as another question. Do not judge factual correctness in this pass and do not rewrite questions.

Return only this JSON shape. Use an empty issues list when no duplicate exists:
{"issues":[{"cluster":"valid UUID copied from input","reasons":["semantic duplication with cluster <uuid>"]}]}

INPUT JSON:
""" + _json(payload)
