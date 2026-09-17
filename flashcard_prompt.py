from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Iterable, Sequence

from flashcard_contract import CARDS_PER_CLUSTER, CONCEPTS_PER_MODULE
from flashcard_types import (
    ConceptPlan,
    FlashcardCluster,
    FlashcardDraft,
    GraphFact,
    ModuleIdentity,
)
from flashcard_validator import _grounding_tokens

SYSTEM_PROMPT = f"""You are a college-level educational assessment generator.

SOURCE AUTHORITY
Use only the supplied graph facts as factual authority. Do not add outside knowledge, repair a fact from memory, or infer unsupported facts. Treat each module independently. Never assess the same underlying learning point twice. If the facts cannot support the requested number of distinct concepts, report insufficient content instead of duplicating or inventing material.

INTERNAL OUTPUT
Return JSON only for every internal request. Do not use Markdown fences, CSV, headings, commentary, or text outside the requested JSON object. Use exactly the requested keys and value types. Preserve the required key spelling expalanation.

CARD FIELD CONTRACT
Every card, with no exceptions, must include all eleven fields: type, question, correct_option, wrong_option_1, wrong_option_2, wrong_option_3, is_true, expalanation, hint, difficulty, assessment_approach. Before returning JSON, verify every card object has exactly these eleven keys. If any card is missing a key, add it before responding.

QUESTION QUALITY
Write clear, authentic college-level assessment items. Assess terminology, distinctions, relationships, mechanisms, processes, causes, effects, classifications, applications, implications, conditions, limitations, or technical reasoning only when the supplied facts support them. Difficulty must come from the required thinking, never confusing wording. Do not mechanically convert a fact into a stem or reveal an answer through its full definition. Within a concept, use {CARDS_PER_CLUSTER} meaningfully different assessment approaches. Changes limited to wording, names, punctuation, order, or distractors are not distinct approaches.

EQUATION-BASED PROBLEM SOLVING
When the supplied facts contain an equation, formula, numerical relationship, or clearly defined quantities, include problem-solving questions when the selected assessment approach supports them. A problem-solving question may use a realistic, concrete scenario such as selecting a valid value, calculating an outcome, comparing results, or determining what changes when one supported quantity changes. Use only variables, units, relationships, and operations explicitly supplied by the facts; do not introduce outside constants, assumptions, or formulas. State every needed value in the question or supplied facts, use plain-text equation syntax, and ensure the answer follows deterministically from the available information. A scenario must test the equation or relationship, not add decorative context. Do not force a numerical problem when the source does not provide enough information.

Never mention a knowledge graph, source, module, document, lesson, slide, file, chunk, citation, URL, header, footer, or source reference in a question, answer, explanation, or hint. This applies to every field: question, correct_option, wrong_option_1, wrong_option_2, wrong_option_3, expalanation, and hint. Refer to the topic itself, never to where it appeared. Bad: "What is the title of the slide that discusses the reading portfolio overview?" Good: "What overview precedes the parts and contents of the reading portfolio?" If a draft question would need the word slide, lesson, module, or document to make sense, rewrite it to name the topic directly instead.

ALLOWED TYPES
Use only multiple-choice, identification, true-false, and scenario analysis. Vary the mix of these types from cluster to cluster; do not repeat the same type distribution in every cluster. Every cluster still needs at least one multiple-choice, one identification, and one true-false card; scenario analysis is an additional option on top of those three, not a replacement for any of them.

type describes a card's structural format, not its reasoning style. Never copy an assessment_approach value (recall, comparison, classification, application, cause/effect, misconception detection, conditions, consequences, reversed reasoning) into the type field -- those are separate from type and go in assessment_approach only. The one exception is scenario analysis, which is both a listed type above and a listed assessment_approach: use it as the type only when the card is actually built as a scenario followed by a question, per the rules below.

Multiple-choice rules:
- Supply exactly one concise correct_option and three plausible, distinct, incorrect options.
- correct_option, wrong_option_1, wrong_option_2, and wrong_option_3 must be four textually different strings. Never let a wrong_option repeat, restate, or closely paraphrase the correct_option or another wrong_option within the same card.
- Set is_true to null.
- Do not place choices, option labels, or the answer in the question.

Identification rules:
- Supply one concise identifiable term, name, concept, classification, principle, process, figure, or title as correct_option.
- The answer must be a short phrase, not a sentence or explanation.
- Set wrong_option_1, wrong_option_2, wrong_option_3 to empty strings and is_true to null. These three fields must literally be "" â€” do not place any distractor words, related terms, or partial answers there, even though that pattern is normal for multiple-choice.
- Ask directly without embedding the answer or its full definition in the stem.
- Neither the complete answer nor its abbreviation may appear anywhere in the question, even as part of a longer phrase. Describe the concept by its function, purpose, defining trait, or relationships, never by restating its name. Bad: "What is the term for the field that focuses on the design of computer technology and human-computer interaction?" (repeats the answer "human-computer interaction"). Good: "What term describes the field concerned with designing computer systems that are efficient, safe, comfortable, and enjoyable to use?" If the supplied fact defining the concept restates the concept's own name, paraphrase around the name instead of quoting the fact.

True-false rules:
- Write only a declarative statement in question.
- For true-false cards, correct_option, wrong_option_1, wrong_option_2, and wrong_option_3 must ALL be empty strings. The only fields that carry the answer are is_true (0 or 1) and expalanation.
- Set is_true to integer 1 for true or integer 0 for false.
- Do not add True or False, labels, or evaluation instructions.
- False items must state a plausible misconception or incorrect relationship that the supplied facts resolve.

Scenario analysis rules:
- Open question with a short, concrete scenario (one or two sentences) built only from the supplied facts, then ask a direct question about what that scenario demonstrates, requires, or implies.
- Supply exactly one concise correct_option and three plausible, distinct, incorrect options, exactly like multiple-choice.
- correct_option, wrong_option_1, wrong_option_2, and wrong_option_3 must be four textually different strings. Never let a wrong_option repeat, restate, or closely paraphrase the correct_option or another wrong_option within the same card.
- Set is_true to null.
- Do not place choices, option labels, or the answer in the question.
- Unlike multiple-choice, do not open with What/Which/Who/etc. -- the scenario sentence comes first, and the question follows it.

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


_VOCAB_EXCLUDED_WORDS = {
    "knowledge",
    "graph",
    "source",
    "module",
    "lesson",
    "slide",
    "slides",
    "document",
    "file",
    "chunk",
    "citation",
    "url",
    "content",
}


def grounded_vocabulary(
    concept: ConceptPlan,
    concept_facts: Sequence[GraphFact],
    distractor_facts: Sequence[GraphFact],
) -> tuple[str, ...]:
    """Compute the exact set of words a wrong_option is allowed to draw its
    substantive content from, using the identical tokenization the
    validator applies (flashcard_validator._grounding_tokens).

    Handing the model a vague instruction like "adapt a term from
    distractor_pool" is too abstract for a small local model to reliably
    follow -- it tends to fall back on plausible-sounding but invented
    content instead. Giving it the literal, mechanically-checkable word
    list the validator will test against turns an abstract judgment call
    into a concrete constraint: "every wrong_option must contain at least
    one exact word from this list."
    """
    terms: set[str] = set()
    for fact in concept_facts:
        terms.update(_grounding_tokens(fact.statement))
    for fact in distractor_facts:
        terms.update(_grounding_tokens(fact.statement))
    terms.update(_grounding_tokens(concept.name))
    for fact_text in concept.facts:
        terms.update(_grounding_tokens(fact_text))
    # Some corpus "facts" are really mislabeled section headers (e.g. a
    # statement that is literally "Module 1: Introduction to HCI"). Words
    # like "module" or "lesson" are still technically present in the source
    # text, but handing them to the model as valid grounding vocabulary
    # would just cause a wrong_option to trip the separate provenance-
    # wording ban instead of the grounding one, so they're excluded here.
    terms -= _VOCAB_EXCLUDED_WORDS
    return tuple(sorted(terms))


def _concept_payload(concept: ConceptPlan) -> dict[str, object]:
    return {
        "name": concept.name,
        "fact_ids": list(concept.fact_ids),
        "facts": list(concept.facts),
        "assessment_approaches": list(concept.assessment_approaches),
    }


PRIOR_QUESTION_LIMIT = 50
PRIOR_QUESTION_CHARS = 80
PRIOR_QUESTION_CHAR_BUDGET = 1200

_LEADING_QUESTION_WORDS = re.compile(
    r"^(?:what\s+term|what|which\s+of\s+the\s+following|which|who|where|when|why|how)\b"
    r"(?:\s+(?:is|are|was|were|does|do|did|best\s+describes?|most\s+accurately\s+describes?))?\s*",
    re.I,
)
_TRAILING_PROVENANCE = re.compile(
    r"\s*(?:according to|based on|as\s+(?:stated|described)\s+in)\s+the\s+"
    r"(?:provided|supplied|given)?\s*(?:facts?|material|module|content|text|information)\s*\.?\s*$",
    re.I,
)


def _card_subject(question: str) -> str:
    """Reduce a question to a short subject phrase for dedup context.

    Strips the interrogative stem ("What is", "Which of the following
    is", ...), a leading article, and trailing provenance filler
    ("according to the provided facts"), then keeps only the first
    handful of remaining words -- the part of the question that actually
    names what it is asking about, not the grammatical wrapper around it.
    """
    text = re.sub(r"\s+", " ", question).strip().rstrip("?").strip()
    text = _TRAILING_PROVENANCE.sub("", text).strip()
    text = _LEADING_QUESTION_WORDS.sub("", text).strip()
    text = re.sub(r"^(?:a|an|the)\s+", "", text, flags=re.I).strip()
    words = text.split()
    subject = " ".join(words[:8])
    return subject or text[:40]


def _card_answer(card: FlashcardDraft) -> str:
    """Reduce a card's answer to a short phrase for dedup context."""
    if card.type == "true-false":
        return "true" if card.is_true else "false"
    words = card.correct_option.strip().split()
    return " ".join(words[:4])


def _dedup_fingerprint(card: FlashcardDraft) -> str:
    """Compact subject/answer signature identifying what a card actually
    tests, for use as cheap dedup context in later cluster prompts.

    Two cards are duplicates because they test the same subject with the
    same answer, not because they share exact wording -- so this is both
    cheaper to send and, if anything, a more robust duplicate signal than
    the full question text would be.
    """
    return f"{_card_subject(card.question)} -> {_card_answer(card)}"


def _condensed_prior_signals(signals: Sequence[str]) -> tuple[list[str], int]:
    """Dedupe and bound already-generated dedup signals before they go into
    a cluster prompt.

    A naive "include everything asked so far" list doesn't just grow with
    the module -- it gets resent on every remaining cluster call, so its
    real cost across a 20-concept module compounds roughly quadratically,
    not linearly. Two things keep the per-call cost flat instead:
    - a fixed total CHARACTER BUDGET for the field (not just a per-item
      truncation and an item-count cap), so a single call's cost has a hard
      ceiling no matter how large the module gets;
    - preferring the MOST RECENT signals over the earliest ones. Adjacent
      clusters are drawn from nearby, overlapping source facts and are the
      far more likely source of an actual near-duplicate, so recent signals
      are higher-value context per character spent than early ones -- which
      a plain head-of-list cap would keep instead.
    Anything trimmed here is still caught by the near-duplicate check in
    validate_module; this list is a cheap first line of defense, not the
    only one.
    """
    deduped: list[str] = []
    seen: set[str] = set()
    for signal in signals:
        text = re.sub(r"\s+", " ", str(signal)).strip()
        if not text:
            continue
        if len(text) > PRIOR_QUESTION_CHARS:
            text = text[:PRIOR_QUESTION_CHARS].rstrip() + "..."
        if text not in seen:
            seen.add(text)
            deduped.append(text)

    kept_reverse: list[str] = []
    budget = PRIOR_QUESTION_CHAR_BUDGET
    for text in reversed(deduped):
        if len(kept_reverse) >= PRIOR_QUESTION_LIMIT:
            break
        cost = len(text) + 1
        if kept_reverse and cost > budget:
            break
        budget -= cost
        kept_reverse.append(text)

    kept = list(reversed(kept_reverse))
    return kept, len(deduped) - len(kept)


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
    return (
        f"""Select exactly {CONCEPTS_PER_MODULE} distinct, explicitly supported concepts for this module.
    Each concept must be assessable in {CARDS_PER_CLUSTER} genuinely different ways. Keep concepts semantically distinct and do not use presentation or provenance details as concepts.

    For each concept, copy one or more fact_ids exactly from the input. Do not copy or rewrite fact statements; Python will resolve the selected IDs to their exact statements.{context_guidance} Choose exactly {CARDS_PER_CLUSTER} distinct approaches from: recall, comparison, classification, application, scenario analysis, cause/effect, misconception detection, conditions, consequences, reversed reasoning.

    Return one JSON object whose top-level key is "concepts" and whose value is an array. The array must contain exactly {CONCEPTS_PER_MODULE} concept objects before its closing bracket. Every concept object has these keys: name (string), fact_ids (non-empty string array), and assessment_approaches (array of exactly {CARDS_PER_CLUSTER} distinct allowed approaches). Do not treat a one-object shape illustration as a complete answer.

    Fill all {CONCEPTS_PER_MODULE} positions in this checklist before closing the concepts array:
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20.
    Do not stop after 11 or 12 objects. Do not add a 21st object. Each position must have a unique concept name.
    Use the JSON key "fact_ids" literally, with a normal underscore and no backslash. Every concept must copy at least one exact fact_id from graph_facts.

    Only if fewer than {CONCEPTS_PER_MODULE} distinct concepts are genuinely supported, return an object with the single key insufficient_content. Its value must specifically state how many concepts are supportable and why, using at least five words. Never copy generic placeholder wording into that field.
    """
        + overlap_guidance
        + """
    INPUT JSON:
    """
        + _json(payload)
    )


def build_cluster_prompt(
    identity: ModuleIdentity,
    concept: ConceptPlan,
    concept_facts: Sequence[GraphFact],
    distractor_facts: Sequence[GraphFact],
    prior_signals: Sequence[str] = (),
) -> str:
    vocabulary = grounded_vocabulary(concept, concept_facts, distractor_facts)
    payload = {
        "course_code": identity.course_code,
        "module_number": identity.module_number,
        "concept": _concept_payload(concept),
        "concept_facts": [
            {"fact_id": fact.fact_id, "statement": fact.statement}
            for fact in concept_facts
        ],
        "distractor_pool": [
            {"fact_id": fact.fact_id, "statement": fact.statement}
            for fact in distractor_facts
        ],
        "grounded_vocabulary": list(vocabulary),
    }
    condensed_prior, omitted_prior = _condensed_prior_signals(prior_signals)
    overlap_guidance = ""
    if condensed_prior:
        payload["already_covered_subjects"] = condensed_prior
        overlap_guidance = (
            '\nalready_covered_subjects lists "subject -> answer" fingerprints '
            "for cards already generated elsewhere in this module -- abbreviated "
            "for brevity, not the literal wording used. Do not write a new card "
            "whose question asks about the same subject expecting the same "
            "answer as one of these, even if the wording, framing, or card type "
            "is different.\n"
        )
        if omitted_prior:
            overlap_guidance += (
                f"({omitted_prior} additional earlier fingerprint(s) omitted "
                "here for brevity -- avoiding the pattern above avoids those "
                "too.)\n"
            )
    assignments = list(enumerate(concept.assessment_approaches, start=1))
    approach_list = "\n".join(
        f'- Card {index}: assessment_approach must be exactly "{approach}"'
        for index, approach in assignments
    )
    approach_checklist = ", ".join(
        f"card {index}={approach}" for index, approach in assignments
    )
    return f"""Generate exactly {CARDS_PER_CLUSTER} assessment cards for the one supplied concept.
    Assign assessment approaches by position, one approach per card, with no repeats and no substitutions:
    {approach_list}
    Self-check mapping before you respond: {approach_checklist}. Every card's assessment_approach value must match its required entry above exactly; it must not duplicate another card's approach and must not use an approach absent from this list.
    {overlap_guidance}
    Include at least one multiple-choice, one identification, and one true-false card among the {CARDS_PER_CLUSTER}; vary the other two cards naturally, using scenario analysis where it fits. Every claim, correct answer, distractor judgment, explanation, and hint must be resolvable using only the supplied facts. Every wrong_option must be a plausible-but-incorrect term drawn from elsewhere in the provided module content; never invent a topic, term, or fact absent from the supplied facts.
    If the supplied facts include an equation, formula, numerical relationship, or clearly defined quantities, use a realistic problem-solving scenario for an appropriate approach when the facts provide enough information. The scenario may ask the learner to calculate, select, compare, or reason about a supported result. Use only supplied variables, units, values, operations, and relationships; state any needed values explicitly; and do not invent constants, assumptions, formulas, or numerical data. Do not force a numerical problem when the facts are insufficient.

    Questions, correct answers, explanations, and hints must use only concept_facts.

    GROUNDING WORD LIST (mechanical rule, not a style suggestion)
    grounded_vocabulary is the complete, exact list of substantive words this
    concept's cards are allowed to draw wrong_option content from. It is
    computed directly from concept_facts, distractor_pool, and the concept
    name -- nothing else. For every multiple-choice wrong_option (including
    scenario, application, and comparison cards), at least one content word
    in that wrong_option (ignoring articles, prepositions, and connective
    words) must appear, in some form, in grounded_vocabulary. Build each
    wrong_option by starting from one specific item in distractor_pool or
    concept_facts and lightly rephrasing it -- do not compose a wrong_option
    out of words that all fall outside grounded_vocabulary, even if the
    result sounds like a normal, plausible course topic. A plausible-sounding
    wrong_option that reuses none of grounded_vocabulary's words WILL be
    rejected regardless of how reasonable it sounds; there is no partial
    credit for "sounds academically similar."

    Concretely, before writing a wrong_option: (1) pick one entry from
    distractor_pool (or, if the concept's own facts contain more than one
    idea, concept_facts) that is a genuinely wrong answer to this specific
    question; (2) reuse at least one exact word from that entry's statement;
    (3) reshape the rest into a natural-sounding option. If nothing in
    distractor_pool or concept_facts yields a workable wrong answer for the
    approach you were assigned, write the card using a different but still
    assigned-correct assessment_approach framing that a distractor_pool item
    does support, rather than inventing unrelated content.

    A distractor must be incorrect for the current question. Do not invent a
    term that is absent from both concept_facts and distractor_pool.

    REPEATED CRITICAL RULES” re-verify each of these on every card before responding:
    - Count the {CARDS_PER_CLUSTER} cards' types before finalizing: at least one must be multiple-choice, one identification, and one true-false. Scenario analysis fills an additional slot on top of those three -- it must never be the type for two or more cards in the same cluster, and it must never be used in place of the true-false card. If your draft has two scenario analysis cards and no true-false card, convert the one whose assessment_approach is NOT "scenario analysis" into a true-false card (a declarative statement using the same underlying claim), keeping its assigned assessment_approach unchanged.
    - Multiple-choice: correct_option, wrong_option_1, wrong_option_2, and wrong_option_3 must be four textually different strings; no wrong_option may repeat or closely restate the correct_option or another wrong_option.
    - Identification: wrong_option_1, wrong_option_2, and wrong_option_3 must literally be "" â€” no words, no distractor terms.
    - Never write knowledge graph, source, source material, module, document, lesson, slide, slides, file, chunk, citation, or url in question, correct_option, wrong_option_1, wrong_option_2, wrong_option_3, expalanation, or hint. Name the topic itself instead of where it appeared.
    - Never refer to where information came from.
    - Never use phrases such as "provided fact", "supplied facts", "module content", "source material", "document", or "lesson".
    - Write hints about the topic itself.
    - Ensure all distractors and correct answers are directly from the supplied facts. Do not invent a term, process, or relationship absent from the supplied facts.
    - For scenario analysis, application, or comparison cards, a wrong_option must still be adapted from an item in distractor_pool (or concept_facts), never a newly invented real-world example, technology, or activity. Wanting a concrete-sounding wrong option is not license to introduce content absent from the supplied facts.
    - Before finalizing each multiple-choice card, check every wrong_option word by word against grounded_vocabulary. If a wrong_option's content words are all absent from grounded_vocabulary, discard it and build a new one starting from an actual distractor_pool or concept_facts entry, per the GROUNDING WORD LIST steps above.
    - Do not put the answer, option labels, or choices in the question. Do not embed the answer in the question stem.
    - Identification: the question text must not contain correct_option's wording anywhere, even as part of a longer phrase (e.g. a question about "Human-Computer Interaction" must not itself contain the words "human-computer interaction"). Describe the concept by its function, purpose, defining trait, or relationships instead of naming it. If the concept's own defining fact restates its name, paraphrase around the name rather than quoting the fact.
    - If already_covered_subjects is present, check every new card's subject and answer against it before responding; do not submit a card matching one of those fingerprints under different wording.

    Invalid:
    "Compare the fields mentioned in the module content."

    Valid:
    "Compare how each field approaches human interaction."

    Invalid (identification, correct_option "Human-Computer Interaction"):
    "What is the term for the field that focuses on the design of computer technology and human-computer interaction?"

    Valid (identification, correct_option "Human-Computer Interaction"):
    "What term describes the field concerned with designing computer systems people can use effectively, safely, and enjoyably?"

    Invalid (scenario-analysis wrong_option, invented and not grounded):
    "Developing a new programming language" / "Creating a database management system"
    (these share zero words with grounded_vocabulary -- "programming",
    "language", "database", and "management" appear nowhere in concept_facts
    or distractor_pool for this concept, however plausible they sound as
    generic computer-science distractors.)

    Valid (scenario-analysis wrong_option, adapted from an actual distractor_pool item):
    a paraphrase of a real entry from distractor_pool -- e.g. if distractor_pool
    describes HCI's history in the 1980s, a valid wrong option is a scenario
    built from that item ("Recounting how HCI methods developed in the 1980s"),
    not an unrelated invented example. This works because "HCI", "1980s", and
    "developed"/"emerged" trace back to words that actually appear in
    grounded_vocabulary.

    Return this JSON shape with exactly {CARDS_PER_CLUSTER} objects in cards, in card-position order matching the mapping above. Every object must contain exactly these eleven keys:
    type, question, correct_option, wrong_option_1, wrong_option_2, wrong_option_3, is_true, expalanation, hint, difficulty, assessment_approach

    Example of the required top-level shape (expand cards to exactly {CARDS_PER_CLUSTER} objects):
    {{"cards":[{{"type":"multiple-choice","question":"...","correct_option":"...","wrong_option_1":"...","wrong_option_2":"...","wrong_option_3":"...","is_true":null,"expalanation":"...","hint":"...","difficulty":2,"assessment_approach":"recall"}}]}}

    Example of a complete true-false card:
    {{"type":"true-false","question":"The stated relationship is supported.","correct_option":"","wrong_option_1":"","wrong_option_2":"","wrong_option_3":"","is_true":1,"expalanation":"The supplied facts support the relationship.","hint":"Check the relationship itself.","difficulty":1,"assessment_approach":"recall"}}

    Example of a complete scenario analysis card (scenario sentence first, then the question -- not a bare What/Which stem):
    {{"type":"scenario analysis","question":"A team redesigns a workflow so users finish tasks with minimal training. Which measure would this primarily assess?","correct_option":"...","wrong_option_1":"...","wrong_option_2":"...","wrong_option_3":"...","is_true":null,"expalanation":"...","hint":"...","difficulty":3,"assessment_approach":"scenario analysis"}}

    For identification and true-false cards, keep all eleven keys and use empty strings for fields that do not apply.

    Use empty strings for fields that the selected type requires to be empty. Use JSON null only where the type rules require null. Before returning your answer, verify every card has all eleven keys, that difficulty and assessment_approach are present, that the assessment_approach mapping above is followed exactly, and that both REPEATED CRITICAL RULES above hold for every card.

    INPUT JSON:
    """ + _json(
        payload
    )


MAX_RETRY_ERROR_COUNT = 12
MAX_RETRY_ERROR_CHARS = 220
MAX_RETRY_CANDIDATE_CHARS = 6000


def _condensed_errors(errors: Sequence[str]) -> tuple[list[str], int]:
    """Dedupe and cap a validation-error list before it goes back to the model.

    A single bad generation can trigger the same kind of error many times
    over (e.g. several concepts each reusing an already-assigned fact), and
    some individual error messages embed a full source statement that can
    run to 100+ words. Left unbounded, a retry prompt built from that list
    -- plus the original prompt and the full rejected candidate -- can
    exceed a local model's context window outright, turning a normal
    validation retry into a hard backend crash instead of another attempt.
    """
    seen: list[str] = []
    for error in errors:
        text = re.sub(r"\s+", " ", str(error)).strip()
        if len(text) > MAX_RETRY_ERROR_CHARS:
            text = text[:MAX_RETRY_ERROR_CHARS].rstrip() + "..."
        if text not in seen:
            seen.append(text)
    omitted = max(0, len(seen) - MAX_RETRY_ERROR_COUNT)
    return seen[:MAX_RETRY_ERROR_COUNT], omitted


MAX_RETRY_VOCAB_TERMS = 80


def build_retry_prompt(
    original_prompt: str,
    candidate: str | None,
    errors: Iterable[str],
    grounded_vocabulary: Sequence[str] = (),
) -> str:
    error_list = [str(error) for error in errors]
    condensed, omitted = _condensed_errors(error_list)
    error_bullets = "\n".join(f"- {error}" for error in condensed)
    if omitted:
        error_bullets += (
            f"\n- (+{omitted} more validation errors of a similar kind, "
            "omitted here for brevity -- fixing the pattern above resolves them too)"
        )

    grounding_block = ""
    has_grounding_error = any(
        "not grounded in supplied module facts" in error for error in error_list
    )
    if has_grounding_error and grounded_vocabulary:
        shown = list(grounded_vocabulary)[:MAX_RETRY_VOCAB_TERMS]
        overflow = len(grounded_vocabulary) - len(shown)
        vocab_text = ", ".join(shown)
        if overflow > 0:
            vocab_text += f", ... (+{overflow} more)"
        grounding_block = f"""

GROUNDED VOCABULARY FOR THE FLAGGED WRONG_OPTION(S):
{vocab_text}

The wrong_option(s) named above as "not grounded" contain NO word from this
list. This is not a matter of degree -- go through the flagged wrong_option
word by word, and if not one of its content words (ignore "a", "the", "of",
"and", etc.) appears in the list above, the option is invalid no matter how
plausible or academically reasonable it sounds. Rewrite it so at least one
of its words is copied exactly from this list, taken from a distractor_pool
or concept_facts entry that is genuinely incorrect for the question. Do not
substitute a different but equally ungrounded invented term."""

    rejected_json = candidate or "{}"
    if len(rejected_json) > MAX_RETRY_CANDIDATE_CHARS:
        rejected_json = (
            rejected_json[:MAX_RETRY_CANDIDATE_CHARS].rstrip()
            + "\n... (truncated; regenerate the full JSON from the ORIGINAL REQUEST, not from this partial excerpt)"
        )

    return f"""Correct the rejected JSON below.

VALIDATION ERRORS:
{error_bullets}
{grounding_block}

MANDATORY CORRECTIONS:
- Rewrite any multiple-choice or identification question that begins with
  "According to", "Based on", or another wrapper.
- It must begin directly with What, Which, Who, Where, When, Why, or How.
- Example:
  Invalid: "According to the design rules, which measure assesses effectiveness?"
  Valid: "Which measure assesses effectiveness?"
- Preserve valid cards.
- Return the complete cards JSON only.
- If an error says "exposes provenance metadata", remove expressions such
  as "provided facts", "supplied facts", and "module content". Describe the
  topic directly without mentioning where the information came from.
- If an error says a wrong_option "is not grounded in supplied module facts",
  that option was invented rather than adapted from distractor_pool. Rewrite
  only that wrong_option, keeping it plausible but built from a term, event,
  principle, or example that literally appears in distractor_pool or
  concept_facts for this concept -- even for a scenario-style question. Do
  not introduce a real-world example, technology, or activity absent from
  the supplied facts.
- If several errors say a fact or concept duplicates one already assigned
  elsewhere, do not rename or reorder it -- pick a different concept
  grounded in fact_ids that no other concept has used.
- If an error says a question "reveals the identification answer", the
  question text repeats correct_option's exact wording. Rewrite only the
  question so it describes the concept by its function, purpose, defining
  trait, or relationships -- do not restate the term, name, or its
  abbreviation anywhere in the question, even inside a longer phrase.
  Keep correct_option unchanged.
  Example: correct_option "Human-Computer Interaction" ->
  Invalid: "What is the term for the field that focuses on ... human-computer interaction?"
  Valid: "What term describes the field concerned with designing computer systems people can use effectively, safely, and enjoyably?"

ORIGINAL REQUEST:
{original_prompt}

REJECTED JSON:
{rejected_json}
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
    """ + _json(
        payload
    )


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
    """ + _json(
        payload
    )
