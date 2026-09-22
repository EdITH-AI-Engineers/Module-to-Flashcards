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
Write clear, authentic college-level assessment items. Assess terminology, distinctions, relationships, mechanisms, processes, causes, effects, classifications, applications, implications, conditions, limitations, or technical reasoning only when the supplied facts support them. Difficulty must come from the required thinking, never confusing wording. Do not mechanically convert a fact into a stem or reveal an answer through its full definition. Assign each card whichever planned assessment approach accurately describes the reasoning it requires. Approaches may repeat, and no planned approach is required to appear. Prefer useful variety when the facts support it, but never mislabel a question merely to cover every approach.

ASSESSMENT APPROACH SEMANTICS
- recall: directly retrieve an explicitly supported term, property, relationship, or fact.
- comparison: reason about a supported similarity, difference, or contrast between at least two things.
- classification: determine a supported category or group from defining characteristics.
- application: use a supported rule, principle, process, or relationship to decide or solve something.
- scenario analysis: interpret a short, concrete situation and determine what it demonstrates, requires, or implies. A definition question such as "What term refers to..." is recall or classification, never scenario analysis.
- cause/effect: connect a supported cause with its effect or explain why a result follows.
- misconception detection: identify or correct a plausible but unsupported belief or relationship.
- conditions: determine the circumstances or requirements under which a supported claim holds.
- consequences: determine a supported outcome or implication.
- reversed reasoning: start from a supported result or property and infer the cause, rule, or concept behind it.
The assessment_approach label must describe the reasoning actually required by the question. Never attach a planned label to a question that uses a different approach.

EQUATION-BASED PROBLEM SOLVING
When the supplied facts contain an equation, formula, numerical relationship, or clearly defined quantities, include problem-solving questions when the selected assessment approach supports them. A problem-solving question may use a realistic, concrete scenario such as selecting a valid value, calculating an outcome, comparing results, or determining what changes when one supported quantity changes. Use only variables, units, relationships, and operations explicitly supplied by the facts; do not introduce outside constants, assumptions, or formulas. State every needed value in the question or supplied facts, use plain-text equation syntax, and ensure the answer follows deterministically from the available information. A scenario must test the equation or relationship, not add decorative context. Do not force a numerical problem when the source does not provide enough information.

Never mention a knowledge graph, concept fact, supplied fact, distractor pool, input field, source, module, document, lesson, slide, file, chunk, citation, URL, header, footer, or source reference in a question, answer, explanation, or hint. This applies to every field: question, correct_option, wrong_option_1, wrong_option_2, wrong_option_3, expalanation, and hint. The allowed wrong-option vocabulary is only for plausible wrong-option wording; never use it as authority for a correct answer, true-false decision, explanation, or hint. Refer to the topic itself, never to where it appeared. Bad: "What is the title of the slide that discusses the reading portfolio overview?" Good: "What overview precedes the parts and contents of the reading portfolio?" If a draft question would need the word slide, lesson, module, or document to make sense, rewrite it to name the topic directly instead.

ALLOWED TYPES
Use only multiple-choice, identification, and true-false. Vary the mix of these types from cluster to cluster; do not repeat the same type distribution in every cluster. Every cluster needs at least one multiple-choice, one identification, and one true-false card.

type describes a card's structural format, not its reasoning style. Never copy an assessment_approach value (recall, comparison, classification, application, scenario analysis, cause/effect, misconception detection, conditions, consequences, reversed reasoning) into the type field -- those are separate from type and go in assessment_approach only. In particular, scenario analysis is an assessment_approach, never a type. A card assigned scenario analysis must still use multiple-choice, identification, or true-false as its type and follow that type's field rules.

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

Scenario analysis assessment approach rules:
- Build a short, concrete situation only from the supplied facts and ask what it demonstrates, requires, or implies.
- Use any one of the three allowed types. The type controls the structure and fields: multiple-choice uses four options, identification uses one concise answer and empty wrong options, and true-false uses a declarative scenario statement with is_true.
- For multiple-choice, the situation may come before the question, as in "A user completes the task with fewer steps. This demonstrates what quality?" For identification, keep the required direct stem by placing the situation after What, Which, Who, Where, When, Why, or How. For true-false, express the situation and conclusion as a declarative statement.
- Set assessment_approach to "scenario analysis" and never set type to "scenario analysis".

DIRECT STEMS
For identification, use a natural direct form beginning with What, Which, Who, Where, When, Why, How, or What term. Multiple-choice may use the same direct form or put a concrete context or scenario before the question. Both types must ask a clear, answerable question ending in a question mark. Do not use wrappers such as According to, Based on, The material states, The following claim, Consider this statement, Evaluate this statement, Identify the concept associated with, or equivalents. The framings "Which of the following", "Which best describes", and "Which most accurately" are allowed when they produce a clear, answerable question. Avoid only vague or subjective wording that cannot be resolved from the supplied facts.

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

    Handing the model full facts from other concepts makes a small local model
    treat those facts as authority for the current question. Instead, give it
    only the literal, mechanically-checkable words the validator will accept
    in wrong options. This turns an abstract judgment call into a concrete
    constraint without exposing another concept's complete learning point.
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
    payload = _cluster_payload(
        identity,
        concept,
        concept_facts,
        distractor_facts,
        prior_signals,
    )
    approach_list = "\n".join(
        f'- "{approach}"' for approach in concept.assessment_approaches
    )
    prior_guidance = ""
    if "already_covered_subjects" in payload:
        prior_guidance = """
    - already_covered_subjects contains compact subject/answer fingerprints from
      earlier cards. Do not test the same subject expecting the same answer."""

    return f"""Generate exactly {CARDS_PER_CLUSTER} cards for this concept.

    PLANNED ASSESSMENT APPROACHES
    {approach_list}
    For each card, select whichever listed approach best describes the actual
    reasoning required. Approaches may repeat, and a listed approach does not
    have to appear. The list order is not the card order, and no approach is
    tied to a numbered card position. Do not mislabel a recall or definition
    question merely to force approach coverage.

    EVIDENCE SCOPE
    - Questions, correct answers, explanations, and hints must be supported by
      concept_facts.
    - Each multiple-choice wrong_option must be plausible but incorrect for its
      question and must contain at least one exact substantive term from
      allowed_wrong_option_terms. That list is for wrong-option wording only;
      never use it as evidence for a correct answer, question, explanation, or
      hint. Do not invent outside topics, examples, or technologies.
    - When the evidence supplies an equation or numerical relationship, an
      assigned application or scenario approach may test it using only supplied
      variables, values, units, and operations.
    {prior_guidance}

    REPEATED CRITICAL RULES
    - Use only multiple-choice, identification, and true-false as type values.
      Include at least one of each. Scenario analysis is an assessment_approach,
      never a type.
    - Multiple-choice: correct_option and all three wrong_option values must be
      four different strings; no wrong_option may repeat or closely restate
      another option. Set is_true to null.
    - Identification: wrong_option_1, wrong_option_2, and wrong_option_3
      must literally be "". Set is_true to null. The question must not reveal the
      answer or its abbreviation.
    - True-false: use a declarative statement, keep all option fields "", and
      set is_true to integer 0 or 1.
    - Every question must be clear and end in ? except true-false statements.
      Do not begin with provenance wrappers such as "According to" or "Based on".
    - Never write knowledge graph, source, module, document, lesson, slide, file,
      chunk, citation, or URL in any card field. Name the topic directly.
    - Keep all eleven keys in every object and preserve the spelling expalanation.
    - Return one complete JSON object only, with no Markdown or surrounding text.

    Required top-level shape:
    {{"cards":[...exactly {CARDS_PER_CLUSTER} complete card objects...]}}

    Every card object must contain exactly:
    type, question, correct_option, wrong_option_1, wrong_option_2,
    wrong_option_3, is_true, expalanation, hint, difficulty,
    assessment_approach

    Scenario analysis may use any allowed structural type, for example:
    {{"type":"multiple-choice","question":"A user encounters a supported condition. Which concept applies?","correct_option":"...","wrong_option_1":"...","wrong_option_2":"...","wrong_option_3":"...","is_true":null,"expalanation":"...","hint":"...","difficulty":3,"assessment_approach":"scenario analysis"}}
    {{"type":"identification","question":"What concept applies when a user encounters the supported condition?","correct_option":"...","wrong_option_1":"","wrong_option_2":"","wrong_option_3":"","is_true":null,"expalanation":"...","hint":"...","difficulty":3,"assessment_approach":"scenario analysis"}}
    {{"type":"true-false","question":"A user encountering the supported condition demonstrates the stated relationship.","correct_option":"","wrong_option_1":"","wrong_option_2":"","wrong_option_3":"","is_true":1,"expalanation":"...","hint":"...","difficulty":3,"assessment_approach":"scenario analysis"}}

    INPUT JSON:
    """ + _json(payload)


def _cluster_payload(
    identity: ModuleIdentity,
    concept: ConceptPlan,
    concept_facts: Sequence[GraphFact],
    distractor_facts: Sequence[GraphFact],
    prior_signals: Sequence[str] = (),
) -> dict[str, object]:
    """Return bounded cluster input without duplicating resolved fact text."""

    payload = {
        "course_code": identity.course_code,
        "module_number": identity.module_number,
        "concept": {
            "name": concept.name,
            "fact_ids": list(concept.fact_ids),
            "assessment_approaches": list(concept.assessment_approaches),
        },
        "concept_facts": [
            {"fact_id": fact.fact_id, "statement": fact.statement}
            for fact in concept_facts
        ],
        "allowed_wrong_option_terms": list(
            grounded_vocabulary(concept, concept_facts, distractor_facts)[:60]
        ),
    }
    condensed_prior, omitted_prior = _condensed_prior_signals(prior_signals)
    if condensed_prior:
        payload["already_covered_subjects"] = condensed_prior
        if omitted_prior:
            payload["omitted_prior_subject_count"] = omitted_prior
    return payload


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
of its words is copied exactly from this list and remains genuinely incorrect
for the question. This vocabulary is for wrong-option wording only. Do not
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
- When a validation error names a card number, correct that exact card.
- Set each assessment_approach to one of the planned values and make the label
  match the question's actual reasoning. Approaches may repeat.
- For identification, wrong_option_1, wrong_option_2, and wrong_option_3 must be all exactly "".
- For multiple-choice, the correct option and three wrong options must be four different strings.
- Rewrite any multiple-choice or identification question that begins with
  "According to", "Based on", or another wrapper.
- Identification must begin directly with What, Which, Who, Where, When, Why, or How. Multiple-choice may instead begin with a concrete context or scenario and then ask the question.
- Example:
  Invalid: "According to the design rules, which measure assesses effectiveness?"
  Valid: "Which measure assesses effectiveness?"
- Preserve valid cards.
- Return a complete replacement as the full cards JSON only.
- If an error says "exposes provenance metadata", remove expressions such
  as "according to the module", "based on the supplied material", and
  "as described in the lesson". Describe the topic directly without
  mentioning where the information came from.
- If an error says a wrong_option "is not grounded in supplied module facts",
  rewrite only that wrong_option using an exact substantive term from the
  supplied grounded vocabulary while keeping the option plausible and
  incorrect. The vocabulary is not evidence for the question, correct answer,
  explanation, or hint. Do not introduce an outside example or technology.
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


def build_cluster_retry_prompt(
    identity: ModuleIdentity,
    concept: ConceptPlan,
    concept_facts: Sequence[GraphFact],
    distractor_facts: Sequence[GraphFact],
    prior_signals: Sequence[str],
    candidate: str | None,
    errors: Iterable[str],
) -> str:
    """Build a self-contained cluster retry without nesting the first prompt."""

    error_list = [str(error) for error in errors]
    condensed, omitted = _condensed_errors(error_list)
    error_bullets = "\n".join(f"- {error}" for error in condensed)
    if omitted:
        error_bullets += f"\n- (+{omitted} similar errors omitted)"

    payload = _cluster_payload(
        identity,
        concept,
        concept_facts,
        distractor_facts,
        prior_signals,
    )

    grounding_block = ""
    if any(
        "not grounded in supplied module facts" in error for error in error_list
    ):
        vocabulary = grounded_vocabulary(concept, concept_facts, distractor_facts)
        shown = list(vocabulary)[:MAX_RETRY_VOCAB_TERMS]
        payload["grounded_vocabulary_for_retry"] = shown
        overflow = len(vocabulary) - len(shown)
        if overflow:
            payload["omitted_grounded_vocabulary_count"] = overflow
        grounding_block = """
    - For each flagged ungrounded wrong_option, replace only that field with a
      plausible incorrect option containing an exact substantive term from
      grounded_vocabulary_for_retry. That vocabulary is for wrong options only,
      not evidence for correct content."""

    rejected_block = ""
    if candidate is not None:
        rejected_json = candidate
        if len(rejected_json) > MAX_RETRY_CANDIDATE_CHARS:
            rejected_json = rejected_json[:MAX_RETRY_CANDIDATE_CHARS].rstrip()
        rejected_block = f"""

    REJECTED JSON TO CORRECT:
    {rejected_json}"""

    return f"""Regenerate one complete replacement five-card JSON object for the supplied concept.

    VALIDATION ERRORS
    {error_bullets}

    CORRECTIONS
    - Return exactly {CARDS_PER_CLUSTER} complete cards. Set each
      assessment_approach to a planned value that matches the question's actual
      reasoning. Approaches may repeat; no planned value is required to appear.
    - Use only multiple-choice, identification, and true-false types, including
      at least one of each. Scenario analysis is an approach, never a type.
    - Identification and true-false option fields must be "". Multiple-choice
      must contain one correct option and three distinct incorrect options.
    - Questions, answers, explanations, and hints must use concept_facts.
      Multiple-choice distractors must use allowed_wrong_option_terms rather
      than invented outside content; those terms are not answer authority.
    - If scenario analysis is flagged, rewrite that card around a concrete
      supported situation that requires interpretation. A direct definition
      question is recall or classification, not scenario analysis. You may
      change the approach label only when it then accurately describes the
      question.
    - Remove provenance wording and keep the answer out of identification stems.
    - Include all eleven required fields and preserve expalanation spelling.
    - Return JSON only with the shape {{"cards":[...]}}.{grounding_block}

    INPUT JSON:
    {_json(payload)}{rejected_block}
    """


def build_duplicate_repair_prompt(
    identity: ModuleIdentity,
    concept: ConceptPlan,
    cards: Sequence[FlashcardDraft],
    reasons: Sequence[str],
    avoid_questions: Sequence[str],
) -> str:
    """Build a focused final-pass request that paraphrases duplicate cards."""

    payload = {
        "course_code": identity.course_code,
        "module_number": identity.module_number,
        "concept": concept.name,
        "duplicate_findings": list(reasons),
        "cards_to_repair": [asdict(card) for card in cards],
        "questions_to_avoid": list(avoid_questions),
    }
    return f"""Paraphrase the duplicate card or cards in this five-card cluster.

    REPAIR RULES
    - When duplicate_findings names card numbers, rewrite those cards and copy
      every unflagged card unchanged. If no card number is named, paraphrase all
      five question stems.
    - Preserve each card's learning point, factual meaning, correct-answer
      meaning, incorrectness of wrong answers, type, is_true value, difficulty,
      and assessment_approach.
    - Natural paraphrases may use words that do not appear verbatim in any
      corpus or fact text. Do not introduce a new claim or change which answer
      is correct.
    - Make every repaired question genuinely distinct from questions_to_avoid.
      Changing only the opening question word or adding filler is insufficient.
    - Keep the eleven required fields and return the complete five-card JSON
      object only. Preserve the required key spelling expalanation.

    Required shape: {{"cards":[...exactly {CARDS_PER_CLUSTER} cards...]}}

    INPUT JSON:
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
    return """Compare all question stems for semantic and near duplication. Flag only clusters containing questions that assess the same learning point in substantially the same way as another question. Every reason must identify both duplicate card numbers in the form "card <number> duplicates cluster <uuid> card <number>". Do not judge factual correctness in this pass and do not rewrite questions.

    Return only this JSON shape. Use an empty issues list when no duplicate exists:
    {"issues":[{"cluster":"valid UUID copied from input","reasons":["card 2 duplicates cluster <uuid> card 4"]}]}

    INPUT JSON:
    """ + _json(
        payload
    )
