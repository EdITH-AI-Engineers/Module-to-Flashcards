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
from flashcard_validator import _contains_metadata_artifact

SYSTEM_PROMPT = f"""You are a college-level educational assessment generator.

SOURCE AUTHORITY
Use only the supplied graph facts as factual authority for questions, correct answers, true-false decisions, explanations, and hints. Do not add outside knowledge, repair a fact from memory, or infer unsupported claims in those fields. Multiple-choice distractors may use common closely related domain terms that are absent from the facts, but absence from the facts is never proof that an option is incorrect. Treat each module independently. Never assess the same underlying learning point twice. If the facts cannot support the requested number of distinct concepts, report insufficient content instead of duplicating or inventing material.

INTERNAL OUTPUT
Return JSON only for every internal request. Do not use Markdown fences, CSV, headings, commentary, or text outside the requested JSON object. Use exactly the requested keys and value types. Preserve the required key spelling expalanation.

CARD FIELD CONTRACT
Every card, with no exceptions, must include all eleven fields: type, question, correct_option, wrong_option_1, wrong_option_2, wrong_option_3, is_true, expalanation, hint, difficulty, assessment_approach. Before returning JSON, verify every card object has exactly these eleven keys. If any card is missing a key, add it before responding.

QUESTION QUALITY
Write clear, authentic college-level assessment items. Assess terminology, distinctions, relationships, mechanisms, processes, causes, effects, classifications, applications, implications, conditions, limitations, or technical reasoning only when the supplied facts support them. Difficulty must come from the required thinking, never confusing wording. Do not mechanically convert a fact into a stem or reveal an answer through its full definition. The five cards must be meaningfully distinct learning checks. They may assess the same concept from supported angles, but changing only the card type, opening phrase, or a single word does not create a different question. Asking the negated form (the exception instead of the member) does. Assign each card whichever planned assessment approach accurately describes the reasoning it requires. Approaches may repeat, and no planned approach is required to appear. Prefer useful variety when the facts support it, but never mislabel a question merely to cover every approach.

ASSESSMENT APPROACH SEMANTICS
- recall: directly retrieve an explicitly supported term, property, relationship, or fact.
- comparison: reason about a supported similarity, difference, or contrast between at least two things.
- classification: determine a supported category or group from defining characteristics.
- application: use a supported rule, principle, process, or relationship to decide or solve something.
- scenario analysis: interpret supported conditions, a case, or an example when that framing naturally helps assessment. Do not invent a story merely to use this label. A definition question such as "What term refers to..." is recall or classification.
- cause/effect: connect a supported cause with its effect or explain why a result follows.
- misconception detection: identify or correct a plausible but unsupported belief or relationship.
- conditions: determine the circumstances or requirements under which a supported claim holds.
- consequences: determine a supported outcome or implication.
- reversed reasoning: start from a supported result or property and infer the cause, rule, or concept behind it.
The assessment_approach label must describe the reasoning actually required by the question. Never attach a planned label to a question that uses a different approach.

EQUATION-BASED PROBLEM SOLVING
When the supplied facts contain an equation, formula, numerical relationship, or clearly defined quantities, include problem-solving questions when the selected assessment approach supports them. A problem-solving question may use a realistic, concrete scenario such as selecting a valid value, calculating an outcome, comparing results, or determining what changes when one supported quantity changes. Use only variables, units, relationships, and operations explicitly supplied by the facts; do not introduce outside constants, assumptions, or formulas. State every needed value in the question or supplied facts, use plain-text equation syntax, and ensure the answer follows deterministically from the available information. A scenario must test the equation or relationship, not add decorative context. Do not force a numerical problem when the source does not provide enough information.

Never mention internal generation details or cite the evidence container in a question, answer, explanation, or hint. Do not write phrases such as "fact f13," "concept facts," "provided vocabulary," "already covered subjects," "according to the module," or "as stated in the document." Ordinary subject-matter uses of words such as module, document, file, slide, source, citation, and URL are allowed. Refer to the topic itself, never to where the information appeared.

ALLOWED TYPES
Use only multiple-choice, identification, and true-false. Vary the mix of these types from cluster to cluster; do not repeat the same type distribution in every cluster. Every cluster needs at least one multiple-choice, one identification, and one true-false card.

type describes a card's structural format, not its reasoning style. Never copy an assessment_approach value (recall, comparison, classification, application, scenario analysis, cause/effect, misconception detection, conditions, consequences, reversed reasoning) into the type field -- those are separate from type and go in assessment_approach only. In particular, scenario analysis is an assessment_approach, never a type. A card assigned scenario analysis must still use multiple-choice, identification, or true-false as its type and follow that type's field rules.

Multiple-choice rules:
- Supply exactly one concise correct_option and three plausible, distinct, incorrect options.
- Exactly one option may satisfy the question. Never use another supported or arguably correct statement as a wrong option.
- Never use another valid member of the requested category as a distractor. If a broad category has several valid candidates, narrow the question with a supported distinguishing property or use distractors that clearly do not satisfy it.
- correct_option, wrong_option_1, wrong_option_2, and wrong_option_3 must be four textually different strings. Never let a wrong_option repeat, restate, or closely paraphrase the correct_option or another wrong_option within the same card.
- Each wrong_option must match the correct answer's semantic category and answer shape, remain relevant to the question and module domain, and be unambiguously incorrect. It may use a familiar related term that is not written verbatim in the supplied facts.
- Prefer positive questions. Do not use NOT or EXCEPT to turn an invented claim into the correct answer.
- Set is_true to null.
- Do not place choices, option labels, or the answer in the question.

Identification rules:
- Supply one concise identifiable term, name, concept, classification, principle, process, figure, or title as correct_option.
- The answer must be a short phrase, not a sentence or explanation.
- Set wrong_option_1, wrong_option_2, wrong_option_3 to empty strings and is_true to null. These three fields must literally be "" â€” do not place any distractor words, related terms, or partial answers there, even though that pattern is normal for multiple-choice.
- Ask directly without embedding the answer or its full definition in the stem.
- Neither the complete answer nor its abbreviation may appear anywhere in the question, even as part of a longer phrase. Describe the concept by its function, purpose, defining trait, or relationships, never by restating its name. Use this content-neutral pattern: a question is invalid when it contains the exact correct_option text or its abbreviation. A valid question asks for the term through a supported function, purpose, defining trait, or relationship. If the supplied fact defining the concept restates the concept's own name, paraphrase around the name instead of quoting the fact.

True-false rules:
- Write only a declarative statement in question.
- For true-false cards, correct_option, wrong_option_1, wrong_option_2, and wrong_option_3 must ALL be empty strings. The only fields that carry the answer are is_true (0 or 1) and expalanation.
- Set is_true to integer 1 for true or integer 0 for false.
- Do not add True or False, labels, or evaluation instructions.
- False items must state a plausible misconception or incorrect relationship that the supplied facts resolve.

Scenario analysis assessment approach rules:
- Use any one of the three allowed types. The type controls the structure and fields: multiple-choice uses four options, identification uses one concise answer and empty wrong options, and true-false uses a declarative statement with is_true.
- Use a supported case or condition only when it improves the question. Do not invent a scenario or add decorative context merely to use the label.
- Set assessment_approach to "scenario analysis" and never set type to "scenario analysis".

DIRECT STEMS
For identification, use a natural direct form beginning with What, Which, Who, Where, When, Why, How, or What term. Multiple-choice may use the same direct form or put a concrete context or scenario before the question. Both types must ask a clear, answerable question ending in a question mark. Do not use wrappers such as According to, Based on, The material states, The following claim, Consider this statement, Evaluate this statement, Identify the concept associated with, or equivalents. The framings "Which of the following", "Which best describes", and "Which most accurately" are allowed when they produce a clear, answerable question. Avoid only vague or subjective wording that cannot be resolved from the supplied facts.

DIFFICULTY
Use integer 1 only for recall or straightforward understanding. Use integer 2 for interpretation, comparison, classification, application, or distinction. Use integer 3 for analysis, complex application, multi-step reasoning, competing explanations, or an unfamiliar but fully supported scenario.

EXPLANATION AND HINT
In expalanation, state the supported relationship or distinction that makes the answer correct. Never write generic filler such as "X is the correct answer here," "This statement is true," or "This statement is false." For a false statement, identify or correct the error when useful. In hint, give a useful clue about the relevant relationship, distinction, process, condition, or reasoning path without stating the answer. Neither field may mention provenance or presentation metadata.

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


def build_concept_plan_retry_prompt(
    identity: ModuleIdentity,
    facts: Sequence[GraphFact],
    prior_concept_names: Sequence[str],
    candidate: str | None,
    errors: Iterable[str],
) -> str:
    """Build a self-contained concept-plan retry, not the cards-oriented one.

    parse_concept_plan reports errors by a concept's *position* in the array
    ("concept 6 is grounded only in presentation/provenance content..."), but
    the model regenerates the whole array from scratch each attempt, so a
    position number from the last rejection describes nothing stable about
    the next one -- the same offending fact just resurfaces under a new
    concept name and a new position. What stays constant across attempts is
    *which fact_ids* are provenance/presentation-only and can never ground a
    concept alone. Naming those fact_ids directly, instead of only replaying
    the old position-indexed error text, is what lets the model avoid them on
    retry instead of reshuffling them into a different slot and failing
    again. This also drops the generic build_retry_prompt's card-only
    MANDATORY CORRECTIONS (identification wording, multiple-choice option
    counts, etc.), which don't apply to a concept plan and only crowd out the
    guidance that does.
    """

    error_list = [str(error) for error in errors]
    condensed, omitted = _condensed_errors(error_list)
    error_bullets = "\n".join(f"- {error}" for error in condensed)
    if omitted:
        error_bullets += f"\n- (+{omitted} similar errors omitted)"

    unusable_ids = sorted(
        {fact.fact_id for fact in facts if _contains_metadata_artifact(fact.statement)}
    )
    unusable_guidance = ""
    if unusable_ids:
        unusable_guidance = (
            "\n- These fact_ids describe presentation or provenance metadata "
            "(a title, header, section, or slide label), not assessable "
            "module content. No concept may rely on them, whether alone or "
            "combined only with other ids from this list: "
            + ", ".join(unusable_ids)
            + "\n"
        )

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
    if prior_concept_names:
        payload["previously_covered_concepts"] = list(prior_concept_names)

    rejected_block = ""
    if candidate is not None:
        rejected_json = candidate
        if len(rejected_json) > MAX_RETRY_CANDIDATE_CHARS:
            rejected_json = rejected_json[:MAX_RETRY_CANDIDATE_CHARS].rstrip()
        rejected_block = f"""

    REJECTED JSON TO CORRECT:
    {rejected_json}"""

    return f"""Regenerate one complete replacement concept plan.

    VALIDATION ERRORS
    {error_bullets}

    CORRECTIONS
    - Return exactly {CONCEPTS_PER_MODULE} concept objects, each with a
      unique name, fact_ids (copied exactly from graph_facts, never rewritten
      or invented), and exactly {CARDS_PER_CLUSTER} distinct assessment
      approaches from the allowed list.
    - A concept name must name an assessable topic. Never use a title,
      header, section, or slide label as a concept name.{unusable_guidance}
    - If two concepts would assess the same underlying learning point, keep
      one and replace the other with a concept grounded in different
      fact_ids.
    - Every fact_id must be copied exactly from graph_facts; do not alter or
      fabricate one.
    - Only return insufficient_content if fewer than {CONCEPTS_PER_MODULE}
      concepts are genuinely supportable once the unusable fact_ids above are
      set aside; state specifically how many are supportable and why.
    - Return JSON only, with no Markdown or surrounding text.

    INPUT JSON:
    {_json(payload)}{rejected_block}
    """


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
      evidence.
    - The five cards must be meaningfully distinct learning checks. They may
      assess the same concept from supported angles, but changing only the card
      type, opening phrase, or one word is not a different question. A negated
      form of a question is a different question.
    - Each multiple-choice wrong_option must be plausible, unambiguously
      incorrect, and relevant to the question and subject domain. It must use
      the same semantic category and answer shape as the correct option. A
      closely related distractor need not appear in the evidence.
    - Exactly one option may satisfy a multiple-choice question; never use
      another supported or arguably correct claim as a wrong option.
    - Never use another valid member of the requested category as a distractor.
      If several candidates satisfy a broad stem, narrow the question with a
      supported distinguishing property or choose clearly invalid alternatives.
    - Prefer positive questions. Do not use NOT or EXCEPT to make an invented
      claim the correct answer.
    - When the evidence supplies an equation or numerical relationship, an
      assigned application or scenario approach may test it using only supplied
      variables, values, units, and operations.

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
    - Never expose internal generation details, evidence labels, fact IDs, or
      source-reference wording. Ordinary domain uses of words such as module,
      document, file, slide, source, citation, and URL are allowed.
    - Every expalanation must state the relationship or distinction that makes
      the answer correct. Never write "X is the correct answer here," "This statement is true,"
      or "This statement is false."
    - Keep all eleven keys in every object and preserve the spelling expalanation.
    - Return one complete JSON object only, with no Markdown or surrounding text.

    Required top-level shape:
    {{"cards":[...exactly {CARDS_PER_CLUSTER} complete card objects...]}}

    Every card object must contain exactly:
    type, question, correct_option, wrong_option_1, wrong_option_2,
    wrong_option_3, is_true, expalanation, hint, difficulty,
    assessment_approach

    Scenario analysis may use any allowed structural type. Use a supported case
    only when it improves assessment; do not invent a story to force the label.

    INPUT JSON:
    """ + _json(payload)


def _cluster_payload(
    identity: ModuleIdentity,
    concept: ConceptPlan,
    concept_facts: Sequence[GraphFact],
    distractor_facts: Sequence[GraphFact],
    prior_signals: Sequence[str] = (),
) -> dict[str, object]:
    """Return only card-writing evidence, without model-visible control data."""

    del identity, distractor_facts, prior_signals
    return {
        "topic": concept.name,
        "assessment_approaches": list(concept.assessment_approaches),
        "evidence": [fact.statement for fact in concept_facts],
    }


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


def build_retry_prompt(
    original_prompt: str,
    candidate: str | None,
    errors: Iterable[str],
    grounded_vocabulary: Sequence[str] = (),
) -> str:
    # Retain the legacy fourth parameter for callers compiled against the old
    # API. It is no longer a hard allow-list for distractor wording.
    del grounded_vocabulary
    error_list = [str(error) for error in errors]
    condensed, omitted = _condensed_errors(error_list)
    error_bullets = "\n".join(f"- {error}" for error in condensed)
    if omitted:
        error_bullets += (
            f"\n- (+{omitted} more validation errors of a similar kind, "
            "omitted here for brevity -- fixing the pattern above resolves them too)"
        )

    rejected_json = candidate or "{}"
    if len(rejected_json) > MAX_RETRY_CANDIDATE_CHARS:
        rejected_json = (
            rejected_json[:MAX_RETRY_CANDIDATE_CHARS].rstrip()
            + "\n... (truncated; regenerate the full JSON from the ORIGINAL REQUEST, not from this partial excerpt)"
        )

    return f"""Correct the rejected JSON below.

VALIDATION ERRORS:
{error_bullets}

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
- If several errors say a fact or concept duplicates one already assigned
  elsewhere, do not rename or reorder it -- pick a different concept
  grounded in fact_ids that no other concept has used.
- If an error says a question "reveals the identification answer", the
  question text repeats correct_option's exact wording. Rewrite only the
  question so it describes the concept by its function, purpose, defining
  trait, or relationships -- do not restate the term, name, or its
  abbreviation anywhere in the question, even inside a longer phrase.
  Keep correct_option unchanged.
  Apply this content-neutral pattern: an invalid question repeats the exact
  correct_option text or its abbreviation. A valid question asks for the term
  through a supported function, purpose, defining trait, or relationship.

ORIGINAL REQUEST:
{original_prompt}

REJECTED JSON:
{rejected_json}
"""


_CARD_NUMBER_RE = re.compile(r"\bcards?\s+(\d+)(?:\s+and\s+(\d+))?\b", re.I)


def _flagged_card_positions(
    errors: Sequence[str], total: int = CARDS_PER_CLUSTER
) -> set[int] | None:
    """Return exactly which card numbers a validation-error list names.

    Every per-card error validate_cluster raises is built as
    f"card {position} ...", and every near-duplicate error pipeline.py
    raises is built as f"cards {i} and {j} ...", so in practice almost all
    errors name their card(s) explicitly. Returns None -- "don't know" --
    the moment any error does NOT name a card (a whole-cluster error such as
    a wrong card count, or a future error format this helper wasn't updated
    for): the caller must then treat every card as possibly wrong rather
    than promise to leave any of them untouched. This is checked against the
    full, pre-condensing error list, not the display-truncated one, so a
    card whose error got dropped by _condensed_errors' cap is never
    mistaken for one that passed.
    """
    flagged: set[int] = set()
    for error in errors:
        matches = list(_CARD_NUMBER_RE.finditer(str(error)))
        if not matches:
            return None
        for match in matches:
            for group in match.groups():
                if group is None:
                    continue
                position = int(group)
                if not 1 <= position <= total:
                    return None
                flagged.add(position)
    return flagged


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

    # Tell the model exactly which cards it is allowed to touch. Without
    # this, "regenerate one complete replacement five-card JSON object"
    # invites the model to silently rewrite cards no error complained
    # about -- which is how a card with four perfectly fine multiple-choice
    # options on one attempt comes back with all four options collapsed to
    # the same sentence on the next, purely as collateral damage from fixing
    # an unrelated card.
    preserve_bullet = ""
    if candidate is not None:
        flagged = _flagged_card_positions(error_list)
        if flagged and len(flagged) < CARDS_PER_CLUSTER:
            flagged_list = sorted(flagged)
            preserved = sorted(set(range(1, CARDS_PER_CLUSTER + 1)) - flagged)
            flagged_words = ", ".join(str(n) for n in flagged_list)
            preserved_words = ", ".join(str(n) for n in preserved)
            flagged_noun = "card" + ("s" if len(flagged_list) > 1 else "")
            flagged_verb = "are" if len(flagged_list) > 1 else "is"
            preserved_noun = "card" + ("s" if len(preserved) > 1 else "")
            preserve_bullet = (
                f"\n    - Only {flagged_noun} {flagged_words} {flagged_verb} "
                "named by a validation error below; every other card in "
                "REJECTED JSON already passed every check. Copy "
                f"{preserved_noun} {preserved_words} into your answer "
                "completely unchanged -- identical question, options, "
                "is_true, expalanation, hint, difficulty, and "
                "assessment_approach, character for character. Rewrite only "
                "the flagged card(s) above."
            )

    payload = _cluster_payload(
        identity,
        concept,
        concept_facts,
        distractor_facts,
        prior_signals,
    )

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
      reasoning. Approaches may repeat; no planned value is required to appear.{preserve_bullet}
    - Use only multiple-choice, identification, and true-false types, including
      at least one of each. Scenario analysis is an approach, never a type.
    - Identification and true-false option fields must be "". Multiple-choice
      must contain one correct option and three distinct incorrect options.
      Distinct means genuinely different claims, not the same claim reworded
      -- if the facts are too narrow to support three different wrong
      options, use a closely related plausible term from outside the facts
      rather than paraphrasing correct_option three times.
    - Questions, correct answers, true-false decisions, explanations, and hints
      must use evidence. Multiple-choice distractors may use a familiar
      closely related term absent from the facts, but must remain relevant,
      plausible, in the same semantic category, and unambiguously incorrect.
      Exactly one multiple-choice option may satisfy its question.
    - Do not invent a scenario merely to use a scenario-analysis label. You may
      change an approach label when another planned value describes the actual
      reasoning more accurately.
    - Make all five cards meaningfully distinct learning checks. They may assess
      the same concept from supported angles, but do not repeat a question by
      changing only its type, opening phrase, or one word.
    - Remove provenance wording and keep the answer out of identification stems.
      Never mention fact IDs, evidence labels, prior subjects, or other internal
      generation details. If an error says a question "reveals the
      identification answer", the question text repeats correct_option's
      exact wording (or its abbreviation) somewhere inside it. Rewrite only
      the question so it asks for the term by its function, purpose,
      defining trait, or relationship to another supplied fact -- never by
      restating the term itself. Keep correct_option unchanged. If the
      concept's facts are too narrow to describe the term any other way,
      write that card about a different supported angle on the concept (a
      citation, a component, a contrast with a distractor fact) instead of
      re-describing the same definition again.
    - Each expalanation must explain the supported relationship or distinction,
      not merely say that an answer is correct or a statement is true or false.
    - Include all eleven required fields and preserve expalanation spelling.
    - Return JSON only with the shape {{"cards":[...]}}.

    INPUT JSON:
    {_json(payload)}{rejected_block}
    """


def _duplicate_card_edit_rules(card: FlashcardDraft) -> str:
    if card.type == "true-false":
        return """TRUE-FALSE DECLARATIVE REQUIREMENT
    - Paraphrase only question. It must be a declarative statement, not a
      question. Do not begin with Does, Do, Is, Are, Can, Could, Should, Would,
      Will, What, Which, Who, Where, When, Why, or How. End it with a period,
      never a question mark.
    - Copy type, is_true, all four option fields, expalanation, hint, difficulty,
      and assessment_approach exactly."""
    if card.type == "identification":
        return """IDENTIFICATION EDIT LIMITS
    - Paraphrase only question and correct_option.
    - Copy type, all three empty wrong-option fields, is_true, expalanation,
      hint, difficulty, and assessment_approach exactly."""
    return """MULTIPLE-CHOICE EDIT LIMITS
    - Paraphrase only question and the four answer-option fields.
    - Copy type, is_true, expalanation, hint, difficulty, and
      assessment_approach exactly."""


def build_duplicate_card_repair_prompt(
    identity: ModuleIdentity,
    concept: ConceptPlan,
    card: FlashcardDraft,
    *,
    cluster_number: int,
    card_number: int,
    cluster_id: str,
    reasons: Sequence[str],
    conflicting_questions: Sequence[dict[str, object]],
) -> str:
    """Build a focused request for one flagged card at one JSON location."""

    payload = {
        "course_code": identity.course_code,
        "module_number": identity.module_number,
        "concept": concept.name,
        "cluster_uuid": cluster_id,
        "json_location": {"cluster": cluster_number, "card": card_number},
        "duplicate_findings": list(reasons),
        "card_to_repair": asdict(card),
        "conflicting_questions": list(conflicting_questions),
    }
    return (
        """Paraphrase exactly one flagged flashcard.

    REPAIR RULES
    - Rewrite only card_to_repair. Python will return it to json_location; do not
      generate, copy, or discuss any other card.
    - Preserve the card's learning point, factual meaning, correct-answer
      meaning, and incorrectness of wrong answers.
    - Natural paraphrases may use words that do not appear verbatim in any
      corpus or fact text. Do not introduce a new claim or change which answer
      is correct.
    - Make the repaired question genuinely distinct from conflicting_questions.
      A question that differs from a conflicting question by only one word (or
      only the opening question word) is still a duplicate. Negating the
      question is allowed and counts as different, but the options must change
      to match. Otherwise restructure the stem (scenario, fill-in-the-blank,
      condition first, or the same relationship asked from the other
      direction).
    - Do not change the assessment approach. Copy every field not explicitly
      permitted by the type-specific edit limits exactly.
    - Keep the eleven required fields and preserve the key spelling expalanation.
    - Return one card only, with no Markdown or surrounding text.

    """
        + _duplicate_card_edit_rules(card)
        + """

    Required shape: {"cards":[{...exactly one complete card...}]}

    INPUT JSON:
    """
        + _json(payload)
    )


_DUPLICATE_PARAPHRASE_TECHNIQUES: dict[str, tuple[str, ...]] = {
    "multiple-choice": (
        "INVERT (swap question and answer): make the concept that is now "
        "correct_option the subject of the stem and ask for the category, "
        "system, or relationship it belongs to (or the reverse). Rebuild "
        "correct_option and all three wrong options for the new question; all "
        "four must stay in one semantic category and answer shape.",
        "FLIP POLARITY WITH A NEW ANSWER SET: turn a NOT / EXCEPT / does-not-"
        "belong stem into a positive 'Which ... is part of ...' stem whose "
        "correct_option is a supported member and whose wrong options are "
        "outside items (or the reverse). Do not merely add or delete the "
        "word NOT; the options must change too.",
        "SCENARIO: wrap the same supported relationship in a short concrete "
        "situation (a person, task, or observation) and ask what applies. The "
        "stem must not reuse the conflicting stem's opening words.",
        "RESTRUCTURE: use fill-in-the-blank, condition-first ('When ..., "
        "which ...'), or definition-first phrasing so the sentence skeleton "
        "differs from every conflicting question.",
        "SHIFT ANGLE: test the same supported fact through a different element "
        "of it (a component, its function, or a contrast with one wrong "
        "option), keeping the learning point but changing what is asked.",
    ),
    "true-false": (
        "NEGATE AND FLIP: state the opposite claim as a declarative statement "
        "and flip is_true to match. Also rewrite the subject/predicate "
        "wording; a bare 'not' insertion is not enough.",
        "REVERSE THE RELATION: swap subject and predicate so the statement "
        "reads from the other direction (e.g. 'X are part of Y' becomes 'Y "
        "includes X'), keeping is_true.",
        "SUBSTITUTE A SUPPORTED DETAIL: state the claim about a different "
        "supported member, function, or condition of the same concept, "
        "setting is_true to match the supplied facts.",
        "SCENARIO OR CONDITION: express the claim inside a brief situation or "
        "an if/when clause, still as a declarative statement.",
    ),
    "identification": (
        "INVERT (swap question and answer): put the current correct_option in "
        "the stem and ask for the supported function, category, or "
        "relationship as the new correct_option. The new answer must not "
        "appear in the stem.",
        "DESCRIBE BY FUNCTION: replace the defining phrase with a supported "
        "purpose, trait, or relationship and keep correct_option.",
        "SHIFT ANGLE: ask for a different supported element of the same "
        "concept (a component, citation, or contrast) instead of its "
        "definition.",
        "RESTRUCTURE: condition-first or scenario-based stem with a different "
        "sentence skeleton from every conflicting question.",
    ),
}


def _duplicate_retry_technique_block(card: FlashcardDraft, attempt: int) -> str:
    techniques = _DUPLICATE_PARAPHRASE_TECHNIQUES.get(
        card.type, _DUPLICATE_PARAPHRASE_TECHNIQUES["multiple-choice"]
    )
    index = max(0, attempt - 2) % len(techniques)
    primary = techniques[index]
    others = "\n".join(f"    - {t}" for i, t in enumerate(techniques) if i != index)
    return f"""PARAPHRASE TECHNIQUE FOR THIS ATTEMPT (attempt {attempt})
    - Apply this technique first: {primary}
    - You may combine it with any technique below if that is what it takes to
      make the question clearly different:
{others}
    - Adding, removing, or swapping a single word (for example inserting
      "key") or changing only the opening question word is NOT a paraphrase;
      the duplicate check compares the whole sentence word by word. Negating
      the question does count as different, provided the options change to
      match.
    - Do not reuse the wording of any rejected question (listed below), and do
      not return to an earlier attempt's stem."""


def _duplicate_retry_edit_rules(card: FlashcardDraft) -> str:
    """Retry edit limits: the technique may require changing the answer."""
    common = """RETRY EDIT LIMITS
    - Always copy type, difficulty, and assessment_approach exactly.
    - The learning point stays the same, and every field must remain supported
      only by the supplied facts."""
    if card.type == "true-false":
        return common + """
    - question must be a declarative statement, not a question. Do not begin
      with Does, Do, Is, Are, Can, Could, Should, Would, Will, What, Which,
      Who, Where, When, Why, or How. End it with a period.
    - Set is_true to whatever the new statement's truth is; keep all four
      option fields empty. Rewrite expalanation and hint to match the new
      statement."""
    if card.type == "identification":
        return common + """
    - You may change question and correct_option. Keep the three wrong-option
      fields exactly "" and is_true null. If correct_option changes, rewrite
      expalanation and hint to match."""
    return common + """
    - You may change question, correct_option, and all three wrong options.
      Keep is_true null and the four options textually different.
    - If correct_option changes, rewrite expalanation and hint so they explain
      the NEW answer; do not copy text that describes the old one."""


def _rejected_question_texts(
    rejected_json: str | None, errors: Sequence[str]
) -> list[str]:
    seen: list[str] = []
    if rejected_json:
        try:
            data = json.loads(rejected_json)
            for item in data.get("cards", []):
                question = str(item.get("question", "")).strip()
                if question:
                    seen.append(question)
        except (ValueError, AttributeError, TypeError):
            pass
    for error in errors:
        seen.extend(re.findall(r"'([^']{8,})'", str(error)))
    return list(dict.fromkeys(seen))


def build_duplicate_card_retry_prompt(
    original_prompt: str,
    rejected_json: str | None,
    errors: Sequence[str],
    card: FlashcardDraft,
    *,
    attempt: int = 2,
    rejected_questions: Sequence[str] = (),
) -> str:
    """Retry one duplicate card with a different paraphrase technique.

    ``attempt`` is the attempt number about to run (2 or 3) so each retry uses
    a different technique. ``rejected_questions`` may carry every stem rejected
    on earlier attempts, so the model cannot cycle back to one.
    """

    rejected = list(
        dict.fromkeys(
            [*rejected_questions, *_rejected_question_texts(rejected_json, errors)]
        )
    )
    rejected_block = "\n".join(f"    - {q}" for q in rejected) or "    - (none)"
    return (
        """Retry the one-card paraphrase. The last attempt was rejected because it
    is still a near-duplicate of another question.

    STRUCTURE CHANGE REQUIRED
    - You must change the structure of the question, not just its wording.
      The duplicate check compares the whole sentence word by word, so a
      question that differs from a conflicting question by one added, removed,
      or swapped word (for example inserting "key" or "primary") is still a
      duplicate. Changing only the opening word (Which/What) is also still a
      duplicate.
    - Build a new sentence skeleton: reorder the clauses, change the question
      form (fill-in-the-blank, condition-first, scenario, or a statement to
      complete), or ask the same relationship from the other direction. Several
      words must differ from every conflicting question, not one.
    - Negating the question is allowed and counts as different, but the answer
      options must change to match.
    - Before answering, compare your new question with each conflicting
      question and each rejected question below. If they still share the same
      sentence skeleton, restructure again.

    """
        + _duplicate_retry_technique_block(card, attempt)
        + f"""

    REJECTED QUESTIONS (never reuse or lightly reword these)
{rejected_block}

    """
        + _duplicate_retry_edit_rules(card)
        + """

    """
        + build_retry_prompt(original_prompt, rejected_json, errors)
    )


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
    return f"""Review these generated clusters against only their supplied facts. Questions, correct answers, true-false decisions, explanations, and hints must be supported by those facts. Flag a cluster if any card contains an unsupported claim, answer leakage, an invalid distractor, or a misleading explanation. Also flag a cluster when two cards ask the same question with at most one word different; cards that test the same fact in differently worded questions, in a different type, or in negated form are not duplicates.

    A wrong option is not invalid merely because its wording does not appear in the supplied facts. It may use a familiar related term, but it must be plausible, unambiguously incorrect, in the same semantic category and answer shape as the correct option, and relevant to the question and module domain. Flag a distractor only when it is correct or arguably correct, duplicates another option, is nonsensical, mismatches the answer category, or is unrelated to the question or module domain. Do not rewrite cards.

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
    return """Compare all question stems for near duplication. Flag only a question that is the same sentence as another question with at most one filler, modifier, or synonym word added, removed, or swapped. Do not flag questions that merely assess the same fact or concept in differently worded questions, questions of different types, or a question and its negated form (for example \"is a component\" versus \"is NOT a component\"). A swapped key term that changes what is asked (for example binary versus octal) makes a different question. Report each duplicate pair once: put the later cluster/card in the issue's cluster field and at the start of its reason, and identify the earlier card after the word cluster. Every reason must use exactly the form "card <number> duplicates cluster <uuid> card <number>". Do not judge factual correctness in this pass and do not rewrite questions.

    Return only this JSON shape. Use an empty issues list when no duplicate exists:
    {"issues":[{"cluster":"valid UUID copied from input","reasons":["card 2 duplicates cluster <uuid> card 4"]}]}

    INPUT JSON:
    """ + _json(
        payload
    )
