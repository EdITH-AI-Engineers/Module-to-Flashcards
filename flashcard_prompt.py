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

Never mention internal generation details or cite the input container in a question, answer, explanation, or hint. Do not write phrases such as "fact f13," "concept facts," "provided vocabulary," "already covered subjects," "according to the module," or "as stated in the document." Ordinary subject-matter uses of words such as module, document, file, slide, source, citation, and URL are allowed. Refer to the topic itself, never to where the information appeared.

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
- Set wrong_option_1, wrong_option_2, wrong_option_3 to empty strings and is_true to null. These three fields must literally be "" - do not place any distractor words, related terms, or partial answers there, even though that pattern is normal for multiple-choice.
- Ask directly without embedding the answer or its full definition in the stem.
- Neither the complete answer nor its abbreviation may appear anywhere in the question, even as part of a longer phrase. Describe the concept by its function, purpose, defining trait, or relationships, never by restating its name. Use this content-neutral pattern: a question is invalid when it contains the exact correct_option text or its abbreviation. A valid question asks for the term through a supported function, purpose, defining trait, or relationship. If the supplied fact defining the concept restates the concept's own name, paraphrase around the name instead of quoting the fact.

True-false rules:
- Write only a declarative statement in question.
- Build a true-false decision only from a supplied fact that explicitly asserts a
  claim. An unresolved question, unanswered choice, or supplied fact ending in
  a question mark cannot establish truth or falsity. Never infer is_true from
  the wording of such material; use a different assertive fact or use a
  multiple-choice or identification card instead.
- For true-false cards, correct_option, wrong_option_1, wrong_option_2, and wrong_option_3 must ALL be empty strings. The only fields that carry the answer are is_true (0 or 1) and expalanation.
- Set is_true to integer 1 for true or integer 0 for false.
- Do not add True or False, labels, or evaluation instructions.
- False items must state a plausible misconception or incorrect relationship that the supplied facts resolve.

Scenario analysis assessment approach rules:
- Use any one of the three allowed types. The type controls the structure and fields: multiple-choice uses four options, identification uses one concise answer and empty wrong options, and true-false uses a declarative statement with is_true.
- Use a supported case or condition only when it improves the question. Do not invent a scenario or add decorative context merely to use the label.
- Set assessment_approach to "scenario analysis" and never set type to "scenario analysis".

DIRECT STEMS
For identification, use a natural direct form beginning with What, Which, Who, Where, When, Why, How, or What term. Multiple-choice may use the same direct form or put a concrete context or scenario before the question. Both types must ask a clear, answerable question ending in a question mark. Do not cite the input container with wrappers such as According to the facts, Based on the material, The material states, The following claim, Consider this statement, Evaluate this statement, Identify the concept associated with, or equivalents. A named theory, law, standard, organization, or subject-matter rule may provide necessary context. The framings "Which of the following", "Which best describes", and "Which most accurately" are allowed when they produce a clear, answerable question. Avoid only vague or subjective wording that cannot be resolved from the supplied facts.

DIFFICULTY
Use integer 1 only for recall or straightforward understanding. Use integer 2 for interpretation, comparison, classification, application, or distinction. Use integer 3 for analysis, complex application, multi-step reasoning, competing explanations, or an unfamiliar but fully supported scenario.

EXPLANATION AND HINT
In expalanation, state the supported relationship or distinction that makes the answer correct. Never write generic filler such as "X is the correct answer here," "This statement is true," or "This statement is false." For a false statement, identify or correct the error when useful. In hint, give a useful clue about the relevant relationship, distinction, process, condition, or reasoning path without stating the answer. Neither field may mention provenance or presentation metadata.

EQUATIONS
Use plain text only: + - * / ^ = < > <= >= sqrt(...) ( ). Do not use LaTeX, MathJax, HTML math, images, superscript glyphs, or unsupported notation.

Before returning JSON, silently check every item against the supplied facts and all requested constraints. Never expose that check.
"""

# SYSTEM_PROMPT remains available to older callers. Generation uses stable,
# task-specific instructions so reviews and planning do not prefill card rules.
PLAN_SYSTEM = """You plan college-level flashcard concepts from graph facts.
Use only the supplied facts as factual authority. Treat each module separately.
Choose distinct assessable learning points, not titles, slide labels, source
metadata, or the same relationship renamed. Never invent facts to fill a quota;
report insufficient_content when fewer than twenty distinct concepts are
supported. A source question without an asserted answer is not a fact.
Return JSON only, using exactly the requested keys. Copy fact_ids exactly;
Python resolves them to statements. Select five distinct assessment approaches
per concept from the allowed list. An approach describes reasoning, not a card
type. Do not repeat a prior module's underlying learning point.
"""

CLUSTER_SYSTEM = """You write college-level assessment cards. Use only supplied
facts for questions, correct answers, true-false decisions, explanations, and
hints. Never infer truth from an unresolved source question or outside knowledge.
Multiple-choice distractors may be familiar related domain terms absent from
the facts; absence alone does not make one incorrect. Never assess the same
learning check twice: changing only card type, opening phrase, or one word is
not enough; a supported negated question is different. Keep each question
clear and authentic, with difficulty from reasoning rather than obscure wording.

Return one JSON object only. Parsed cards have eleven canonical fields: type,
question, correct_option, wrong_option_1, wrong_option_2, wrong_option_3,
is_true, expalanation, hint, difficulty, assessment_approach. Preserve the
spelling expalanation. Follow the per-type schema: omit fixed empty options
and null is_true where the schema excludes them; Python fills these before
validation. Allowed types are multiple-choice, identification, and
true-false; each five-card cluster has at least one of each and should vary its
type mix across clusters. Type is structural; scenario analysis and the other
assessment_approach values are never types.

Approach meanings: recall retrieves an explicit fact; comparison contrasts
supported things; classification places an item in a supported category;
application uses a rule or relationship; scenario analysis interprets a
supported case only when helpful; cause/effect connects a supported cause and
result; misconception detection corrects an unsupported belief; conditions
tests requirements; consequences tests an outcome; reversed reasoning infers
a cause or rule from a supported result. The assessment_approach label must
describe the actual reasoning. Approaches may repeat; no planned approach is
required to appear. A definition question is recall or classification, not
scenario analysis. Never invent a story merely to use a label.

Multiple-choice: supply exactly one concise correct option and three plausible,
distinct, incorrect options of the same semantic category and answer shape.
Exactly one may satisfy the stem. Never use another supported or arguably
correct claim or valid category member as a distractor; narrow a broad stem
with a supported distinguishing property. All four options must be textually
different, not paraphrases of one another. Keep wrong answers relevant and
unambiguously incorrect. Prefer positive stems; never make an invented claim
correct through NOT or EXCEPT. is_true is null after parsing. Do not put choices or the
answer in the stem.

Identification: correct_option is one concise term, name, concept, class,
principle, process, figure, or title, not a sentence. All three wrong_option
fields are empty strings and is_true null after parsing. Ask directly without
the answer or its abbreviation anywhere in the question, even inside a longer
phrase; describe its supported function, purpose, defining trait, or relation
instead of repeating its name or full definition.

True-false: question is a declarative statement, not an instruction or
question. Use only a supplied fact that explicitly asserts a claim, never an
unresolved question. All four option fields are empty strings after parsing; is_true is
integer 1 for true or 0 for false. False claims must be plausible errors the
facts resolve. Do not add True/False labels to the stem.

Identification starts with What, Which, Who, Where, When, Why, or How.
Multiple-choice may start with a direct question or a concrete supported
context. Both end in a question mark; true-false statements do not. Avoid
vague subjective wording and provenance wrappers such as 'according to the
facts' or 'based on the material'. Never expose fact IDs, input labels, prior
subjects, or internal generation details. Ordinary subject-matter uses of
module, document, file, slide, source, citation, or URL are allowed.

When facts supply equations, numerical relationships, or defined quantities,
use supported problem solving for an appropriate approach. Use only supplied
variables, units, relationships, and operations; state every needed value and
derive the answer deterministically. Do not force numerical work without
enough information. Equations use plain-text + - * / ^ = < > <= >= sqrt(...) ( ),
not LaTeX, HTML math, superscript glyphs, or images.

Difficulty 1 is recall or straightforward understanding; 2 is interpretation,
comparison, classification, application, or distinction; 3 is analysis,
complex or multi-step application, competing explanations, or a fully supported
unfamiliar scenario. expalanation states the supported relationship making the
answer correct, not generic 'X is correct' filler; for a false claim identify
the error when useful. hint gives a useful clue without the answer. Neither
field mentions provenance or presentation metadata.
"""

REVIEW_SYSTEM = """You review college-level flashcards using only supplied
facts and requested review criteria. Report defects, not rewrites. A related
term absent from supplied facts is not automatically an invalid distractor;
judge whether it is plausible, same-category, and unambiguously incorrect.
Do not confuse a shared topic with a near-duplicate question. Use only the
short cluster indices shown in the request. Return JSON only with an issues
array; use an empty array when there are no defects.
"""

REPAIR_SYSTEM = """You repair college-level flashcards. Use only supplied
facts for questions, correct answers, true-false truth, explanations, and
hints; closely related distractor terms may be absent from the facts but must
be plausible and unambiguously incorrect. Preserve the supported learning
point and all fields locked by the request. Make the corrected question
distinct, not a one-word or opening-phrase change. Never mention input labels,
fact IDs, or provenance. Return JSON only, with the exact requested shape and
key spelling expalanation.

Allowed types: multiple-choice, identification, true-false. Multiple-choice
has exactly one correct answer, three distinct, same-category incorrect
answers, and null is_true after parsing. Identification has one concise answer absent from
its direct question, three empty wrong options, and null is_true after parsing. True-false
has a declarative statement based on an asserted fact, all option fields
empty after parsing, and integer is_true of 0 or 1. Omit fixed fields the
schema excludes. Scenario analysis is an approach, not
a type, and must use a supported case only when helpful. The approach must
match the actual reasoning. expalanation states the supported relationship,
not generic filler; hint clues the reasoning without giving the answer.
"""


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _plan_fact_lines(facts: Sequence[GraphFact]) -> list[str]:
    """Keep exact fact IDs while avoiding repeated JSON object keys and provenance."""
    return [
        f"{fact.fact_id}: "
        + (f"[{fact.topic}] " if fact.topic else "")
        + fact.statement
        for fact in facts
    ]


def build_concept_plan_prompt(
    identity: ModuleIdentity,
    facts: Sequence[GraphFact],
    prior_concept_names: Sequence[str] = (),
) -> str:
    payload: dict[str, object] = {
        "course_code": identity.course_code,
        "module_number": identity.module_number,
        "graph_facts": _plan_fact_lines(facts),
    }
    overlap_guidance = ""
    context_guidance = ""
    if any(fact.topic for fact in facts):
        context_guidance = (
            " Bracketed topics provide grouping context, not additional facts."
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

    Each graph_facts line starts with an exact fact_id followed by a colon. For each concept, copy one or more of those fact_ids exactly. Do not copy or rewrite fact statements; Python will resolve the selected IDs to their exact statements.{context_guidance} Choose exactly {CARDS_PER_CLUSTER} distinct approaches from: recall, comparison, classification, application, scenario analysis, cause/effect, misconception detection, conditions, consequences, reversed reasoning.

    Return one JSON object with a concepts array of exactly {CONCEPTS_PER_MODULE} uniquely named objects. Each object has name (string), fact_ids (non-empty string array), and assessment_approaches (array of exactly {CARDS_PER_CLUSTER} distinct allowed approaches).
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
        "graph_facts": _plan_fact_lines(facts),
    }
    if prior_concept_names:
        payload["previously_covered_concepts"] = list(prior_concept_names)

    rejected_block = ""
    if candidate is not None:
        rejected_json, omitted = _trim_rejected_json(candidate)
        rejected_block = (
            "\n\n    REJECTED JSON TO CORRECT:\n    "
            + (rejected_json or "(invalid JSON omitted)")
            + (f"\n    ({omitted} whole entries omitted)" if omitted else "")
        )

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
PLANNED ASSESSMENT APPROACHES (not card positions):
{approach_list}
Choose the label matching each card's actual reasoning; a listed approach does
not have to appear, and approaches may repeat.
CRITICAL REMINDER
- Use only supported facts for answers; exactly one multiple-choice option may be correct.
- Make five meaningfully distinct learning checks, not one-word rewrites.
- Include multiple-choice, identification, and true-false types; scenario analysis is an approach.
- Identification parses to three empty wrong options and has no answer text in its question.
- True-false is declarative, parses to four empty options, and uses integer is_true.
- Keep expalanation grounded; no provenance wording or internal labels.
INPUT JSON:
{_json(payload)}
"""


def _cluster_payload(
    identity: ModuleIdentity,
    concept: ConceptPlan,
    concept_facts: Sequence[GraphFact],
    distractor_facts: Sequence[GraphFact],
    prior_signals: Sequence[str] = (),
) -> dict[str, object]:
    """Return only card-writing facts and their topic."""

    del identity, distractor_facts, prior_signals
    return {
        "topic": concept.name,
        "assessment_approaches": list(concept.assessment_approaches),
        "facts": [fact.statement for fact in concept_facts],
    }


def build_cluster_batch_prompt(
    numbered_concepts: Sequence[tuple[int, ConceptPlan, Sequence[GraphFact]]],
) -> str:
    """Request several independent five-card clusters in one model response."""
    payload = {
        "concepts": [
            {
                "number": number,
                "topic": concept.name,
                "assessment_approaches": list(concept.assessment_approaches),
                "facts": [fact.statement for fact in concept_facts],
            }
            for number, concept, concept_facts in numbered_concepts
        ]
    }
    return f"""Generate one independent five-card cluster for each numbered
concept below. Return one JSON object with a clusters array of exactly
{len(numbered_concepts)} entries. Each entry has its input number and exactly
five cards. Do not mix facts or assessment approaches between concepts.

CRITICAL REMINDER
- Each cluster has at least one multiple-choice, identification, and true-false card.
- Each card must be grounded in its own concept's facts and use a matching approach.
- Keep five meaningfully distinct questions per cluster; no one-word rewrites.
- Follow the per-type schema and preserve expalanation spelling.

INPUT JSON:
{_json(payload)}"""


MAX_RETRY_ERROR_COUNT = 12
MAX_RETRY_ERROR_CHARS = 220
MAX_RETRY_CANDIDATE_CHARS = 6000


def _trim_rejected_json(candidate: str | None) -> tuple[str | None, int]:
    """Trim whole array entries; never put a sliced JSON fragment in a prompt."""
    if candidate is None:
        return None, 0
    try:
        value = json.loads(candidate)
    except (ValueError, TypeError):
        return None, 1
    if not isinstance(value, dict):
        return None, 1
    compact = _json(value)
    if len(compact) <= MAX_RETRY_CANDIDATE_CHARS:
        return compact, 0
    for key in ("cards", "concepts", "issues"):
        entries = value.get(key)
        if not isinstance(entries, list):
            continue
        omitted = 0
        while entries and len(compact) > MAX_RETRY_CANDIDATE_CHARS:
            entries.pop()
            omitted += 1
            compact = _json(value)
        if len(compact) <= MAX_RETRY_CANDIDATE_CHARS:
            return compact, omitted
    return None, 1


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

    rejected_json, omitted_entries = _trim_rejected_json(candidate)
    rejected_block = rejected_json or "(rejected JSON omitted because it was unavailable or invalid)"
    if omitted_entries:
        rejected_block += f"\n({omitted_entries} whole entries omitted to fit context)"

    return f"""Correct the rejected JSON below.

VALIDATION ERRORS:
{error_bullets}

MANDATORY CORRECTIONS:
- When a validation error names a card number, correct that exact card.
- Set each assessment_approach to one of the planned values and make the label
  match the question's actual reasoning. Approaches may repeat.
- For identification, wrong_option_1, wrong_option_2, and wrong_option_3 must be all exactly "".
- For multiple-choice, the correct option and three wrong options must be four different strings.
- Rewrite any multiple-choice or identification question that cites the input
  through a wrapper such as "According to the supplied facts" or "Based on the
  material". Preserve relevant named subject-matter authorities and rules.
- Identification must begin directly with What, Which, Who, Where, When, Why, or How. Multiple-choice may instead begin with a concrete context or scenario and then ask the question.
- Example:
  Invalid: "According to the supplied facts, which measure assesses effectiveness?"
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
{rejected_block}
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

    answer_leak_guidance = ""
    true_false_guidance = ""
    if any(
        "true-false question must be a declarative statement only" in error
        for error in error_list
    ):
        true_false_guidance = """

    TRUE-FALSE DECLARATIVE CORRECTION
    - A true-false question must be a declarative statement ending with a
      period, never an interrogative beginning with Does, Do, Is, Are, Can,
      Could, Should, Would, Will, What, Which, Who, Where, When, Why, or How.
    - Do not infer is_true from an unresolved question, an unanswered choice,
      or the mere fact that an issue was raised. If the supplied facts do not
      explicitly settle the claim, change this card to multiple-choice or
      identification when another true-false card remains in the cluster.
      Otherwise, replace it with a declarative true-false statement grounded
      in a different explicit assertion from the facts.
    """
    if candidate is not None:
        try:
            candidate_value = json.loads(candidate)
            candidate_cards = candidate_value.get("cards", [])
        except (json.JSONDecodeError, AttributeError):
            candidate_cards = []
        forbidden_answers: list[str] = []
        for error in error_list:
            match = re.match(
                r"^card (\d+) question reveals the identification answer$",
                error,
            )
            if match is None:
                continue
            card_number = int(match.group(1))
            if not 1 <= card_number <= len(candidate_cards):
                continue
            card = candidate_cards[card_number - 1]
            if not isinstance(card, dict):
                continue
            answer = card.get("correct_option")
            if isinstance(answer, str) and answer.strip():
                forbidden_answers.append(
                    f'    - Card {card_number}: the question must not contain '
                    f'the answer text {json.dumps(answer.strip(), ensure_ascii=False)} '
                    "or its abbreviation. Keep that correct_option and describe "
                    "its supported function or defining relationship instead."
                )
        if forbidden_answers:
            answer_leak_guidance = (
                "\n\n    ANSWER TEXT FORBIDDEN IN IDENTIFICATION QUESTIONS\n"
                + "\n".join(forbidden_answers)
            )

    flagged = _flagged_card_positions(error_list)
    candidate_cards: list[object] = []
    if candidate is not None:
        try:
            candidate_value = json.loads(candidate)
            if isinstance(candidate_value, dict) and isinstance(candidate_value.get("cards"), list):
                candidate_cards = candidate_value["cards"]
        except (ValueError, TypeError):
            pass
    if (
        flagged
        and len(flagged) < CARDS_PER_CLUSTER
        and len(candidate_cards) == CARDS_PER_CLUSTER
    ):
        numbered = {
            "cards": [
                {"card_number": number, "card": candidate_cards[number - 1]}
                for number in sorted(flagged)
            ]
        }
        rejected_json, omitted = _trim_rejected_json(_json(numbered))
        payload = _cluster_payload(
            identity, concept, concept_facts, distractor_facts, prior_signals
        )
        return f"""Repair only the numbered cards in REJECTED JSON. Python keeps
all other cards byte-identical. Return exactly {len(flagged)} entries as
{{"cards":[{{"card_number":2,"card":{{...}}}}]}}; use only these card numbers:
{', '.join(str(number) for number in sorted(flagged))}.

VALIDATION ERRORS
{error_bullets}{answer_leak_guidance}{true_false_guidance}

Keep each repaired card grounded in the facts, preserve its supported learning
point, use an assessment approach matching the actual reasoning, and do not
repeat an unchanged question. Follow the per-type JSON schema and keep the
spelling expalanation.

REJECTED JSON TO CORRECT:
{rejected_json or '(invalid JSON omitted)'}
{f'({omitted} whole entries omitted)' if omitted else ''}

INPUT JSON:
{_json(payload)}
"""

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
        rejected_json, omitted = _trim_rejected_json(candidate)
        rejected_block = (
            "\n\n    REJECTED JSON TO CORRECT:\n    "
            + (rejected_json or "(invalid JSON omitted)")
            + (f"\n    ({omitted} whole entries omitted)" if omitted else "")
        )

    return f"""Regenerate one complete replacement five-card JSON object for the supplied concept.

    VALIDATION ERRORS
    {error_bullets}

    CORRECTIONS
    - Return exactly {CARDS_PER_CLUSTER} complete cards. Set each
      assessment_approach to a planned value that matches the question's actual
      reasoning. Approaches may repeat; no planned value is required to appear.{preserve_bullet}{answer_leak_guidance}{true_false_guidance}
    - Use only multiple-choice, identification, and true-false types, including
      at least one of each. Scenario analysis is an approach, never a type.
    - Identification and true-false option fields must be "". Multiple-choice
      must contain one correct option and three distinct incorrect options.
      Distinct means genuinely different claims, not the same claim reworded
      -- if the facts are too narrow to support three different wrong
      options, use a closely related plausible term from outside the facts
      rather than paraphrasing correct_option three times.
    - Questions, correct answers, true-false decisions, explanations, and hints
      must use the supplied facts. Multiple-choice distractors may use a familiar
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
      Never mention fact IDs, input labels, prior subjects, or other internal
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
    - Follow the per-type JSON schema; Python fills omitted structural constants
      before validation. Preserve expalanation spelling.
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
        "facts": list(concept.facts),
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
    - Follow the per-type JSON schema; Python fills omitted structural constants.
      Preserve the key spelling expalanation.
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
    context: dict[str, object] = {"card_to_repair": asdict(card)}
    marker = "INPUT JSON:\n"
    if marker in original_prompt:
        try:
            source, _end = json.JSONDecoder().raw_decode(
                original_prompt.split(marker, 1)[1].lstrip()
            )
            if isinstance(source, dict):
                context = {
                    key: source[key]
                    for key in (
                        "facts",
                        "card_to_repair",
                        "conflicting_questions",
                        "json_location",
                    )
                    if key in source
                }
        except (ValueError, TypeError):
            pass
    rejected_candidate, omitted = _trim_rejected_json(rejected_json)
    errors_short, _ = _condensed_errors([str(error) for error in errors])
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
        + f"""

    VALIDATION ERRORS
    {"; ".join(errors_short)}

    REJECTED ONE-CARD JSON
    {rejected_candidate or "(invalid JSON omitted)"}
    {f"({omitted} whole entries omitted)" if omitted else ""}

    Return only {{"cards":[{{...one corrected card...}}]}}.

    INPUT JSON:
    {_json(context)}
    """
    )


def build_grounding_review_prompt(
    clusters: Sequence[FlashcardCluster],
) -> str:
    payload = {
        "clusters": [
            {
                "cluster": str(index),
                "concept": cluster.concept.name,
                "facts": list(cluster.concept.facts),
                "cards": [
                    {
                        "question": card.question,
                        **({"correct_option": card.correct_option} if card.correct_option else {}),
                        **({"wrong_option_1": card.wrong_option_1} if card.wrong_option_1 else {}),
                        **({"wrong_option_2": card.wrong_option_2} if card.wrong_option_2 else {}),
                        **({"wrong_option_3": card.wrong_option_3} if card.wrong_option_3 else {}),
                        **({"is_true": card.is_true} if card.is_true is not None else {}),
                        "expalanation": card.expalanation,
                        "hint": card.hint,
                    }
                    for card in cluster.cards
                ],
            }
            for index, cluster in enumerate(clusters, start=1)
        ]
    }
    return f"""Review these generated clusters against only their supplied facts. Questions, correct answers, true-false decisions, explanations, and hints must be supported by those facts. Flag a cluster if any card contains an unsupported claim, answer leakage, an invalid distractor, or a misleading explanation. Also flag a cluster when two cards ask the same question with at most one word different; cards that test the same fact in differently worded questions, in a different type, or in negated form are not duplicates. Do not infer duplication merely from the same answer-bearing relationship when there is a different type, polarity, or wording.

    A wrong option is not invalid merely because its wording does not appear in the supplied facts. It may use a familiar related term, but it must be plausible, unambiguously incorrect, in the same semantic category and answer shape as the correct option, and relevant to the question and module domain. Flag a distractor only when it is correct or arguably correct, duplicates another option, is nonsensical, mismatches the answer category, or is unrelated to the question or module domain. Do not rewrite cards.

    Return only this JSON shape. Use an empty issues list when no defect exists:
    {{"issues":[{{"cluster":"index copied from input","reasons":["unsupported claim"]}}]}}

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
                "cluster": str(index),
                "concept": cluster.concept.name,
                "questions": [card.question for card in cluster.cards],
            }
            for index, cluster in enumerate(clusters, start=1)
        ]
    }
    return """Compare all question stems for near duplication. Flag only a question that is the same sentence as another question with at most one filler, modifier, or synonym word added, removed, or swapped. Do not flag questions that merely assess the same fact or concept in differently worded questions, questions of different types, or a question and its negated form (for example \"is a component\" versus \"is NOT a component\"). Under this surface-only rule, a definition and its negated restatement are not duplicates unless the stems otherwise differ by at most one word. A swapped key term that changes what is asked (for example binary versus octal) makes a different question. Report each duplicate pair once: put the later cluster/card in the issue's cluster field and at the start of its reason, and identify the earlier card after the word cluster. Every reason must use exactly the form "card <number> duplicates cluster <index> card <number>". Do not judge factual correctness in this pass and do not rewrite questions.

    Return only this JSON shape. Use an empty issues list when no duplicate exists:
    {"issues":[{"cluster":"index copied from input","reasons":["card 2 duplicates cluster 3 card 4"]}]}

    INPUT JSON:
    """ + _json(
        payload
    )
