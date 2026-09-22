from __future__ import annotations

import json
import difflib
import re
import unicodedata
import uuid
import warnings
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from flashcard_contract import (
    CARDS_PER_CLUSTER,
    CARDS_PER_MODULE,
    CLUSTERS_PER_MODULE,
    CONCEPTS_PER_MODULE,
)
from flashcard_types import (
    ConceptPlan,
    FlashcardCluster,
    FlashcardDraft,
    GraphFact,
    ReviewIssue,
)

ALLOWED_TYPES = {"multiple-choice", "identification", "true-false"}
REQUIRED_TYPES = ALLOWED_TYPES
ALLOWED_APPROACHES = {
    "recall",
    "comparison",
    "classification",
    "application",
    "scenario analysis",
    "cause/effect",
    "misconception detection",
    "conditions",
    "consequences",
    "reversed reasoning",
}
BANNED_FRAMING = (
    "This statement accurately describes",
    "The following claim",
    "According to the graph",
    "According to the material",
    "Based on the material",
    "The material/module/lesson/document states",
    "Identify the concept associated with",
    "Consider the following statement",
    "Evaluate this statement",
)
PROVENANCE_PATTERNS = (
    r"\bknowledge graph\b",
    r"\bthe source\b",
    r"\bsource material\b",
    r"\bmodule(?:\s+content)?\b",
    r"\bdocument\b",
    r"\blesson\b",
    r"\bslides?\b",
    r"\bfile\b",
    r"\bchunk\b",
    r"\bcitation\b",
    r"\burl\b",
    r"\bconcept facts?\b",
    r"\bdistractor pool\b",
    r"\ballowed[_\s]wrong[_\s]option[_\s]terms\b",
    r"\b(?:provided|supplied|given)\s+"
    r"(?:facts?|material|information|text|source(?:\s+material)?)\b",
    r"\b(?:in|from)\s+the\s+(?:facts?|material|text|information|slides?)\b",
)
DIRECT_STEM = re.compile(r"^(what(?:\s+term)?|which|who|where|when|why|how)\b", re.I)
LEADING_WRAPPER = re.compile(r"^(according to|based on)\b", re.I)
SCENARIO_ACTOR = re.compile(
    r"\b(?:a|an|the)\s+(?:user|learner|student|designer|developer|team|"
    r"organization|operator|employee|customer|participant|person|group|company|"
    r"system|interface|application|website|device|product|workflow|task)\b",
    re.I,
)
SCENARIO_LEAD = re.compile(
    r"^(?:if|when|whenever|after|before|during|suppose|imagine|given that)\b",
    re.I,
)
CARD_FIELDS = {
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
}
COMMON_CARD_FIELDS = {
    "type",
    "question",
    "hint",
    "difficulty",
    "assessment_approach",
}
TYPE_SPECIFIC_FIELDS = {
    "multiple-choice": {
        "correct_option",
        "wrong_option_1",
        "wrong_option_2",
        "wrong_option_3",
        "is_true",
    },
    "identification": {"correct_option", "is_true"},
    "true-false": {"is_true"},
}
DEFAULT_ASSESSMENT_APPROACHES = (
    "recall",
    "comparison",
    "application",
    "misconception detection",
    "reversed reasoning",
)


class ValidationError(ValueError):
    def __init__(self, errors: str | Iterable[str]):
        if isinstance(errors, str):
            normalized = (errors,)
        else:
            normalized = tuple(str(error) for error in errors)
        self.errors = normalized or ("validation failed",)
        super().__init__("; ".join(self.errors))


class InsufficientContentError(ValidationError):
    """Raised when the model explicitly reports too few supported concepts."""


def _parse_json_object(raw: str) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise ValidationError("response must be text containing a JSON object only")
    stripped = raw.strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        raise ValidationError("response must contain a JSON object only")
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"response is not valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValidationError("response JSON root must be an object")
    return value


def _string_list(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValidationError(f"{label} must be a non-empty JSON array")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValidationError(f"{label} must contain only non-empty strings")
    return tuple(value)


def _tolerant_string_list(value: Any, label: str) -> tuple[str, ...]:
    """Like _string_list, but also accepts a single comma-separated string
    in place of a proper JSON array — the local model sometimes collapses
    a short list of tokens (fact IDs, approach names) into one string
    instead of an actual array. Only use this for token-like fields with
    no legitimate commas inside a single item; freeform text fields
    (e.g. review reasons) must keep using the strict _string_list."""
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed.startswith("[") and trimmed.endswith("]"):
            trimmed = trimmed[1:-1]
        parts = [part.strip().strip("'\"") for part in trimmed.split(",")]
        parts = [part for part in parts if part]
        if parts:
            value = parts
    return _string_list(value, label)


def normalize_stem(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w\s]", " ", value)
    return " ".join(value.split())


def _stem(token: str) -> str:
    """Light, dependency-free suffix stripping so trivial inflections (the
    module facts saying "systems" while a generated option says "system",
    "designing" vs "design", etc.) don't fail grounding purely on plurals
    or verb endings. This is intentionally crude -- it only exists to stop
    the validator from rejecting options that a human would clearly
    recognize as reusing the same word, not to do real linguistic
    stemming."""
    derivational_roots = {
        "safe": "safe",
        "safety": "safe",
        "comfort": "comfort",
        "comfortable": "comfort",
        "enjoy": "enjoy",
        "enjoyable": "enjoy",
        "enjoyment": "enjoy",
        "efficient": "efficient",
        "efficiency": "efficient",
        "effective": "effective",
        "effectiveness": "effective",
        "satisfied": "satisfy",
        "satisfaction": "satisfy",
    }
    if token in derivational_roots:
        return derivational_roots[token]
    if token.isdigit():
        return token
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    return token


def _grounding_tokens(value: str) -> set[str]:
    stop_words = {"a", "an", "and", "or", "the", "of", "to", "in", "is", "are"}
    return {
        _stem(token)
        for token in re.findall(r"[a-z]+|\d+", value.casefold())
        if token not in stop_words
    }


def _normalized(value: str) -> str:
    return normalize_stem(value)


def parse_concept_plan(
    raw: str,
    known_facts: Sequence[GraphFact],
) -> tuple[ConceptPlan, ...]:
    value = _parse_json_object(raw)
    if "insufficient_content" in value:
        reason = value["insufficient_content"]
        if (
            not isinstance(reason, str)
            or len(re.findall(r"\b\w+\b", reason)) < 5
            or reason.strip().casefold() in {"concise reason", "specific reason"}
            or "..." in reason
        ):
            raise ValidationError(
                "insufficient_content must explain the specific limitation in at least five words"
            )
        raise InsufficientContentError(str(reason))

    concepts_value = value.get("concepts")
    if not isinstance(concepts_value, list):
        raise ValidationError("concepts must be a JSON array")
    if len(concepts_value) < CONCEPTS_PER_MODULE:
        raise ValidationError(
            f"expected at least {CONCEPTS_PER_MODULE} concepts, "
            f"received {len(concepts_value)}"
        )
    concepts_value = concepts_value[:CONCEPTS_PER_MODULE]

    known = {fact.fact_id: fact.statement for fact in known_facts}
    results: list[ConceptPlan] = []
    errors: list[str] = []
    seen_names: set[str] = set()

    for position, item in enumerate(concepts_value, start=1):
        prefix = f"concept {position}"
        if not isinstance(item, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{prefix} name must be a non-empty string")
            continue
        normalized_name = _normalized(name)
        if normalized_name in seen_names:
            errors.append(f"{prefix} duplicates another concept name")
        seen_names.add(normalized_name)

        try:
            fact_ids = _tolerant_string_list(item.get("fact_ids"), f"{prefix} fact_ids")
            approaches = _tolerant_string_list(
                item.get("assessment_approaches"),
                f"{prefix} assessment_approaches",
            )
        except ValidationError as exc:
            errors.extend(exc.errors)
            continue

        resolved_facts: list[str] = []
        for fact_id in fact_ids:
            if fact_id not in known:
                errors.append(f"{prefix} uses unknown fact id {fact_id!r}")
            else:
                resolved_facts.append(known[fact_id])
        if (
            len(approaches) != CARDS_PER_CLUSTER
            or len(set(approaches)) != CARDS_PER_CLUSTER
        ):
            errors.append(
                f"{prefix} must contain exactly {CARDS_PER_CLUSTER} "
                "distinct assessment approaches"
            )
        invalid_approaches = sorted(set(approaches) - ALLOWED_APPROACHES)
        if invalid_approaches:
            errors.append(
                f"{prefix} contains unsupported assessment approaches: "
                f"{', '.join(invalid_approaches)}"
            )
        results.append(
            ConceptPlan(
                name=name,
                fact_ids=fact_ids,
                facts=tuple(resolved_facts),
                assessment_approaches=approaches,
            )
        )

    if errors:
        raise ValidationError(errors)
    return tuple(results)


def _required_string(item: Mapping[str, Any], key: str, position: int) -> str:
    value = item.get(key)
    return _required_string_value(value, key, position)


def _required_string_value(value: Any, key: str, position: int) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"card {position} field {key!r} must be a string")
    return normalize_generated_text(value)


def normalize_generated_text(value: str) -> str:
    """Return model text with apostrophes in a portable ASCII form.

    Local model output can contain either smart apostrophes or their common
    UTF-8-as-Windows-1252 mojibake forms. Normalizing at the parser boundary
    keeps every exported card field consistent regardless of which form the
    backend returned.
    """

    normalized = unicodedata.normalize("NFC", value)
    for broken in ("\u00e2\u20ac\u2122", "\u00e2\u20ac\u02dc"):
        normalized = normalized.replace(broken, "'")
    return normalized.translate(
        str.maketrans(
            {
                "\u2018": "'",
                "\u2019": "'",
                "\u02bc": "'",
            }
        )
    )


def _option_value(
    item: Mapping[str, Any], key: str, position: int, card_type: str
) -> str:
    """Return an option value, normalizing structural True/False labels.

    Some local models fill the otherwise-unused option fields of a true-false
    card with the literal labels ``True`` and ``False``. The actual answer is
    carried by ``is_true``, so those labels are safe to treat as empty fields.
    Other non-empty values remain untouched and are rejected by validation.
    """
    if key not in item:
        return ""
    value = _required_string(item, key, position)
    if card_type == "true-false" and value.strip().casefold() in {"true", "false"}:
        return ""
    return value


def parse_cards(raw: str) -> tuple[FlashcardDraft, ...]:
    value = _parse_json_object(raw)
    cards_value = value.get("cards")
    if not isinstance(cards_value, list):
        raise ValidationError("cards must be a JSON array")

    results: list[FlashcardDraft] = []
    for position, item in enumerate(cards_value, start=1):
        if not isinstance(item, Mapping):
            raise ValidationError(f"card {position} must be an object")
        card_type = item.get("type")
        if not isinstance(card_type, str):
            raise ValidationError(f"card {position} field 'type' must be a string")
        allowed_fields = CARD_FIELDS | {"explanation"}
        unknown_fields = sorted(set(item) - allowed_fields)
        if unknown_fields:
            raise ValidationError(
                f"card {position} contains unknown fields: {', '.join(unknown_fields)}"
            )
        required = COMMON_CARD_FIELDS | TYPE_SPECIFIC_FIELDS.get(card_type, set())
        missing = sorted(required - set(item))
        if "expalanation" not in item and "explanation" not in item:
            missing.append("expalanation")
        metadata_missing = set(missing) & {"difficulty", "assessment_approach"}
        missing = [field for field in missing if field not in metadata_missing]
        if missing:
            raise ValidationError(
                f"card {position} is missing fields: {', '.join(missing)}"
            )
        if any(
            isinstance(field_value, str)
            and re.search(
                r"\b(?:distractor pool|allowed[_\s]wrong[_\s]option[_\s]terms)\b",
                field_value,
                flags=re.I,
            )
            for field_value in item.values()
        ):
            raise ValidationError(
                f"card {position} treats internal distractor data as answer authority"
            )

        if "difficulty" in metadata_missing:
            warnings.warn(
                f"card {position} missing difficulty; defaulting to 2",
                RuntimeWarning,
                stacklevel=2,
            )
        if "assessment_approach" in metadata_missing:
            warnings.warn(
                f"card {position} missing assessment_approach; defaulting to a standard approach",
                RuntimeWarning,
                stacklevel=2,
            )

        is_true = item.get("is_true")
        if is_true is not None and not (type(is_true) is int and is_true in (0, 1)):
            raise ValidationError(
                f"card {position} is_true must be integer 0, integer 1, or null"
            )
        difficulty = item.get("difficulty", 2)
        if type(difficulty) is not int:
            raise ValidationError(f"card {position} difficulty must be an integer")
        assessment_approach = item.get("assessment_approach")
        if assessment_approach is None:
            assessment_approach = DEFAULT_ASSESSMENT_APPROACHES[
                (position - 1) % len(DEFAULT_ASSESSMENT_APPROACHES)
            ]

        correct_option = _option_value(item, "correct_option", position, card_type)
        wrong_option_1 = _option_value(item, "wrong_option_1", position, card_type)
        wrong_option_2 = _option_value(item, "wrong_option_2", position, card_type)
        wrong_option_3 = _option_value(item, "wrong_option_3", position, card_type)
        raw_expalanation = _required_string(
            item,
            "expalanation" if "expalanation" in item else "explanation",
            position,
        )
        raw_hint = _required_string(item, "hint", position)

        raw_question = _required_string(item, "question", position)
        results.append(
            FlashcardDraft(
                type=card_type,
                question=_sanitize_question(raw_question),
                correct_option=correct_option,
                wrong_option_1=wrong_option_1,
                wrong_option_2=wrong_option_2,
                wrong_option_3=wrong_option_3,
                is_true=is_true,
                expalanation=_sanitize_explanation(
                    raw_expalanation, card_type, correct_option, is_true
                ),
                hint=_sanitize_hint(raw_hint),
                difficulty=difficulty,
                assessment_approach=_required_string_value(
                    assessment_approach, "assessment_approach", position
                ),
            )
        )
    return tuple(results)


def _text_fields(card: FlashcardDraft) -> tuple[str, ...]:
    return (
        card.question,
        card.correct_option,
        card.wrong_option_1,
        card.wrong_option_2,
        card.wrong_option_3,
        card.expalanation,
        card.hint,
    )


def _contains_provenance(value: str) -> bool:
    return any(re.search(pattern, value, flags=re.I) for pattern in PROVENANCE_PATTERNS)


_PROVENANCE_SOURCE = (
    r"(?:(?:provided|supplied|given)\s+)?"
    r"(?:concept\s+facts?|distractor\s+pool|facts?|material|"
    r"module(?:\s+content)?|document|lesson|slides?|"
    r"source(?:\s+material)?|knowledge\s+graph|file|text|information)"
)
_EXPLICIT_STATED_AS_RE = re.compile(
    r"\bis\s+explicitly\s+stated\s+as\s+(.+?)\s+in\s+the\s+(?:provided|supplied)\s+facts?\b",
    re.I,
)
_LEADING_FACTS_STATE_RE = re.compile(
    r"^\s*the\s+(?:provided|supplied)\s+facts?\s+(?:explicitly\s+)?"
    r"(?:state|states|stated|support|supports|supported|indicate|indicates|"
    r"show|shows|confirm|confirms)\s+that\s+",
    re.I,
)
_PROVENANCE_WRAPPER_RE = re.compile(
    rf",?\s*(?:(?:according\s+to|based\s+on)\s+(?:the\s+)?"
    rf"{_PROVENANCE_SOURCE}|as\s+(?:stated|described)\s+in\s+"
    rf"(?:the\s+)?{_PROVENANCE_SOURCE})\b(?=\s*[,?.!]|$),?\s*",
    re.I,
)
_DANGLING_STATED_IN_RE = re.compile(
    r"\bis\s+explicitly\s+stated\s+in\s+the\s+(?:provided|supplied)\s+facts?\b", re.I
)
_LEFTOVER_FACTS_RE = re.compile(
    r"\b(?:in\s+)?(?:the\s+)?(?:provided|supplied)\s+facts?\b", re.I
)


def _strip_citation_phrasing(text: str) -> str:
    """Remove formulaic "as stated in the provided/supplied fact(s)" citation
    clauses from model-written prose, keeping any real content around them.

    A small local model very often narrates where a claim came from instead
    of just stating the claim -- e.g. "X is explicitly stated as Y in the
    provided fact" instead of "X is Y". That trips the provenance-wording
    rule even though the underlying claim is perfectly fine, and the model
    frequently fails to stop doing it even after being told to in a retry.
    This rewrites the common shapes in place of relying on further retries.
    """
    text = _EXPLICIT_STATED_AS_RE.sub(r"is \1", text)
    text = _LEADING_FACTS_STATE_RE.sub("", text)
    text = _PROVENANCE_WRAPPER_RE.sub(" ", text)
    text = _DANGLING_STATED_IN_RE.sub("", text)
    text = _LEFTOVER_FACTS_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,?!])", r"\1", text)
    text = re.sub(r"\.\s*\.", ".", text)
    text = text.strip(" ,")
    if text and not text.endswith((".", "!", "?")):
        text += "."
    if text:
        text = text[0].upper() + text[1:]
    return text


def _explanation_fallback(
    card_type: str, correct_option: str, is_true: int | None
) -> str:
    if card_type == "true-false":
        return f"This statement is {'true' if is_true == 1 else 'false'}."
    option = correct_option.strip().rstrip(".")
    if option:
        return f"{option} is the correct answer here."
    return "This is the correct answer for this question."


def _sanitize_explanation(
    text: str, card_type: str, correct_option: str, is_true: int | None
) -> str:
    cleaned = _strip_citation_phrasing(text)
    words = re.findall(r"[A-Za-z0-9]+", cleaned)
    # After stripping the citation clause there may be nothing substantive
    # left (e.g. "X is explicitly stated in the provided fact." carried no
    # real content beyond the citation) -- fall back rather than ship a
    # near-empty or still-flagged explanation.
    if len(words) < 6 or _contains_provenance(cleaned):
        return _explanation_fallback(card_type, correct_option, is_true)
    return cleaned


def _sanitize_hint(text: str) -> str:
    cleaned = _strip_citation_phrasing(text)
    if not cleaned or _contains_provenance(cleaned):
        return "Consider the key relationship or distinction central to this topic."
    return cleaned


def _sanitize_question(text: str) -> str:
    """Strip the same citation clauses from the question stem itself.

    Explanations and hints aren't the only place this leaks -- the model
    just as often tacks the wrapper directly onto the question, e.g. "What
    is the key aspect of HCI according to the provided facts?" The wrapper
    carries no assessable content, so removing it only makes the stem
    cleaner; it never changes what is actually being asked. If stripping
    would remove more than the wrapper (leaving too little of the
    question), the original text is kept so normal validation still
    catches whatever is really wrong with it.
    """
    cleaned = _strip_citation_phrasing(text)
    if not cleaned:
        return text
    if len(re.findall(r"[A-Za-z0-9]+", cleaned)) < 3:
        return text
    return cleaned


def validate_cluster(
    cards: Sequence[FlashcardDraft],
    concept: ConceptPlan,
    source_facts: Sequence[GraphFact] = (),
) -> tuple[str, ...]:
    errors: list[str] = []
    if len(cards) != CARDS_PER_CLUSTER:
        errors.append(
            f"expected exactly {CARDS_PER_CLUSTER} cards, received {len(cards)}"
        )

    counts = Counter(card.type for card in cards)
    if any(counts.get(card_type, 0) == 0 for card_type in REQUIRED_TYPES):
        errors.append(
            "cluster must contain at least one multiple-choice, identification, and true-false card"
        )

    expected_approaches = set(concept.assessment_approaches)
    for position, card in enumerate(cards, start=1):
        if card.assessment_approach not in expected_approaches:
            errors.append(
                f"card {position} assessment_approach must be one of the planned "
                f"approaches: {', '.join(sorted(expected_approaches))}"
            )

    if source_facts:
        grounded_terms = {
            token
            for fact in source_facts
            for token in _grounding_tokens(fact.statement)
        }
        for fact in source_facts:
            if fact.topic:
                grounded_terms.update(_grounding_tokens(fact.topic))
        grounded_terms.update(_grounding_tokens(concept.name))
        grounded_terms.update(
            token for fact in concept.facts for token in _grounding_tokens(fact)
        )
        for position, card in enumerate(cards, start=1):
            if card.type != "multiple-choice":
                continue
            for option_name in ("wrong_option_1", "wrong_option_2", "wrong_option_3"):
                option_terms = set(_grounding_tokens(getattr(card, option_name)))
                if option_terms and not option_terms & grounded_terms:
                    errors.append(
                        f"card {position} {option_name} is not grounded in supplied module facts"
                    )

    for position, card in enumerate(cards, start=1):
        prefix = f"card {position}"
        if card.type not in ALLOWED_TYPES:
            errors.append(f"{prefix} type must be one of {sorted(ALLOWED_TYPES)}")
        if not card.question.strip():
            errors.append(f"{prefix} question must not be empty")
        if not card.expalanation.strip():
            errors.append(f"{prefix} expalanation must not be empty")
        if not card.hint.strip():
            errors.append(f"{prefix} hint must not be empty")
        if type(card.difficulty) is not int or card.difficulty not in (1, 2, 3):
            errors.append(f"{prefix} difficulty must be 1, 2, or 3")
        if (
            not isinstance(card.assessment_approach, str)
            or not card.assessment_approach.strip()
        ):
            errors.append(f"{prefix} assessment approach must not be empty")
        elif (
            card.assessment_approach == "scenario analysis"
            and not (
                SCENARIO_ACTOR.search(card.question)
                or SCENARIO_LEAD.search(card.question.strip())
            )
        ):
            errors.append(
                f"{prefix} scenario analysis must present a concrete situation "
                "before asking for interpretation"
            )
        if any("\n" in value or "\r" in value for value in _text_fields(card)):
            errors.append(f"{prefix} fields must not contain line breaks")
        if LEADING_WRAPPER.match(card.question.strip()) or any(
            phrase.casefold() in card.question.casefold() for phrase in BANNED_FRAMING
        ):
            errors.append(f"{prefix} question contains banned framing")
        if any(_contains_provenance(value) for value in _text_fields(card) if value):
            errors.append(f"{prefix} exposes provenance metadata")

        if card.type == "identification" and not DIRECT_STEM.match(
            card.question.strip()
        ):
            errors.append(f"{prefix} must use a direct question stem")
        if card.type in {"multiple-choice", "identification"}:
            if not card.question.rstrip().endswith("?"):
                errors.append(f"{prefix} direct question must end with a question mark")
        if card.type == "true-false":
            lowered = card.question.strip().casefold()
            if "?" in card.question or lowered.startswith(
                ("true or false", "true/false")
            ):
                errors.append(
                    f"{prefix} true-false question must be a declarative statement only"
                )

        if card.type == "multiple-choice":
            options = (
                card.correct_option,
                card.wrong_option_1,
                card.wrong_option_2,
                card.wrong_option_3,
            )
            if not card.correct_option.strip() or any(
                not option.strip() for option in options[1:]
            ):
                errors.append(
                    f"{prefix} {card.type} requires one correct and three non-empty wrong options"
                )
            if len({_normalized(option) for option in options}) != 4:
                errors.append(f"{prefix} {card.type} options must be distinct")
            if card.is_true is not None:
                errors.append(f"{prefix} {card.type} is_true must be empty")
        elif card.type == "identification":
            if not card.correct_option.strip():
                errors.append(
                    f"{prefix} identification correct option must not be empty"
                )
            if any(
                option.strip()
                for option in (
                    card.wrong_option_1,
                    card.wrong_option_2,
                    card.wrong_option_3,
                )
            ):
                errors.append(f"{prefix} identification wrong options must be empty")
            if card.is_true is not None:
                errors.append(f"{prefix} identification is_true must be empty")
            answer = card.correct_option.strip()
            if len(answer.split()) > 12 or answer.endswith((".", "?", "!")):
                errors.append(
                    f"{prefix} identification answer must be a concise phrase"
                )
            normalized_answer = _normalized(answer)
            if normalized_answer and normalized_answer in _normalized(card.question):
                errors.append(f"{prefix} question reveals the identification answer")

        elif card.type == "true-false":
            if any(
                option.strip()
                for option in (
                    card.correct_option,
                    card.wrong_option_1,
                    card.wrong_option_2,
                    card.wrong_option_3,
                )
            ):
                errors.append(f"{prefix} true-false answer options must be empty")
            if type(card.is_true) is not int or card.is_true not in (0, 1):
                errors.append(f"{prefix} true-false is_true must be 0 or 1")

        normalized_answer = _normalized(card.correct_option)
        if normalized_answer and normalized_answer in _normalized(card.hint):
            errors.append(f"{prefix} hint reveals the correct answer")

    return tuple(errors)


def _polarity_variant(left: str, right: str) -> bool:
    polarity = {"not", "never", "without", "except", "least", "most"}
    left_tokens = normalize_stem(left).split()
    right_tokens = normalize_stem(right).split()
    left_base = [token for token in left_tokens if token not in polarity]
    right_base = [token for token in right_tokens if token not in polarity]
    return left_base == right_base and bool(
        (set(left_tokens) ^ set(left_base)) or (set(right_tokens) ^ set(right_base))
    )


_QUESTION_FRAME_TOKENS = {
    "how",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
}


def _question_content_tokens(value: str) -> tuple[str, ...]:
    return tuple(
        _stem(token)
        for token in normalize_stem(value).split()
        if token not in _QUESTION_FRAME_TOKENS
    )


def are_near_duplicates(left: str, right: str) -> bool:
    normalized_left = normalize_stem(left)
    normalized_right = normalize_stem(right)
    if normalized_left == normalized_right:
        return True

    left_sequence = _question_content_tokens(left)
    right_sequence = _question_content_tokens(right)
    left_tokens = set(left_sequence)
    right_tokens = set(right_sequence)
    if not left_tokens or not right_tokens:
        return False

    # A high surface score can come from a long shared template with one
    # different learning point (for example, "binary" versus "octal").
    # Deterministic rejection is safe only when the content difference is
    # one-sided, such as an added filler word, or is only question framing.
    if left_tokens - right_tokens and right_tokens - left_tokens:
        return False

    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 1.0
    sequence = difflib.SequenceMatcher(
        None,
        " ".join(left_sequence),
        " ".join(right_sequence),
    ).ratio()
    return jaccard >= 0.85 and sequence >= 0.88


def validate_module(
    clusters: Sequence[FlashcardCluster],
    source_facts: Sequence[GraphFact] = (),
    *,
    check_question_duplicates: bool = True,
    skip_grounding_cluster_ids: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    errors: list[str] = []
    if len(clusters) != CLUSTERS_PER_MODULE:
        errors.append(
            f"module must contain exactly {CLUSTERS_PER_MODULE} clusters, "
            f"received {len(clusters)}"
        )

    cluster_ids = [cluster.cluster for cluster in clusters]
    if len(set(cluster_ids)) != len(cluster_ids):
        errors.append("cluster UUIDs must be unique")
    for position, cluster_id in enumerate(cluster_ids, start=1):
        try:
            parsed = uuid.UUID(cluster_id)
        except (ValueError, AttributeError, TypeError):
            errors.append(f"cluster {position} must use a valid UUID")
            continue
        if str(parsed) != cluster_id.casefold():
            errors.append(f"cluster {position} must use a canonical valid UUID")

    total_cards = sum(len(cluster.cards) for cluster in clusters)
    if total_cards != CARDS_PER_MODULE:
        errors.append(
            f"module must contain exactly {CARDS_PER_MODULE} cards, "
            f"received {total_cards}"
        )

    indexed_cards: list[tuple[int, int, FlashcardDraft]] = []
    for cluster_position, cluster in enumerate(clusters, start=1):
        grounding_facts = (
            () if cluster.cluster in skip_grounding_cluster_ids else source_facts
        )
        cluster_errors = validate_cluster(
            cluster.cards,
            cluster.concept,
            grounding_facts,
        )
        errors.extend(
            f"cluster {cluster_position}: {error}" for error in cluster_errors
        )
        indexed_cards.extend(
            (cluster_position, card_position, card)
            for card_position, card in enumerate(cluster.cards, start=1)
        )

    if check_question_duplicates:
        for left_index, (left_cluster, left_card, left) in enumerate(indexed_cards):
            for right_cluster, right_card, right in indexed_cards[left_index + 1 :]:
                if are_near_duplicates(left.question, right.question):
                    errors.append(
                        "near-duplicate questions at "
                        f"cluster {left_cluster} card {left_card} and "
                        f"cluster {right_cluster} card {right_card}"
                    )

    return tuple(errors)


def parse_review_issues(
    raw: str,
    known_clusters: set[str],
) -> tuple[ReviewIssue, ...]:
    value = _parse_json_object(raw)
    issues_value = value.get("issues")
    if not isinstance(issues_value, list):
        raise ValidationError("issues must be a JSON array")

    merged: dict[str, list[str]] = {}
    for position, item in enumerate(issues_value, start=1):
        if not isinstance(item, Mapping):
            raise ValidationError(f"review issue {position} must be an object")
        cluster = item.get("cluster")
        if not isinstance(cluster, str) or not cluster:
            raise ValidationError(f"review issue {position} cluster must be a string")
        if cluster not in known_clusters:
            raise ValidationError(
                f"review issue {position} references unknown cluster {cluster!r}"
            )
        reasons = _string_list(item.get("reasons"), f"review issue {position} reasons")
        target = merged.setdefault(cluster, [])
        for reason in reasons:
            if reason not in target:
                target.append(reason)

    return tuple(
        ReviewIssue(cluster=cluster, reasons=tuple(reasons))
        for cluster, reasons in merged.items()
    )
