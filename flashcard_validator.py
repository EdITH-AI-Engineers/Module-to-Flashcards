from __future__ import annotations

import json
import difflib
import re
import unicodedata
import uuid
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
    "according to",
    "based on",
    "the material states",
    "the following claim",
    "consider this statement",
    "evaluate this statement",
    "identify the concept associated with",
)
PROVENANCE_PATTERNS = (
    r"\bknowledge graph\b",
    r"\bthe source\b",
    r"\bsource material\b",
    r"\bmodule\b",
    r"\bdocument\b",
    r"\blesson\b",
    r"\bslide\b",
    r"\bfile\b",
    r"\bchunk\b",
    r"\bcitation\b",
    r"\burl\b",
)
DIRECT_STEM = re.compile(r"^(what(?:\s+term)?|which|who|where|when|why|how)\b", re.I)
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


def normalize_stem(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w\s]", " ", value)
    return " ".join(value.split())


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
    if len(concepts_value) != CONCEPTS_PER_MODULE:
        raise ValidationError(
            f"expected exactly {CONCEPTS_PER_MODULE} concepts, "
            f"received {len(concepts_value)}"
        )

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
            fact_ids = _string_list(item.get("fact_ids"), f"{prefix} fact_ids")
            approaches = _string_list(
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
                f"{prefix} contains unsupported assessment approaches: {', '.join(invalid_approaches)}"
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
    return tuple(results[:20])


def _required_string(item: Mapping[str, Any], key: str, position: int) -> str:
    value = item.get(key)
    if not isinstance(value, str):
        raise ValidationError(f"card {position} field {key!r} must be a string")
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
        missing = sorted(CARD_FIELDS - set(item))
        if missing:
            raise ValidationError(f"card {position} is missing fields: {', '.join(missing)}")

        is_true = item["is_true"]
        if is_true is not None and not (type(is_true) is int and is_true in (0, 1)):
            raise ValidationError(
                f"card {position} is_true must be integer 0, integer 1, or null"
            )
        difficulty = item["difficulty"]
        if type(difficulty) is not int:
            raise ValidationError(f"card {position} difficulty must be an integer")

        results.append(
            FlashcardDraft(
                type=_required_string(item, "type", position),
                question=_required_string(item, "question", position),
                correct_option=_required_string(item, "correct_option", position),
                wrong_option_1=_required_string(item, "wrong_option_1", position),
                wrong_option_2=_required_string(item, "wrong_option_2", position),
                wrong_option_3=_required_string(item, "wrong_option_3", position),
                is_true=is_true,
                expalanation=_required_string(item, "expalanation", position),
                hint=_required_string(item, "hint", position),
                difficulty=difficulty,
                assessment_approach=_required_string(
                    item, "assessment_approach", position
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


def validate_cluster(
    cards: Sequence[FlashcardDraft],
    concept: ConceptPlan,
) -> tuple[str, ...]:
    errors: list[str] = []
    if len(cards) != CARDS_PER_CLUSTER:
        errors.append(
            f"expected exactly {CARDS_PER_CLUSTER} cards, received {len(cards)}"
        )

    counts = Counter(card.type for card in cards)
    if any(counts.get(card_type, 0) == 0 for card_type in ALLOWED_TYPES):
        errors.append(
            "cluster must contain at least one multiple-choice, identification, and true-false card"
        )

    approaches = [card.assessment_approach for card in cards]
    if (
        len(approaches) != CARDS_PER_CLUSTER
        or len(set(approaches)) != CARDS_PER_CLUSTER
    ):
        errors.append(
            f"cluster must use {CARDS_PER_CLUSTER} distinct assessment approaches"
        )
    if set(approaches) != set(concept.assessment_approaches):
        errors.append("cluster approaches must exactly match the planned assessment approaches")

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
        if card.assessment_approach not in ALLOWED_APPROACHES:
            errors.append(f"{prefix} assessment approach is not allowed")
        if any("\n" in value or "\r" in value for value in _text_fields(card)):
            errors.append(f"{prefix} fields must not contain line breaks")
        if any(phrase in card.question.casefold() for phrase in BANNED_FRAMING):
            errors.append(f"{prefix} question contains banned framing")
        if any(_contains_provenance(value) for value in _text_fields(card) if value):
            errors.append(f"{prefix} exposes provenance metadata")

        if card.type in {"multiple-choice", "identification"}:
            if not DIRECT_STEM.match(card.question.strip()):
                errors.append(f"{prefix} must use a direct question stem")
            if not card.question.rstrip().endswith("?"):
                errors.append(f"{prefix} direct question must end with a question mark")
        if card.type == "true-false":
            lowered = card.question.strip().casefold()
            if "?" in card.question or lowered.startswith(("true or false", "true/false")):
                errors.append(f"{prefix} true-false question must be a declarative statement only")

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
                    f"{prefix} multiple-choice requires one correct and three non-empty wrong options"
                )
            if len({_normalized(option) for option in options}) != 4:
                errors.append(f"{prefix} multiple-choice options must be distinct")
            if card.is_true is not None:
                errors.append(f"{prefix} multiple-choice is_true must be empty")

        elif card.type == "identification":
            if not card.correct_option.strip():
                errors.append(f"{prefix} identification correct option must not be empty")
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
                errors.append(f"{prefix} identification answer must be a concise phrase")
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


def are_near_duplicates(left: str, right: str) -> bool:
    normalized_left = normalize_stem(left)
    normalized_right = normalize_stem(right)
    if normalized_left == normalized_right:
        return True

    left_tokens = set(normalized_left.split())
    right_tokens = set(normalized_right.split())
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 1.0
    sequence = difflib.SequenceMatcher(
        None, normalized_left, normalized_right
    ).ratio()
    return jaccard >= 0.85 and sequence >= 0.88


def validate_module(
    clusters: Sequence[FlashcardCluster],
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
        cluster_errors = validate_cluster(cluster.cards, cluster.concept)
        errors.extend(
            f"cluster {cluster_position}: {error}" for error in cluster_errors
        )
        indexed_cards.extend(
            (cluster_position, card_position, card)
            for card_position, card in enumerate(cluster.cards, start=1)
        )

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
            raise ValidationError(f"review issue {position} references unknown cluster {cluster!r}")
        reasons = _string_list(item.get("reasons"), f"review issue {position} reasons")
        target = merged.setdefault(cluster, [])
        for reason in reasons:
            if reason not in target:
                target.append(reason)

    return tuple(
        ReviewIssue(cluster=cluster, reasons=tuple(reasons))
        for cluster, reasons in merged.items()
    )
