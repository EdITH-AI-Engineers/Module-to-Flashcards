from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Callable, Sequence, TypeVar
from uuid import uuid4

from flashcard_contract import (
    CARDS_PER_BLOCK,
    CARDS_PER_CLUSTER,
    CARDS_PER_MODULE,
    CLUSTERS_PER_MODULE,
    CONCEPTS_PER_MODULE,
)
from flashcard_prompt import (
    SYSTEM_PROMPT,
    build_cluster_prompt,
    build_cluster_retry_prompt,
    build_concept_plan_prompt,
    build_concept_plan_retry_prompt,
    build_duplicate_card_repair_prompt,
    build_duplicate_card_retry_prompt,
    build_duplicate_review_prompt,
    build_grounding_review_prompt,
    build_retry_prompt,
    _card_subject,
    _dedup_fingerprint,
)
from flashcard_schema import (
    build_card_cluster_schema,
    build_concept_plan_schema,
    build_review_schema,
    build_single_card_schema,
)
from flashcard_types import (
    ChatBackend,
    CompletionTruncatedError,
    ConceptPlan,
    FlashcardCluster,
    FlashcardDraft,
    GraphFact,
    ModuleIdentity,
    ReviewIssue,
)
from flashcard_validator import (
    InsufficientContentError,
    ValidationError,
    are_near_duplicates,
    parse_cards,
    parse_concept_plan,
    parse_review_issues,
    _polarity_variant,
    validate_cluster,
    validate_module,
)


class GenerationError(RuntimeError):
    """Raised when bounded generation cannot produce a valid module."""


@dataclass(frozen=True)
class PipelineConfig:
    max_retries: int = 3
    plan_max_tokens: int = 3072
    cluster_max_tokens: int = 1536
    review_max_tokens: int = 1024
    final_review: bool = True

    def __post_init__(self) -> None:
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1")


Parsed = TypeVar("Parsed")
PLAN_FACT_LIMIT = 50

# Fact `kind` values that mark structural/presentation content (slide titles,
# section headers, module headings) rather than teachable material. Facts
# tagged with one of these never reach the concept planner, so it cannot
# cluster on them and produce an unwritable "Module N Title"-style concept.
# `parse_concept_plan` also rejects any concept whose name or every backing
# fact reads as provenance/presentation content, as a second layer of
# defense for facts whose `kind` doesn't (or can't) mark them this way.
STRUCTURAL_FACT_KINDS = frozenset({"title", "header", "heading", "objective"})


def _exclude_structural_facts(
    facts: Sequence[GraphFact],
) -> tuple[GraphFact, ...]:
    """Drop facts whose `kind` marks them as structural/provenance-only.

    These facts (slide titles, section headers, module headings) have no
    content of their own to assess -- any card built from one can only ever
    restate the title/heading itself, which is exactly the presentation
    metadata the card-writing prompt is told never to expose. Keeping them
    out of the planner's candidate pool prevents it from ever proposing a
    concept that is unwritable for that reason.
    """
    return tuple(fact for fact in facts if fact.kind not in STRUCTURAL_FACT_KINDS)


def _balanced_plan_facts(
    facts: Sequence[GraphFact],
    limit: int = PLAN_FACT_LIMIT,
) -> tuple[GraphFact, ...]:
    """Bound prompt size while retaining coverage across lesson topics/slides."""
    values = tuple(facts)
    if len(values) <= limit:
        return values

    buckets: dict[tuple[str, object], list[GraphFact]] = {}
    for fact in values:
        if fact.slides:
            key: tuple[str, object] = ("slide", fact.slides[0])
        elif fact.topic:
            key = ("topic", fact.topic.casefold())
        else:
            key = ("general", "")
        buckets.setdefault(key, []).append(fact)

    selected: list[GraphFact] = []
    offset = 0
    while len(selected) < limit:
        added = False
        for bucket in buckets.values():
            if offset < len(bucket):
                selected.append(bucket[offset])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        offset += 1
    return tuple(selected)


DISTRACTOR_FACT_LIMIT = 12


_HINT_LEAK_ERROR_RE = re.compile(r"^card (\d+) hint reveals the correct answer$")
_MODULE_DUPLICATE_RE = re.compile(
    r"^near-duplicate questions at cluster (\d+) card (\d+) "
    r"and cluster (\d+) card (\d+)$"
)
_DUPLICATE_REVIEW_REASON_RE = re.compile(
    r"^card (\d+) duplicates cluster (.+?) card (\d+)$",
    re.I,
)


@dataclass(frozen=True)
class _DuplicateRepairItem:
    cluster_index: int
    card_index: int
    original_card: FlashcardDraft
    conflicts: tuple[tuple[int, int], ...]
    reasons: tuple[str, ...]


def _repair_grounding_errors(
    cards: Sequence[FlashcardDraft],
    errors: Sequence[str],
    grounding_facts: Sequence[GraphFact],
    phrase_facts: Sequence[GraphFact] | None = None,
) -> tuple[FlashcardDraft, ...] | None:
    """Deterministically repair hint leakage without rewriting distractors.

    The legacy function name and unused fact parameters are retained for
    compatibility with existing callers. Wrong options are semantic choices:
    they may use relevant outside-corpus terminology and must never be replaced
    mechanically with an arbitrary phrase copied from another fact.
    """
    del grounding_facts, phrase_facts
    hint_matches = [
        match
        for error in errors
        if (match := _HINT_LEAK_ERROR_RE.match(error)) is not None
    ]
    if not errors or len(hint_matches) != len(errors):
        return None

    cards_list = list(cards)
    for match in hint_matches:
        card_index = int(match.group(1)) - 1
        if not (0 <= card_index < len(cards_list)):
            return None
        cards_list[card_index] = replace(
            cards_list[card_index],
            hint="Consider the relationship or distinction needed to answer.",
        )

    return tuple(cards_list)


def _select_distractor_facts(
    facts: Sequence[GraphFact],
    concept: ConceptPlan,
    limit: int = DISTRACTOR_FACT_LIMIT,
) -> tuple[GraphFact, ...]:
    """Pick nearby material that can suggest topically relevant wrong options.

    A blind positional slice of "the first N facts that are not this
    concept's own" tends to hand the model facts from a completely
    unrelated part of the module. With nothing usable to adapt, the model
    tends to fall back on a generic or unrelated guess. Preferring facts that
    share the concept's topic, then facts on nearby slides, gives the model
    useful vocabulary without making exact corpus wording mandatory.
    """
    concept_fact_ids = set(concept.fact_ids)
    candidates = [fact for fact in facts if fact.fact_id not in concept_fact_ids]
    if len(candidates) <= limit:
        return tuple(candidates)

    concept_topics = {
        fact.topic.casefold()
        for fact in facts
        if fact.fact_id in concept_fact_ids and fact.topic
    }
    concept_slides = {
        slide
        for fact in facts
        if fact.fact_id in concept_fact_ids
        for slide in fact.slides
    }

    def _slide_distance(fact: GraphFact) -> int:
        if not fact.slides or not concept_slides:
            return 10_000
        return min(
            abs(slide - target) for slide in fact.slides for target in concept_slides
        )

    same_topic = [
        fact
        for fact in candidates
        if fact.topic and fact.topic.casefold() in concept_topics
    ]
    remaining = [fact for fact in candidates if fact not in same_topic]
    remaining.sort(key=_slide_distance)

    return tuple((same_topic + remaining)[:limit])


class FlashcardPipeline:
    def __init__(
        self,
        backend: ChatBackend,
        config: PipelineConfig = PipelineConfig(),
        *,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.backend = backend
        self.config = config
        self._progress = progress or (lambda message: print(message, flush=True))
        self._attempt_count = 0
        self._rejected_attempt_count = 0
        self._rejection_counts: Counter[str] = Counter()

    @property
    def rejection_stats(self) -> dict[str, object]:
        """Return rejection metrics for the current or most recent run."""

        rate = (
            self._rejected_attempt_count / self._attempt_count
            if self._attempt_count
            else 0.0
        )
        return {
            "attempts": self._attempt_count,
            "rejected_attempts": self._rejected_attempt_count,
            "rejection_rate": rate,
            "categories": dict(sorted(self._rejection_counts.items())),
        }

    @staticmethod
    def _rejection_category(error: str) -> str:
        lowered = error.casefold()
        if "truncated" in lowered or "length limit" in lowered:
            return "truncation"
        if "not grounded" in lowered:
            return "grounding"
        if "assessment approach" in lowered or "scenario analysis" in lowered:
            return "assessment approach"
        if "duplicate" in lowered:
            return "duplication"
        if any(
            marker in lowered
            for marker in (
                "unknown fields",
                "missing fields",
                "json",
                "expected exactly",
                "must be an object",
                "must be a json array",
            )
        ):
            return "schema/structure"
        return "other validation"

    def _record_rejection(self, errors: Sequence[str]) -> None:
        self._rejected_attempt_count += 1
        categories = {self._rejection_category(error) for error in errors}
        self._rejection_counts.update(categories)

    def _report_rejection_stats(self) -> None:
        stats = self.rejection_stats
        attempts = int(stats["attempts"])
        rejected = int(stats["rejected_attempts"])
        rate = float(stats["rejection_rate"])
        categories = stats["categories"]
        category_text = ""
        if isinstance(categories, dict) and categories:
            category_text = "; categories: " + ", ".join(
                f"{name}={count}" for name, count in categories.items()
            )
        self._progress(
            f"Generation quality: {attempts} model responses, {rejected} rejected "
            f"({rate:.1%} rejection rate){category_text}."
        )

    def _complete_with_retries(
        self,
        original_prompt: str,
        parser: Callable[[str], Parsed],
        *,
        max_tokens: int,
        label: str,
        response_schema: Mapping[str, object] | None = None,
        include_rejected_candidate: bool = True,
        retry_prompt_builder: (
            Callable[[str, str | None, Sequence[str]], str] | None
        ) = None,
    ) -> Parsed:
        build_retry = retry_prompt_builder or build_retry_prompt
        prompt = original_prompt
        last_errors: tuple[str, ...] = ()
        for attempt in range(1, self.config.max_retries + 1):
            self._attempt_count += 1
            try:
                candidate = self.backend.complete(
                    SYSTEM_PROMPT,
                    prompt,
                    max_tokens=max_tokens,
                    schema=response_schema,
                )
            except CompletionTruncatedError as exc:
                token_detail = ""
                if exc.prompt_tokens is not None and exc.completion_tokens is not None:
                    token_detail = (
                        f" ({exc.prompt_tokens} prompt tokens, "
                        f"{exc.completion_tokens} completion tokens)"
                    )
                last_errors = (
                    "response was truncated before completing the JSON" + token_detail,
                )
                self._record_rejection(last_errors)
                self._progress(
                    f"[{label}] attempt {attempt}/{self.config.max_retries} "
                    f"rejected: {last_errors[0]}"
                )
                prompt = build_retry(original_prompt, None, last_errors)
                if attempt < self.config.max_retries:
                    self._progress(
                        f"{label}: output length limit reached. Retrying with a "
                        f"compact regeneration request "
                        f"(attempt {attempt + 1}/{self.config.max_retries})."
                    )
                continue
            self._progress(
                f"[{label}] attempt {attempt}/{self.config.max_retries} "
                f"generated output:\n{candidate}"
            )
            try:
                return parser(candidate)
            except InsufficientContentError as exc:
                raise GenerationError(f"more content is required: {exc}") from exc
            except ValidationError as exc:
                last_errors = exc.errors
                self._record_rejection(last_errors)
                self._progress(
                    f"[{label}] attempt {attempt}/{self.config.max_retries} "
                    f"rejected: {' | '.join(last_errors)}"
                )
                rejected = candidate if include_rejected_candidate else None
                prompt = build_retry(original_prompt, rejected, last_errors)
                if attempt < self.config.max_retries:
                    summary = " | ".join(last_errors[:3])
                    self._progress(
                        f"{label}: {summary}. Retrying after validation errors "
                        f"(attempt {attempt + 1}/{self.config.max_retries})."
                    )

        details = "; ".join(last_errors) if last_errors else "unknown validation error"
        raise GenerationError(
            f"{label} failed after {self.config.max_retries} attempts: {details}"
        )

    @staticmethod
    def _duplicate_errors(
        cards: Sequence[FlashcardDraft],
        existing: Sequence[FlashcardCluster],
        prior_questions: Sequence[str] = (),
    ) -> tuple[str, ...]:
        errors: list[str] = []
        for left_index, left in enumerate(cards):
            for right_index, right in enumerate(
                cards[left_index + 1 :], start=left_index + 1
            ):
                if are_near_duplicates(left.question, right.question):
                    errors.append(
                        f"cards {left_index + 1} and {right_index + 1} are near duplicates"
                    )
                elif _polarity_variant(left.question, right.question):
                    errors.append(
                        f"cards {left_index + 1} and {right_index + 1} are mirrored polarity variants"
                    )
            # Do not reject a whole cluster merely because an adjacent concept
            # produced a similar question. Small local models commonly repeat
            # shared terminology across related concepts, and retrying the
            # cluster tends to reproduce the same wording until attempts are
            # exhausted. Cross-cluster overlap is handled by the module-level
            # review after all concepts have been generated. Keep the stricter
            # checks above for duplicates inside this cluster and below for
            # questions already used in an earlier module.
            for prior_question in prior_questions:
                if are_near_duplicates(left.question, prior_question):
                    errors.append(
                        f"card {left_index + 1} duplicates a question from a "
                        "previously generated module in this course"
                    )
        return tuple(errors)

    def _generate_cards(
        self,
        identity: ModuleIdentity,
        concept: ConceptPlan,
        existing: Sequence[FlashcardCluster],
        *,
        label: str,
        review_feedback: Sequence[str] = (),
        rejected_cards: Sequence[FlashcardDraft] = (),
        cluster_id: str | None = None,
        prior_questions: Sequence[str] = (),
        concept_facts: Sequence[GraphFact] = (),
        distractor_facts: Sequence[GraphFact] = (),
        module_facts: Sequence[GraphFact] = (),
    ) -> tuple[FlashcardDraft, ...]:
        prior_signals = tuple(
            _dedup_fingerprint(card) for cluster in existing for card in cluster.cards
        ) + tuple(_card_subject(question) for question in prior_questions)
        base_prompt = build_cluster_prompt(
            identity,
            concept,
            concept_facts,
            distractor_facts,
            prior_signals=prior_signals,
        )

        def build_cluster_retry(
            original_prompt: str,
            candidate: str | None,
            errors: Sequence[str],
        ) -> str:
            return build_cluster_retry_prompt(
                identity,
                concept,
                concept_facts,
                distractor_facts,
                prior_signals,
                candidate,
                errors,
            )

        if review_feedback:
            rejected = json.dumps(
                {"cards": [asdict(card) for card in rejected_cards]},
                ensure_ascii=False,
            )
            feedback = list(review_feedback)
            if cluster_id:
                feedback.insert(0, f"cluster UUID {cluster_id} requires regeneration")
            base_prompt = build_cluster_retry(base_prompt, rejected, feedback)

        def parse_and_validate(raw: str) -> tuple[FlashcardDraft, ...]:
            cards = parse_cards(raw)
            for card_position, card in enumerate(cards, start=1):
                self._progress(
                    f"[{label}] card {card_position}/{len(cards)} generated: "
                    + json.dumps(asdict(card), ensure_ascii=False)
                )
            grounding_facts = (
                tuple(module_facts)
                if module_facts
                else tuple(concept_facts) + tuple(distractor_facts)
            )
            errors = list(validate_cluster(cards, concept, grounding_facts))
            errors.extend(self._duplicate_errors(cards, existing, prior_questions))
            if errors:
                repaired = _repair_grounding_errors(
                    cards, errors, grounding_facts, phrase_facts=distractor_facts
                )
                if repaired is not None:
                    repair_errors = list(
                        validate_cluster(repaired, concept, grounding_facts)
                    )
                    repair_errors.extend(
                        self._duplicate_errors(repaired, existing, prior_questions)
                    )
                    if not repair_errors:
                        self._progress(
                            f"[{label}] auto-repaired {len(errors)} mechanical "
                            "validation issue(s) locally, "
                            "skipping LLM retry"
                        )
                        return repaired
                raise ValidationError(errors)
            return cards

        return self._complete_with_retries(
            base_prompt,
            parse_and_validate,
            max_tokens=self.config.cluster_max_tokens,
            label=label,
            response_schema=build_card_cluster_schema(concept.assessment_approaches),
            include_rejected_candidate=True,
            retry_prompt_builder=build_cluster_retry,
        )

    def _review(
        self,
        prompt: str,
        known_clusters: set[str],
        *,
        label: str,
        require_duplicate_locations: bool = False,
    ) -> tuple[ReviewIssue, ...]:
        def parse(raw: str) -> tuple[ReviewIssue, ...]:
            issues = parse_review_issues(raw, known_clusters)
            if not require_duplicate_locations:
                return issues

            errors: list[str] = []
            for issue in issues:
                for reason in issue.reasons:
                    match = _DUPLICATE_REVIEW_REASON_RE.match(reason)
                    if match is None:
                        errors.append(
                            "duplicate reason must use 'card <number> duplicates "
                            "cluster <uuid> card <number>'"
                        )
                        continue
                    source_card = int(match.group(1))
                    target_cluster = match.group(2)
                    target_card = int(match.group(3))
                    if not 1 <= source_card <= CARDS_PER_CLUSTER:
                        errors.append(
                            f"duplicate reason source card {source_card} is out of range"
                        )
                    if target_cluster not in known_clusters:
                        errors.append(
                            f"duplicate reason references unknown cluster {target_cluster!r}"
                        )
                    elif target_cluster == issue.cluster and target_card == source_card:
                        errors.append("duplicate reason cannot reference the same card")
                    if not 1 <= target_card <= CARDS_PER_CLUSTER:
                        errors.append(
                            f"duplicate reason target card {target_card} is out of range"
                        )
            if errors:
                raise ValidationError(errors)
            return issues

        return self._complete_with_retries(
            prompt,
            parse,
            max_tokens=self.config.review_max_tokens,
            label=label,
            response_schema=build_review_schema(tuple(known_clusters)),
        )

    def _collect_review_issues(
        self,
        clusters: Sequence[FlashcardCluster],
    ) -> tuple[
        dict[str, list[str]],
        frozenset[str],
        tuple[ReviewIssue, ...],
    ]:
        merged: dict[str, list[str]] = {}
        grounding_clusters: set[str] = set()
        review_group_size = 4
        review_group_count = (
            CLUSTERS_PER_MODULE + review_group_size - 1
        ) // review_group_size
        for group_number, start in enumerate(
            range(0, CLUSTERS_PER_MODULE, review_group_size),
            start=1,
        ):
            self._progress(f"Grounding review {group_number}/{review_group_count}...")
            group = tuple(clusters[start : start + review_group_size])
            issues = self._review(
                build_grounding_review_prompt(group),
                {cluster.cluster for cluster in group},
                label=f"grounding review {group_number}",
            )
            self._merge_issues(merged, issues)
            grounding_clusters.update(issue.cluster for issue in issues)

        self._progress("Global duplicate review...")
        global_issues = self._review(
            build_duplicate_review_prompt(clusters),
            {cluster.cluster for cluster in clusters},
            label="global duplicate review",
            require_duplicate_locations=True,
        )
        self._merge_issues(merged, global_issues)
        return (
            merged,
            frozenset(grounding_clusters),
            global_issues,
        )

    @staticmethod
    def _merge_issues(
        target: dict[str, list[str]],
        issues: Sequence[ReviewIssue],
    ) -> None:
        for issue in issues:
            reasons = target.setdefault(issue.cluster, [])
            for reason in issue.reasons:
                if reason not in reasons:
                    reasons.append(reason)

    @staticmethod
    def _build_duplicate_repair_stack(
        clusters: Sequence[FlashcardCluster],
        *,
        review_issues: Sequence[ReviewIssue] = (),
        module_errors: Sequence[str] = (),
    ) -> list[_DuplicateRepairItem]:
        """Collect duplicate pairs into one LIFO item per card location."""

        cluster_by_id = {
            cluster.cluster: index for index, cluster in enumerate(clusters)
        }
        conflicts_by_path: dict[
            tuple[int, int],
            list[tuple[int, int]],
        ] = {}
        reasons_by_path: dict[tuple[int, int], list[str]] = {}

        def add_location(
            repair_path: tuple[int, int],
            conflict_path: tuple[int, int],
            reason: str,
        ) -> None:
            conflicts = conflicts_by_path.setdefault(repair_path, [])
            reasons = reasons_by_path.setdefault(repair_path, [])
            if conflict_path not in conflicts:
                conflicts.append(conflict_path)
            if reason not in reasons:
                reasons.append(reason)

        for issue in review_issues:
            source_cluster = cluster_by_id[issue.cluster]
            for reason in issue.reasons:
                match = _DUPLICATE_REVIEW_REASON_RE.match(reason)
                if match is None:
                    raise GenerationError(
                        f"global duplicate review omitted an exact card location: {reason}"
                    )
                source = (source_cluster, int(match.group(1)) - 1)
                target = (
                    cluster_by_id[match.group(2)],
                    int(match.group(3)) - 1,
                )
                add_location(source, target, reason)

        for error in module_errors:
            match = _MODULE_DUPLICATE_RE.match(error)
            if match is None:
                continue
            first = (int(match.group(1)) - 1, int(match.group(2)) - 1)
            second = (int(match.group(3)) - 1, int(match.group(4)) - 1)
            add_location(max(first, second), min(first, second), error)

        return [
            _DuplicateRepairItem(
                cluster_index=cluster_index,
                card_index=card_index,
                original_card=clusters[cluster_index].cards[card_index],
                conflicts=tuple(conflicts_by_path[(cluster_index, card_index)]),
                reasons=tuple(reasons_by_path[(cluster_index, card_index)]),
            )
            for cluster_index, card_index in sorted(conflicts_by_path)
        ]

    def _repair_duplicate_stack(
        self,
        identity: ModuleIdentity,
        clusters: list[FlashcardCluster],
        stack: list[_DuplicateRepairItem],
        prior_questions: Sequence[str],
    ) -> frozenset[str]:
        """Pop flagged cards, paraphrase one, and replace its exact JSON slot."""

        repaired_cluster_ids: set[str] = set()
        while stack:
            item = stack.pop()
            old = clusters[item.cluster_index]
            original_card = item.original_card
            conflicting_questions = [
                {
                    "cluster": cluster_index + 1,
                    "card": card_index + 1,
                    "question": clusters[cluster_index].cards[card_index].question,
                }
                for cluster_index, card_index in item.conflicts
            ]
            self._progress(
                "Paraphrasing duplicate card at cluster "
                f"{item.cluster_index + 1}/{CLUSTERS_PER_MODULE} card "
                f"{item.card_index + 1}: {old.concept.name}"
            )
            prompt = build_duplicate_card_repair_prompt(
                identity,
                old.concept,
                original_card,
                cluster_number=item.cluster_index + 1,
                card_number=item.card_index + 1,
                cluster_id=old.cluster,
                reasons=item.reasons,
                conflicting_questions=conflicting_questions,
            )

            def parse_and_validate(raw: str) -> FlashcardDraft:
                cards = parse_cards(raw)
                if len(cards) != 1:
                    raise ValidationError(
                        f"expected exactly 1 replacement card, received {len(cards)}"
                    )
                candidate = cards[0]
                errors: list[str] = []
                for field in (
                    "type",
                    "is_true",
                    "difficulty",
                    "assessment_approach",
                ):
                    if getattr(candidate, field) != getattr(original_card, field):
                        errors.append(f"replacement card must preserve {field}")
                locked_answer_fields: tuple[str, ...] = ()
                if original_card.type == "true-false":
                    locked_answer_fields = (
                        "correct_option",
                        "wrong_option_1",
                        "wrong_option_2",
                        "wrong_option_3",
                    )
                elif original_card.type == "identification":
                    locked_answer_fields = (
                        "wrong_option_1",
                        "wrong_option_2",
                        "wrong_option_3",
                    )
                for field in locked_answer_fields:
                    if getattr(candidate, field) != getattr(original_card, field):
                        errors.append(f"replacement card must preserve {field}")
                candidate = replace(
                    candidate,
                    expalanation=original_card.expalanation,
                    hint=original_card.hint,
                )
                if are_near_duplicates(candidate.question, original_card.question):
                    errors.append(
                        "replacement question is still a near-duplicate of the "
                        "original flagged question"
                    )

                replacement_cards = list(old.cards)
                replacement_cards[item.card_index] = candidate
                errors.extend(
                    validate_cluster(tuple(replacement_cards), old.concept, ())
                )

                for cluster_index, cluster in enumerate(clusters):
                    for card_index, card in enumerate(cluster.cards):
                        if (
                            cluster_index == item.cluster_index
                            and card_index == item.card_index
                        ):
                            continue
                        if are_near_duplicates(candidate.question, card.question):
                            errors.append(
                                "replacement question near-duplicates cluster "
                                f"{cluster_index + 1} card {card_index + 1}: "
                                f"{card.question!r}"
                            )
                for prior_question in prior_questions:
                    if are_near_duplicates(candidate.question, prior_question):
                        errors.append(
                            "replacement question duplicates a previously generated "
                            f"module question: {prior_question!r}"
                        )
                if errors:
                    raise ValidationError(errors)
                return candidate

            repaired_card = self._complete_with_retries(
                prompt,
                parse_and_validate,
                max_tokens=self.config.cluster_max_tokens,
                label=(
                    f"duplicate repair {item.cluster_index + 1} card "
                    f"{item.card_index + 1} ({old.concept.name})"
                ),
                response_schema=build_single_card_schema(original_card),
                retry_prompt_builder=lambda original, rejected, errors: (
                    build_duplicate_card_retry_prompt(
                        original,
                        rejected,
                        errors,
                        original_card,
                    )
                ),
            )
            updated_cards = list(old.cards)
            updated_cards[item.card_index] = repaired_card
            clusters[item.cluster_index] = replace(old, cards=tuple(updated_cards))
            repaired_cluster_ids.add(old.cluster)
        return frozenset(repaired_cluster_ids)

    def run(
        self,
        identity: ModuleIdentity,
        facts: Sequence[GraphFact],
        *,
        prior_concept_names: Sequence[str] = (),
        prior_questions: Sequence[str] = (),
    ) -> tuple[FlashcardCluster, ...]:
        self._attempt_count = 0
        self._rejected_attempt_count = 0
        self._rejection_counts.clear()
        plan_facts = _balanced_plan_facts(_exclude_structural_facts(facts))
        self._progress(
            f"Planning {CONCEPTS_PER_MODULE} concepts from "
            f"{len(plan_facts)} grounded lesson facts..."
        )
        plan_prompt = build_concept_plan_prompt(
            identity,
            plan_facts,
            prior_concept_names,
        )

        def build_concept_plan_retry(
            original_prompt: str,
            candidate: str | None,
            errors: Sequence[str],
        ) -> str:
            del original_prompt  # rebuilt fresh from identity/plan_facts below
            return build_concept_plan_retry_prompt(
                identity,
                plan_facts,
                prior_concept_names,
                candidate,
                errors,
            )

        concepts = self._complete_with_retries(
            plan_prompt,
            lambda raw: parse_concept_plan(raw, plan_facts),
            max_tokens=self.config.plan_max_tokens,
            label="concept plan",
            response_schema=build_concept_plan_schema(
                tuple(fact.fact_id for fact in plan_facts)
            ),
            include_rejected_candidate=False,
            retry_prompt_builder=build_concept_plan_retry,
        )

        clusters: list[FlashcardCluster] = []
        for position, concept in enumerate(concepts, start=1):
            self._progress(
                f"Generating cluster {position}/{CLUSTERS_PER_MODULE}: {concept.name}"
            )
            concept_fact_ids = set(concept.fact_ids)

            concept_facts = tuple(
                fact for fact in facts if fact.fact_id in concept_fact_ids
            )
            distractor_facts = _select_distractor_facts(facts, concept)
            cards = self._generate_cards(
                identity,
                concept,
                clusters,
                label=f"concept {position} ({concept.name})",
                prior_questions=prior_questions,
                concept_facts=concept_facts,
                distractor_facts=distractor_facts,
                module_facts=facts,
            )
            clusters.append(
                FlashcardCluster(
                    cluster=str(uuid4()),
                    concept=concept,
                    cards=cards,
                )
            )

        # Cross-cluster similarity belongs to the global review below. Running
        # it here would abort after generation but before the reviewer could
        # identify and regenerate the overlapping cluster.
        initial_errors = validate_module(
            clusters,
            facts,
            check_question_duplicates=False,
        )
        if initial_errors:
            raise GenerationError(
                "initial module validation failed: " + "; ".join(initial_errors)
            )

        relaxed_grounding_cluster_ids: set[str] = set()
        duplicate_repairs = 0
        if self.config.final_review:
            issues, grounding_clusters, duplicate_issues = self._collect_review_issues(
                clusters
            )
            duplicate_cluster_ids = {issue.cluster for issue in duplicate_issues}
            if issues:
                by_id = {
                    cluster.cluster: index for index, cluster in enumerate(clusters)
                }
                for cluster_id, reasons in issues.items():
                    if (
                        cluster_id in duplicate_cluster_ids
                        and cluster_id not in grounding_clusters
                    ):
                        continue
                    index = by_id[cluster_id]
                    old = clusters[index]
                    self._progress(
                        "Regenerating reviewed cluster "
                        f"{index + 1}/{CLUSTERS_PER_MODULE}: {old.concept.name}"
                    )
                    others = tuple(
                        cluster
                        for other_index, cluster in enumerate(clusters)
                        if other_index != index
                    )
                    review_fact_ids = set(old.concept.fact_ids)
                    review_concept_facts = tuple(
                        fact for fact in facts if fact.fact_id in review_fact_ids
                    )
                    review_distractor_facts = _select_distractor_facts(
                        facts, old.concept
                    )
                    cards = self._generate_cards(
                        identity,
                        old.concept,
                        others,
                        label=f"reviewed concept {index + 1} ({old.concept.name})",
                        review_feedback=reasons,
                        rejected_cards=old.cards,
                        cluster_id=old.cluster,
                        prior_questions=prior_questions,
                        concept_facts=review_concept_facts,
                        distractor_facts=review_distractor_facts,
                        module_facts=facts,
                    )
                    clusters[index] = replace(old, cards=cards)

            duplicate_only_issues = tuple(
                issue
                for issue in duplicate_issues
                if issue.cluster not in grounding_clusters
            )
            review_stack = self._build_duplicate_repair_stack(
                clusters,
                review_issues=duplicate_only_issues,
            )
            duplicate_repairs += len(review_stack)
            if review_stack:
                relaxed_grounding_cluster_ids.update(
                    self._repair_duplicate_stack(
                        identity,
                        clusters,
                        review_stack,
                        prior_questions,
                    )
                )

        final_errors = validate_module(
            clusters,
            facts,
            skip_grounding_cluster_ids=frozenset(relaxed_grounding_cluster_ids),
        )
        while True:
            duplicate_errors = tuple(
                error for error in final_errors if _MODULE_DUPLICATE_RE.match(error)
            )
            if not duplicate_errors:
                break
            repair_stack = self._build_duplicate_repair_stack(
                clusters,
                module_errors=duplicate_errors,
            )
            duplicate_repairs += len(repair_stack)
            if duplicate_repairs > CARDS_PER_MODULE:
                raise GenerationError(
                    "duplicate repair limit reached before the module became unique"
                )
            relaxed_grounding_cluster_ids.update(
                self._repair_duplicate_stack(
                    identity,
                    clusters,
                    repair_stack,
                    prior_questions,
                )
            )
            final_errors = validate_module(
                clusters,
                facts,
                skip_grounding_cluster_ids=frozenset(relaxed_grounding_cluster_ids),
            )
        if final_errors:
            raise GenerationError(
                "final module validation failed: " + "; ".join(final_errors)
            )
        self._report_rejection_stats()
        self._progress(
            f"{CARDS_PER_MODULE} flashcards generated "
            f"({CARDS_PER_BLOCK} + {CARDS_PER_BLOCK})."
        )
        return tuple(clusters)
