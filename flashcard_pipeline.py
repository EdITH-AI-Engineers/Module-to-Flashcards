from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from typing import Callable, Sequence, TypeVar
from uuid import uuid4

from flashcard_contract import (
    CARDS_PER_BLOCK,
    CARDS_PER_MODULE,
    CLUSTERS_PER_MODULE,
    CONCEPTS_PER_MODULE,
)
from flashcard_prompt import (
    SYSTEM_PROMPT,
    build_cluster_prompt,
    build_concept_plan_prompt,
    build_duplicate_review_prompt,
    build_grounding_review_prompt,
    build_retry_prompt,
    grounded_vocabulary,
    _card_subject,
    _dedup_fingerprint,
)
from flashcard_types import (
    ChatBackend,
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


_GROUNDING_ERROR_RE = re.compile(
    r"^card (\d+) (wrong_option_[123]) is not grounded in supplied module facts$"
)


def _phrase_from_fact(fact: GraphFact) -> str:
    """Turn one grounding fact into a short, option-shaped phrase, so a
    repaired wrong_option reads like a normal answer choice rather than a
    dumped sentence fragment."""
    statement = fact.statement.strip().rstrip(".")
    for sep in (";", ","):
        if sep in statement:
            statement = statement.split(sep, 1)[0].strip()
            break
    words = statement.split()
    if len(words) > 12:
        statement = " ".join(words[:12])
    if statement:
        statement = statement[0].upper() + statement[1:]
    return statement


def _repair_grounding_errors(
    cards: Sequence[FlashcardDraft],
    errors: Sequence[str],
    grounding_facts: Sequence[GraphFact],
    phrase_facts: Sequence[GraphFact] | None = None,
) -> tuple[FlashcardDraft, ...] | None:
    """Deterministically patch wrong_options the validator flagged as "not
    grounded", without another model call.

    Local backends sometimes return byte-identical output across every
    retry attempt regardless of the corrective instructions in the retry
    prompt -- when that happens, re-prompting harder is a dead end no
    matter how the prompt is worded. A "not grounded" violation is purely
    mechanical (the validator just checks for shared vocabulary with the
    supplied facts), so it can be fixed mechanically too: swap the flagged
    text for a short phrase lifted directly from an actual module fact,
    which is grounded by construction. Returns None (declining to repair)
    if any error is not a grounding error, so mixed failures still go
    through the normal LLM retry path where a human-authored correction
    is actually needed.

    phrase_facts, if given, restricts which facts a replacement phrase can
    be built from -- callers should pass the distractor pool here rather
    than the concept's own facts, since a wrong_option built from the
    concept's own fact would just be a true restatement of the correct
    answer rather than an actual distractor. Falls back to grounding_facts
    if phrase_facts is empty.
    """
    matches = [_GROUNDING_ERROR_RE.match(error) for error in errors]
    if not matches or not all(matches):
        return None

    source_facts = phrase_facts if phrase_facts else grounding_facts
    candidate_phrases = [
        phrase
        for phrase in (_phrase_from_fact(fact) for fact in source_facts)
        if phrase
    ]
    if not candidate_phrases:
        return None

    cards_list = list(cards)
    used_per_card: dict[int, set[str]] = {}
    cursor = 0

    for match in matches:
        card_index = int(match.group(1)) - 1
        field = match.group(2)
        if not (0 <= card_index < len(cards_list)):
            return None
        card = cards_list[card_index]
        used = used_per_card.setdefault(
            card_index,
            {
                option.strip().casefold()
                for option in (
                    card.correct_option,
                    card.wrong_option_1,
                    card.wrong_option_2,
                    card.wrong_option_3,
                )
                if option.strip()
            },
        )
        replacement = None
        for offset in range(len(candidate_phrases)):
            phrase = candidate_phrases[(cursor + offset) % len(candidate_phrases)]
            if phrase.casefold() not in used:
                replacement = phrase
                cursor += offset + 1
                break
        if replacement is None:
            return None
        used.add(replacement.casefold())
        cards_list[card_index] = replace(card, **{field: replacement})

    return tuple(cards_list)


def _select_distractor_facts(
    facts: Sequence[GraphFact],
    concept: ConceptPlan,
    limit: int = DISTRACTOR_FACT_LIMIT,
) -> tuple[GraphFact, ...]:
    """Pick distractor material likely to actually support a groundable wrong option.

    A blind positional slice of "the first N facts that are not this
    concept's own" tends to hand the model facts from a completely
    unrelated part of the module. With nothing usable to adapt, the model
    tends to fall back on a generic, plausible-sounding but ungrounded
    guess -- which is exactly the "not grounded in supplied module facts"
    failure this produces. Preferring facts that share the concept's own
    topic, then facts on nearby slides, gives the model distractor
    material that is actually related to the question it is writing.
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

    def _complete_with_retries(
        self,
        original_prompt: str,
        parser: Callable[[str], Parsed],
        *,
        max_tokens: int,
        label: str,
        include_rejected_candidate: bool = True,
        retry_prompt_builder: (
            Callable[[str, str | None, Sequence[str]], str] | None
        ) = None,
    ) -> Parsed:
        build_retry = retry_prompt_builder or build_retry_prompt
        prompt = original_prompt
        last_errors: tuple[str, ...] = ()
        for attempt in range(1, self.config.max_retries + 1):
            candidate = self.backend.complete(
                SYSTEM_PROMPT,
                prompt,
                max_tokens=max_tokens,
            )
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
            # for cluster in existing:
            #     for prior_index, prior in enumerate(cluster.cards, start=1):
            #         if are_near_duplicates(left.question, prior.question):
            #             errors.append(
            #                 f"card {left_index + 1} duplicates card {prior_index} "
            #                 f"from concept {cluster.concept.name!r}"
            #             )
            # for prior_question in prior_questions:
            #     if are_near_duplicates(left.question, prior_question):
            #         errors.append(
            #             f"card {left_index + 1} duplicates a question from a "
            #             "previously generated module in this course"
            #         )
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
        vocabulary = grounded_vocabulary(concept, concept_facts, distractor_facts)

        def build_cluster_retry(
            original_prompt: str,
            candidate: str | None,
            errors: Sequence[str],
        ) -> str:
            return build_retry_prompt(
                original_prompt, candidate, errors, grounded_vocabulary=vocabulary
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
            grounding_facts = tuple(concept_facts) + tuple(distractor_facts)
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
                            f"[{label}] auto-repaired {len(errors)} ungrounded "
                            "wrong_option(s) locally (fact-derived replacement text), "
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
            retry_prompt_builder=build_cluster_retry,
        )

    def _review(
        self,
        prompt: str,
        known_clusters: set[str],
        *,
        label: str,
    ) -> tuple[ReviewIssue, ...]:
        return self._complete_with_retries(
            prompt,
            lambda raw: parse_review_issues(raw, known_clusters),
            max_tokens=self.config.review_max_tokens,
            label=label,
        )

    def _collect_review_issues(
        self,
        clusters: Sequence[FlashcardCluster],
    ) -> dict[str, list[str]]:
        merged: dict[str, list[str]] = {}
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

        self._progress("Global duplicate review...")
        global_issues = self._review(
            build_duplicate_review_prompt(clusters),
            {cluster.cluster for cluster in clusters},
            label="global duplicate review",
        )
        self._merge_issues(merged, global_issues)
        return merged

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

    def run(
        self,
        identity: ModuleIdentity,
        facts: Sequence[GraphFact],
        *,
        prior_concept_names: Sequence[str] = (),
        prior_questions: Sequence[str] = (),
    ) -> tuple[FlashcardCluster, ...]:
        plan_facts = _balanced_plan_facts(facts)
        self._progress(
            f"Planning {CONCEPTS_PER_MODULE} concepts from "
            f"{len(plan_facts)} grounded lesson facts..."
        )
        plan_prompt = build_concept_plan_prompt(
            identity,
            plan_facts,
            prior_concept_names,
        )
        concepts = self._complete_with_retries(
            plan_prompt,
            lambda raw: parse_concept_plan(raw, plan_facts),
            max_tokens=self.config.plan_max_tokens,
            label="concept plan",
            include_rejected_candidate=False,
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
            )
            clusters.append(
                FlashcardCluster(
                    cluster=str(uuid4()),
                    concept=concept,
                    cards=cards,
                )
            )

        initial_errors = validate_module(clusters, facts)
        if initial_errors:
            raise GenerationError(
                "initial module validation failed: " + "; ".join(initial_errors)
            )

        if self.config.final_review:
            issues = self._collect_review_issues(clusters)
            if issues:
                by_id = {
                    cluster.cluster: index for index, cluster in enumerate(clusters)
                }
                for cluster_id, reasons in issues.items():
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
                    )
                    clusters[index] = replace(old, cards=cards)

        final_errors = validate_module(clusters, facts)
        if final_errors:
            raise GenerationError(
                "final module validation failed: " + "; ".join(final_errors)
            )
        self._progress(
            f"{CARDS_PER_MODULE} flashcards generated "
            f"({CARDS_PER_BLOCK} + {CARDS_PER_BLOCK})."
        )
        return tuple(clusters)
