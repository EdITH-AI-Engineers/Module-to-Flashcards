from __future__ import annotations

import json
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
        self._progress = progress or (lambda _message: None)

    def _complete_with_retries(
        self,
        original_prompt: str,
        parser: Callable[[str], Parsed],
        *,
        max_tokens: int,
        label: str,
        include_rejected_candidate: bool = True,
    ) -> Parsed:
        prompt = original_prompt
        last_errors: tuple[str, ...] = ()
        for attempt in range(1, self.config.max_retries + 1):
            candidate = self.backend.complete(
                SYSTEM_PROMPT,
                prompt,
                max_tokens=max_tokens,
            )
            try:
                return parser(candidate)
            except InsufficientContentError as exc:
                raise GenerationError(f"more content is required: {exc}") from exc
            except ValidationError as exc:
                last_errors = exc.errors
                rejected = candidate if include_rejected_candidate else None
                prompt = build_retry_prompt(original_prompt, rejected, last_errors)
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
    ) -> tuple[str, ...]:
        errors: list[str] = []
        for left_index, left in enumerate(cards):
            for right_index, right in enumerate(cards[left_index + 1 :], start=left_index + 1):
                if are_near_duplicates(left.question, right.question):
                    errors.append(
                        f"cards {left_index + 1} and {right_index + 1} are near duplicates"
                    )
            for cluster in existing:
                for prior_index, prior in enumerate(cluster.cards, start=1):
                    if are_near_duplicates(left.question, prior.question):
                        errors.append(
                            f"card {left_index + 1} duplicates card {prior_index} "
                            f"from concept {cluster.concept.name!r}"
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
    ) -> tuple[FlashcardDraft, ...]:
        base_prompt = build_cluster_prompt(identity, concept)
        if review_feedback:
            rejected = json.dumps(
                {"cards": [asdict(card) for card in rejected_cards]},
                ensure_ascii=False,
            )
            feedback = list(review_feedback)
            if cluster_id:
                feedback.insert(0, f"cluster UUID {cluster_id} requires regeneration")
            base_prompt = build_retry_prompt(base_prompt, rejected, feedback)

        def parse_and_validate(raw: str) -> tuple[FlashcardDraft, ...]:
            cards = parse_cards(raw)
            errors = list(validate_cluster(cards, concept))
            errors.extend(self._duplicate_errors(cards, existing))
            if errors:
                raise ValidationError(errors)
            return cards

        return self._complete_with_retries(
            base_prompt,
            parse_and_validate,
            max_tokens=self.config.cluster_max_tokens,
            label=label,
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
    ) -> tuple[FlashcardCluster, ...]:
        self._progress(f"Planning {CONCEPTS_PER_MODULE} concepts...")
        plan_prompt = build_concept_plan_prompt(identity, facts)
        concepts = self._complete_with_retries(
            plan_prompt,
            lambda raw: parse_concept_plan(raw, facts),
            max_tokens=self.config.plan_max_tokens,
            label="concept plan",
            include_rejected_candidate=False,
        )

        clusters: list[FlashcardCluster] = []
        for position, concept in enumerate(concepts, start=1):
            self._progress(
                f"Generating cluster {position}/{CLUSTERS_PER_MODULE}: {concept.name}"
            )
            cards = self._generate_cards(
                identity,
                concept,
                clusters,
                label=f"concept {position} ({concept.name})",
            )
            clusters.append(
                FlashcardCluster(
                    cluster=str(uuid4()),
                    concept=concept,
                    cards=cards,
                )
            )

        initial_errors = validate_module(clusters)
        if initial_errors:
            raise GenerationError(
                "initial module validation failed: " + "; ".join(initial_errors)
            )

        if self.config.final_review:
            issues = self._collect_review_issues(clusters)
            if issues:
                by_id = {cluster.cluster: index for index, cluster in enumerate(clusters)}
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
                    cards = self._generate_cards(
                        identity,
                        old.concept,
                        others,
                        label=f"reviewed concept {index + 1} ({old.concept.name})",
                        review_feedback=reasons,
                        rejected_cards=old.cards,
                        cluster_id=old.cluster,
                    )
                    clusters[index] = replace(old, cards=cards)

        final_errors = validate_module(clusters)
        if final_errors:
            raise GenerationError(
                "final module validation failed: " + "; ".join(final_errors)
            )
        self._progress(
            f"{CARDS_PER_MODULE} flashcards generated "
            f"({CARDS_PER_BLOCK} + {CARDS_PER_BLOCK})."
        )
        return tuple(clusters)
