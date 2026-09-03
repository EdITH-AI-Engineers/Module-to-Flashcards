from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ModuleIdentity:
    course_code: str
    module_number: str


@dataclass(frozen=True)
class GraphFact:
    fact_id: str
    statement: str


@dataclass(frozen=True)
class ConceptPlan:
    name: str
    fact_ids: tuple[str, ...]
    facts: tuple[str, ...]
    assessment_approaches: tuple[str, ...]


@dataclass(frozen=True)
class FlashcardDraft:
    type: str
    question: str
    correct_option: str
    wrong_option_1: str
    wrong_option_2: str
    wrong_option_3: str
    is_true: int | None
    expalanation: str
    hint: str
    difficulty: int
    assessment_approach: str


@dataclass(frozen=True)
class FlashcardCluster:
    cluster: str
    concept: ConceptPlan
    cards: tuple[FlashcardDraft, ...]


@dataclass(frozen=True)
class ReviewIssue:
    cluster: str
    reasons: tuple[str, ...]


class ChatBackend(Protocol):
    def complete(self, system: str, user: str, *, max_tokens: int) -> str:
        """Return only the assistant's textual content."""

