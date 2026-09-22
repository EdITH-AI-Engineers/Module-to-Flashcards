from __future__ import annotations

from dataclasses import dataclass
import difflib
import re
from typing import Iterable, Mapping, Sequence
import unicodedata


FORMAT_VERSION = "1"
NOT_SPECIFIED = "Not Specified"
UNREADABLE = "[Unreadable Text]"

_CONTROL_TAG = re.compile(
    r"^\s*\[/?(?:MODULE|SLIDE(?:\s+\d+)?|TITLE|CONTENT|VISUAL_TEXT|"
    r"DEFINITIONS|KNOWLEDGE_STATEMENTS|BRIEF_EXPLANATION)\]\s*$",
    flags=re.IGNORECASE,
)
_SECTION_TAG = re.compile(r"^\[(/?)([A-Z_]+)(?:\s+\d+)?\]$")
_SLIDE_OPEN_TAG = re.compile(r"^\[SLIDE\s+(\d+)\]$", flags=re.IGNORECASE)
_PRESENTATION_NOISE = re.compile(
    r"^(?:(?:this|the|the first|the current|first)\s+(?:slide|page)\b|"
    r"the module title\b|(?:this|the)\s+module\s+"
    r"(?:introduces|covers|presents|provides an overview of)\b)",
    flags=re.IGNORECASE,
)

_FACT_BRIDGE_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "been",
    "being",
    "described",
    "describes",
    "in",
    "include",
    "includes",
    "is",
    "its",
    "of",
    "refer",
    "refers",
    "that",
    "the",
    "through",
    "to",
    "was",
    "were",
    "which",
    "within",
    "with",
}
_FACT_NEGATIONS = {"except", "never", "no", "not", "without"}


def _normalized_fact_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _fact_content_tokens(value: object) -> tuple[str, ...]:
    return tuple(
        token
        for token in _normalized_fact_text(value).split()
        if token not in _FACT_BRIDGE_WORDS
    )


def _looks_like_enumeration(value: object) -> bool:
    text = str(value or "")
    separators = len(re.findall(r"[,;]", text))
    numbered_items = len(re.findall(r"(?:^|\s)\d+[.)]\s", text))
    return separators >= 2 or numbered_items >= 3


def _facts_are_duplicates(left: object, right: object) -> bool:
    """Match reworded facts and cumulative list fragments conservatively."""

    normalized_left = _normalized_fact_text(left)
    normalized_right = _normalized_fact_text(right)
    if not normalized_left or not normalized_right:
        return False
    if normalized_left == normalized_right:
        return True

    left_tokens = _fact_content_tokens(left)
    right_tokens = _fact_content_tokens(right)
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    if not left_set or not right_set:
        return False
    if (left_set & _FACT_NEGATIONS) != (right_set & _FACT_NEGATIONS):
        return False

    shared = left_set & right_set
    overlap = len(shared) / min(len(left_set), len(right_set))
    jaccard = len(shared) / len(left_set | right_set)
    sequence = difflib.SequenceMatcher(
        None, normalized_left, normalized_right
    ).ratio()

    if (
        _looks_like_enumeration(left)
        and _looks_like_enumeration(right)
        and min(len(left_tokens), len(right_tokens)) >= 10
    ):
        return overlap >= 0.80 and jaccard >= 0.62 and sequence >= 0.62

    return (
        min(len(left_tokens), len(right_tokens)) >= 8
        and overlap >= 0.95
        and jaccard >= 0.85
        and sequence >= 0.90
    )


def deduplicate_lesson_fact_records(
    records: Sequence[Mapping[str, object]],
    *,
    reassign_ids: bool = False,
) -> tuple[dict[str, object], ...]:
    """Collapse repeated facts while retaining the best text and all slides.

    The longest cumulative enumeration wins. For ordinary paraphrases, the
    first statement remains canonical so stable source order is preserved.
    """

    merged: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        statement = re.sub(
            r"\s+", " ", str(record.get("statement", ""))
        ).strip()
        if not statement:
            continue
        candidate = dict(record)
        candidate["statement"] = statement
        raw_slides = candidate.get("slides")
        candidate_slides: list[int] = []
        if isinstance(raw_slides, (list, tuple)):
            for value in raw_slides:
                try:
                    number = int(value)
                except (TypeError, ValueError):
                    continue
                if number > 0 and number not in candidate_slides:
                    candidate_slides.append(number)
        candidate["slides"] = candidate_slides

        matches = [
            index
            for index, existing in enumerate(merged)
            if _facts_are_duplicates(existing["statement"], statement)
        ]
        if not matches:
            merged.append(candidate)
            continue

        primary = matches[0]
        duplicate_group = [merged[index] for index in matches]
        all_slides = set(candidate_slides)
        for existing in duplicate_group:
            existing_slides = existing["slides"]
            assert isinstance(existing_slides, list)
            all_slides.update(existing_slides)

        chosen = duplicate_group[0]
        if _looks_like_enumeration(statement):
            chosen = max(
                (*duplicate_group, candidate),
                key=lambda item: len(_fact_content_tokens(item["statement"])),
            )
        chosen["slides"] = sorted(all_slides)
        merged[primary] = chosen
        for index in reversed(matches[1:]):
            del merged[index]

    if reassign_ids:
        for index, fact in enumerate(merged, start=1):
            fact["id"] = f"f{index}"
    return tuple(merged)


def _single_line(value: object, fallback: str = NOT_SPECIFIED) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text or fallback


def _safe_item(value: object) -> str:
    text = _single_line(value)
    if _CONTROL_TAG.fullmatch(text):
        return text.replace("[", "［").replace("]", "］")
    return text


def _items(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(_safe_item(value) for value in values if str(value or "").strip())


def _meaningful(value: str) -> bool:
    return value.strip().casefold() not in {
        "",
        NOT_SPECIFIED.casefold(),
        UNREADABLE.casefold(),
    }


@dataclass(frozen=True)
class Definition:
    term: str
    definition: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "term", _safe_item(self.term))
        object.__setattr__(self, "definition", _safe_item(self.definition))


@dataclass(frozen=True)
class StructuredSlide:
    number: int
    extraction_method: str
    title: str
    content: tuple[str, ...]
    visual_text: tuple[str, ...]
    definitions: tuple[Definition, ...]
    knowledge_statements: tuple[str, ...]
    brief_explanation: str

    def __post_init__(self) -> None:
        if self.number < 1:
            raise ValueError("slide number must be positive")
        method = _single_line(self.extraction_method).casefold()
        if method not in {"text", "ocr"}:
            raise ValueError("extraction method must be text or ocr")
        object.__setattr__(self, "extraction_method", method)
        object.__setattr__(self, "title", _safe_item(self.title))
        object.__setattr__(self, "content", _items(self.content) or (NOT_SPECIFIED,))
        object.__setattr__(
            self,
            "visual_text",
            _items(self.visual_text) or (NOT_SPECIFIED,),
        )
        object.__setattr__(self, "definitions", tuple(self.definitions))
        object.__setattr__(
            self,
            "knowledge_statements",
            _items(self.knowledge_statements),
        )
        object.__setattr__(self, "brief_explanation", _safe_item(self.brief_explanation))

    def has_readable_content(self) -> bool:
        values = (
            self.title,
            *self.content,
            *self.visual_text,
            *(item.term for item in self.definitions),
            *(item.definition for item in self.definitions),
            *self.knowledge_statements,
            self.brief_explanation,
        )
        return any(_meaningful(value) for value in values)


@dataclass(frozen=True)
class StructuredModule:
    course_code: str
    module_number: str
    module_title: str
    source_file: str
    slides: tuple[StructuredSlide, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "course_code", _safe_item(self.course_code))
        object.__setattr__(self, "module_number", _safe_item(self.module_number))
        object.__setattr__(self, "module_title", _safe_item(self.module_title))
        object.__setattr__(self, "source_file", _safe_item(self.source_file))
        object.__setattr__(self, "slides", tuple(self.slides))
        if not self.slides:
            raise ValueError("structured module must contain at least one slide")
        if not any(slide.has_readable_content() for slide in self.slides):
            raise ValueError("structured module contains no readable learning content")


def _section(name: str, values: Iterable[str]) -> list[str]:
    return [f"[{name}]", *values, f"[/{name}]"]


def render_structured_module(module: StructuredModule) -> str:
    lines = [
        "[MODULE]",
        f"format_version: {FORMAT_VERSION}",
        f"course_code: {module.course_code}",
        f"module_number: {module.module_number}",
        f"module_title: {module.module_title}",
        f"source_file: {module.source_file}",
        "[/MODULE]",
    ]

    for slide in module.slides:
        definitions = [
            f"- {item.term} :: {item.definition}" for item in slide.definitions
        ] or [NOT_SPECIFIED]
        statements = [f"- {item}" for item in slide.knowledge_statements] or [
            NOT_SPECIFIED
        ]
        content = [f"- {item}" for item in slide.content]
        visual_text = [f"- {item}" for item in slide.visual_text]
        lines.extend(
            [
                "",
                f"[SLIDE {slide.number}]",
                f"extraction_method: {slide.extraction_method}",
                *_section("TITLE", [slide.title]),
                *_section("CONTENT", content),
                *_section("VISUAL_TEXT", visual_text),
                *_section("DEFINITIONS", definitions),
                *_section("KNOWLEDGE_STATEMENTS", statements),
                *_section("BRIEF_EXPLANATION", [slide.brief_explanation]),
                "[/SLIDE]",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def parse_module_metadata(text: str) -> dict[str, str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.splitlines()
    try:
        start = lines.index("[MODULE]") + 1
        end = lines.index("[/MODULE]", start)
    except ValueError:
        return {}

    metadata: dict[str, str] = {}
    allowed = {
        "format_version",
        "course_code",
        "module_number",
        "module_title",
        "source_file",
    }
    for line in lines[start:end]:
        key, separator, value = line.partition(":")
        key = key.strip()
        if separator and key in allowed and value.strip():
            metadata[key] = value.strip()
    return metadata


def _lesson_fact_text(value: str) -> str | None:
    text = re.sub(r"\s+", " ", value).strip()
    if text.startswith("- "):
        text = text[2:].strip()
    if not _meaningful(text) or _PRESENTATION_NOISE.match(text):
        return None
    if len(text) < 12 or len(text.split()) < 3:
        return None
    return text


def extract_lesson_facts(text: str) -> tuple[dict[str, object], ...]:
    """Extract grounded lesson facts from normalized structured module text."""
    metadata = parse_module_metadata(text)
    if not metadata:
        return ()

    slides: list[dict[str, object]] = []
    current_slide: dict[str, object] | None = None
    current_section: str | None = None
    included_sections = {"TITLE", "CONTENT", "DEFINITIONS", "KNOWLEDGE_STATEMENTS"}

    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = raw_line.strip()
        slide_tag = _SLIDE_OPEN_TAG.fullmatch(line)
        if slide_tag:
            current_slide = {
                "number": int(slide_tag.group(1)),
                "TITLE": [],
                "CONTENT": [],
                "DEFINITIONS": [],
                "KNOWLEDGE_STATEMENTS": [],
            }
            slides.append(current_slide)
            current_section = None
            continue
        if line.casefold() == "[/slide]":
            current_slide = None
            current_section = None
            continue

        tag = _SECTION_TAG.fullmatch(line)
        if tag:
            closing, name = tag.groups()
            current_section = None if closing else name
            continue
        if (
            current_slide is None
            or current_section not in included_sections
            or not line
        ):
            continue
        values = current_slide[current_section]
        assert isinstance(values, list)
        values.append(line)

    module_title = metadata.get("module_title", "")
    facts: list[dict[str, object]] = []

    def add_fact(
        statement: str,
        *,
        slide_number: int,
        kind: str,
        topic: str,
    ) -> None:
        cleaned = _lesson_fact_text(statement)
        if cleaned is None:
            return
        facts.append(
            {
                "statement": cleaned,
                "slides": [slide_number],
                "kind": kind,
                "topic": topic,
            }
        )

    for slide in slides:
        slide_number = int(slide["number"])
        title_values = slide["TITLE"]
        assert isinstance(title_values, list)
        title = next(
            (
                value
                for raw_value in title_values
                if _meaningful(value := re.sub(r"\s+", " ", str(raw_value)).strip())
            ),
            None,
        )
        topic = title or (
            module_title if _meaningful(module_title) else f"Slide {slide_number}"
        )

        fact_count_before_slide = len(facts)
        definition_values = slide["DEFINITIONS"]
        assert isinstance(definition_values, list)
        for raw_definition in definition_values:
            definition = str(raw_definition)
            if definition.startswith("- "):
                definition = definition[2:].strip()
            if " :: " not in definition:
                continue
            term, meaning = definition.split(" :: ", 1)
            add_fact(
                f"{term.strip()}: {meaning.strip()}",
                slide_number=slide_number,
                kind="definition",
                topic=topic,
            )

        statement_values = slide["KNOWLEDGE_STATEMENTS"]
        assert isinstance(statement_values, list)
        for statement in statement_values:
            add_fact(
                str(statement),
                slide_number=slide_number,
                kind="knowledge_statement",
                topic=topic,
            )

        if len(facts) == fact_count_before_slide:
            content_values = slide["CONTENT"]
            assert isinstance(content_values, list)
            for statement in content_values:
                add_fact(
                    str(statement),
                    slide_number=slide_number,
                    kind="content",
                    topic=topic,
                )

    return deduplicate_lesson_fact_records(facts, reassign_ids=True)


def graph_ready_text(text: str) -> str:
    if not parse_module_metadata(text):
        return text.strip()

    output: list[str] = []
    current_section: str | None = None
    included_sections = {
        "TITLE",
        "CONTENT",
        "VISUAL_TEXT",
        "DEFINITIONS",
        "KNOWLEDGE_STATEMENTS",
        "BRIEF_EXPLANATION",
    }
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = raw_line.strip()
        slide_tag = _SLIDE_OPEN_TAG.fullmatch(line)
        if slide_tag:
            if output:
                output.append("")
            output.append(f"Slide {slide_tag.group(1)}")
            current_section = None
            continue
        tag = _SECTION_TAG.fullmatch(line)
        if tag:
            closing, name = tag.groups()
            if closing:
                current_section = None
            elif name in included_sections:
                current_section = name
                if name == "TITLE" and output:
                    output.append("")
            continue
        if line.startswith("extraction_method:") or current_section is None:
            continue
        if line in {NOT_SPECIFIED, f"- {NOT_SPECIFIED}", UNREADABLE, f"- {UNREADABLE}"}:
            continue
        if current_section == "DEFINITIONS" and line.startswith("- ") and " :: " in line:
            term, definition = line[2:].split(" :: ", 1)
            output.append(f"{term}: {definition}")
        else:
            output.append(line[2:] if line.startswith("- ") else line)

    projected = "\n".join(output)
    projected = re.sub(r"\n{3,}", "\n\n", projected).strip()
    return projected
