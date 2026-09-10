from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


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
    r"the module title\b)",
    flags=re.IGNORECASE,
)


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
    used_statements: set[str] = set()

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
        key = re.sub(r"[^\w]+", " ", cleaned, flags=re.UNICODE).strip().casefold()
        if key in used_statements:
            return
        used_statements.add(key)
        facts.append(
            {
                "id": f"f{len(facts) + 1}",
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

    return tuple(facts)


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
