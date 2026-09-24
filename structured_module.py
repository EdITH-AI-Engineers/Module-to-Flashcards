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
    r"the module title\b|(?:this|the)\s+(?:module|lesson|chapter|section)\s+"
    r"(?:introduces|covers|discusses|presents|contains|provides an overview of)\b|"
    r"(?:this|the)\s+(?:content|material|presentation)\s+"
    r"(?:includes|contains|covers|discusses|introduces|presents|is titled)\b)",
    flags=re.IGNORECASE,
)
_FACT_LIST_MARKER = re.compile(r"^(?:[-\u2022\u25aa\u25e6\u2023]|\d+[.)])\s*")
_FACT_URL = re.compile(r"(?:https?://|www\.|\bdoi\s*:)", flags=re.IGNORECASE)
_FACT_SECTION_PATH = re.compile(
    r"^(?:module|unit|chapter|lesson|section)\s*[a-z0-9.-]*\s*"
    r"[:|>\u203a\u2192-]",
    flags=re.IGNORECASE,
)
_FACT_LEARNING_OBJECTIVE = re.compile(
    r"^(?:(?:learning\s+)?objectives?\b|"
    r"(?:the\s+)?(?:learner|student|reader)s?\s+"
    r"(?:will|should|can|must)\s+(?:be\s+able\s+to\s+)?)",
    flags=re.IGNORECASE,
)
_FACT_PRESENCE_ONLY = re.compile(
    r"\b(?:is|are|was|were)\s+(?:a\s+|the\s+)?(?:key\s+)?"
    r"(?:aspect|concept|item|part|section|topic)s?\s+"
    r"(?:covered|discussed|included|introduced|mentioned|presented)\s+in\b",
    flags=re.IGNORECASE,
)
_FACT_PROVENANCE_SUFFIX = re.compile(
    r"\b(?:covered|discussed|included|introduced|mentioned|presented)\s+in\s+"
    r"(?:this|the)\s+(?:module|lesson|chapter|section|slide|presentation)\b",
    flags=re.IGNORECASE,
)
_FACT_QUOTE_META = re.compile(
    r"^(?:the\s+)?(?:quote|quotation)\s+(?:by|from)\b.*\b"
    r"(?:mentions|states|says|describes)\b",
    flags=re.IGNORECASE,
)
# The normalizer prompt already tells the model never to write a knowledge
# statement that describes presentation metadata, but a noncompliant
# generation can still narrate the slide's own title/content fields back as
# if they were a fact -- e.g. "The module is titled 'Module 3: ...' and the
# content is 'Responders'." or "The content of the slide is about the
# properties of light." Neither reads as an enumeration, a presence-only
# claim, or provenance-suffixed like the checks above, since the sentence
# itself has verb structure -- it just happens to be about the slide's own
# labeling rather than any module content. Ban that family explicitly rather
# than relying on the graph-consumption side to catch every phrasing.
_FACT_LABEL_NARRATION = re.compile(
    r"^(?:the\s+)?(?:title|heading|name|number|topic|focus|overview|content)\s+"
    r"(?:of|for)\s+(?:the\s+)?(?:module|unit|chapter|lesson|section|slide|page)"
    r"(?:\s+(?:\d+[a-z]?|[ivxlcdm]+))?\b|"
    r"^(?:the\s+)?(?:module|unit|chapter|lesson|section|slide|page)\s+"
    r"(?:\d+[a-z]?|[ivxlcdm]+)\b(?:\s*(?:[:|>\-])|\s+"
    r"(?:title|heading|name|number|topic|focus|overview|content|"
    r"is\s+(?:sub)?titled|is\s+named|is\s+called|covers?|discusses?|"
    r"focuses?\s+on|provides?\s+an?\s+overview))|"
    r"^(?:this|the)\s+(?:module|unit|chapter|lesson|section|slide|page)\s+"
    r"(?:is\s+)?(?:sub)?titled\b|"
    r"^content\s+of\s+(?:this|the)\s+"
    r"(?:module|unit|chapter|lesson|section|slide|page)\s+"
    r"(?:is\s+(?:about|titled)\b|discusses\b|mentions\b|includes\b)",
    flags=re.IGNORECASE,
)
_FACT_FRAGMENT_START_WORDS = {
    "and",
    "or",
    "but",
    "it",
    "this",
    "these",
    "those",
    "that",
    "which",
    "who",
    "whose",
    "where",
    "when",
    "while",
    "because",
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "can",
    "could",
    "may",
    "might",
    "must",
    "should",
    "would",
}
_FACT_FRAGMENT_END = re.compile(
    r"(?:[,;:/-]|\b(?:a|an|and|as|at|by|for|from|in|of|on|or|the|to|with))$",
    flags=re.IGNORECASE,
)
_FACT_RELATION_VERB = re.compile(
    r"\b(?:is|are|was|were|becomes?|refers?|means?|defines?|describes?|"
    r"includes?|contains?|comprises?|consists?|involves?|uses?|stores?|"
    r"produces?|converts?|causes?|affects?|determines?|requires?|enables?|"
    r"supports?|provides?|allows?|prevents?|reduces?|increases?|improves?|"
    r"measures?|evaluates?|assesses?|represents?|follows?|precedes?|creates?|"
    r"controls?|performs?|occurs?|exists?|depends?|leads?|results?)\b",
    flags=re.IGNORECASE,
)
_FACT_PREDICATE_START = re.compile(
    r"\b(?:is|are|was|were|has|have|had|can|could|may|might|must|should|would|"
    r"refers?|means?|defines?|describes?|outlines?|addresses?|includes?|contains?|"
    r"comprises?|consists?|involves?|uses?|stores?|produces?|converts?|causes?|"
    r"affects?|determines?|requires?|enables?|supports?|provides?|allows?|prevents?|"
    r"reduces?|increases?|improves?|measures?|measured|measuring|evaluates?|"
    r"evaluated|evaluating|assesses?|assessed|assessing|categorizes?|categorized|"
    r"represents?|follows?|precedes?|creates?|controls?|performs?|occurs?|exists?|"
    r"depends?|leads?|results?)\b",
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


def _clean_fact_text(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if text.startswith("- "):
        text = text[2:].strip()
    return _FACT_LIST_MARKER.sub("", text).strip()


def _is_self_contained_content_fact(statement: str) -> bool:
    """Return whether fallback slide text states a usable claim on its own.

    Normalized definitions and knowledge statements have already been rewritten
    as facts. Raw CONTENT is only a fallback, so incomplete OCR lines, list
    labels, and headings need a stricter check before becoming fact IDs.
    """

    first_word_match = re.match(r"[A-Za-z]+", statement)
    first_word = first_word_match.group(0) if first_word_match else ""
    starts_with_fragment_word = (
        first_word.casefold() in _FACT_FRAGMENT_START_WORDS
        and not first_word.isupper()
    )
    if starts_with_fragment_word or _FACT_FRAGMENT_END.search(statement):
        return False
    if statement[0].islower():
        return False
    words = re.findall(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)*", statement)
    if len(words) < 2:
        return False
    if len(words) == 2:
        return statement.endswith((".", "!")) and bool(
            re.search(r"[a-z]", words[1])
        )
    if statement.endswith((".", "!")) or ": " in statement:
        return True
    if _FACT_RELATION_VERB.search(statement):
        return True
    # A longer line can be a complete unpunctuated OCR sentence. Requiring
    # length and some lower-case prose prevents title-like text from passing.
    lower_case_words = sum(bool(re.search(r"[a-z]", word)) for word in words)
    return len(words) >= 8 and lower_case_words >= len(words) // 2


def filter_lesson_fact_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Remove non-facts without imposing a module-specific vocabulary.

    The rules identify presentation form rather than subject matter, so they
    apply equally to technical, humanities, and other lesson modules. No
    minimum fact count is imposed: a sparse module retains its few substantive
    claims and can still follow the normal insufficient-content path.
    """

    usable: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        statement = _clean_fact_text(record.get("statement", ""))
        if (
            not _meaningful(statement)
            or len(statement) < 5
            or len(statement.split()) < 2
        ):
            continue
        topic = _clean_fact_text(record.get("topic", ""))
        if topic and _normalized_fact_text(statement) == _normalized_fact_text(topic):
            continue
        if (
            _FACT_URL.search(statement)
            or statement.endswith("?")
            or _FACT_SECTION_PATH.match(statement)
            or _FACT_LEARNING_OBJECTIVE.match(statement)
            or _PRESENTATION_NOISE.match(statement)
            or _FACT_PRESENCE_ONLY.search(statement)
            or _FACT_PROVENANCE_SUFFIX.search(statement)
            or _FACT_QUOTE_META.match(statement)
            or _FACT_LABEL_NARRATION.search(statement)
        ):
            continue
        kind = str(record.get("kind", "")).strip().casefold()
        if kind in {"title", "header", "heading", "objective"}:
            continue
        if kind == "content" and not _is_self_contained_content_fact(statement):
            continue
        candidate = dict(record)
        candidate["statement"] = statement
        usable.append(candidate)
    return tuple(usable)


def _fact_content_tokens(value: object) -> tuple[str, ...]:
    return tuple(
        token
        for token in _normalized_fact_text(value).split()
        if token not in _FACT_BRIDGE_WORDS
    )


def _fact_subject_tokens(value: object) -> frozenset[str]:
    normalized = _normalized_fact_text(value)
    match = _FACT_PREDICATE_START.search(normalized)
    if match is None:
        return frozenset()
    return frozenset(
        token
        for token in normalized[: match.start()].split()
        if token not in _FACT_BRIDGE_WORDS
    )


def _same_fact_subject(left: object, right: object) -> bool:
    left_subject = _fact_subject_tokens(left)
    right_subject = _fact_subject_tokens(right)
    if not left_subject or not right_subject:
        return False
    smaller, larger = sorted(
        (left_subject, right_subject),
        key=len,
    )
    return smaller <= larger and len(larger) - len(smaller) <= 1


def _looks_like_enumeration(value: object) -> bool:
    text = str(value or "")
    separators = len(re.findall(r"[,;]", text))
    numbered_items = len(re.findall(r"(?:^|\s)\d+[.)]\s", text))
    return separators >= 2 or numbered_items >= 3


def _fact_information_score(record: Mapping[str, object]) -> tuple[int, int]:
    statement = record["statement"]
    return (
        int(_looks_like_enumeration(statement)),
        len(_fact_content_tokens(statement)),
    )


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
        and overlap >= 0.80
        and jaccard >= 0.62
        and sequence >= 0.62
    ):
        return True

    subject_tokens = _fact_subject_tokens(left) | _fact_subject_tokens(right)
    shared_claim_tokens = shared - subject_tokens
    if (
        _same_fact_subject(left, right)
        and min(len(left_tokens), len(right_tokens)) >= 5
        and overlap >= 0.65
        and len(shared_claim_tokens) >= 3
    ):
        return True

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

    The richest statement wins, so partial or cumulative renderings do not
    survive as separate fact IDs. Ties preserve stable source order.
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

        chosen = max(
            (*duplicate_group, candidate),
            key=_fact_information_score,
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
    text = _clean_fact_text(value)
    if not _meaningful(text) or _PRESENTATION_NOISE.match(text):
        return None
    if len(text) < 5 or len(text.split()) < 2:
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
        candidate = {
            "statement": cleaned,
            "slides": [slide_number],
            "kind": kind,
            "topic": topic,
        }
        usable = filter_lesson_fact_records((candidate,))
        if usable:
            facts.append(usable[0])

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

    usable_facts = filter_lesson_fact_records(facts)
    return deduplicate_lesson_fact_records(usable_facts, reassign_ids=True)


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
