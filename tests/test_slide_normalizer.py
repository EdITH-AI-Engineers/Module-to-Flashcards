import json
from types import SimpleNamespace

import pytest

from slide_normalizer import (
    SlideNormalizationError,
    normalize_document,
    normalize_slide,
)


def response(**overrides):
    value = {
        "title": "Instruction Cycle",
        "content": ["The processor fetches and decodes instructions."],
        "visual_text": ["Not Specified"],
        "definitions": [
            {"term": "Fetch", "definition": "Retrieve an instruction from memory."}
        ],
        "knowledge_statements": [
            "The processor fetches an instruction before decoding it."
        ],
        "brief_explanation": "The page describes two ordered processing steps.",
        "module_number": "7",
        "module_title": "Computer Architecture",
    }
    value.update(overrides)
    return json.dumps(value)


class FakeBackend:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, user, *, max_tokens):
        self.calls.append((system, user, max_tokens))
        return self.replies.pop(0)


def test_normalize_slide_accepts_fenced_json_and_builds_record():
    backend = FakeBackend([f"```json\n{response()}\n```"])

    slide = normalize_slide(
        backend,
        number=1,
        source_text="Instruction Cycle. The processor fetches and decodes instructions.",
        extraction_method="text",
        attempts=1,
        max_tokens=900,
    )

    assert slide.number == 1
    assert slide.title == "Instruction Cycle"
    assert slide.definitions[0].term == "Fetch"
    assert backend.calls[0][2] == 900
    assert "Use only the supplied extracted page text" in backend.calls[0][0]


def test_normalize_slide_retries_with_focused_validation_feedback():
    backend = FakeBackend(
        [
            response(definitions="not a list"),
            response(title="[SLIDE 99]"),
        ]
    )

    slide = normalize_slide(
        backend,
        number=1,
        source_text="Visible source text",
        extraction_method="ocr",
        attempts=2,
    )

    assert len(backend.calls) == 2
    assert "previous JSON failed validation" in backend.calls[1][1]
    assert "definitions must be a list" in backend.calls[1][1]
    assert slide.title != "[SLIDE 99]"


def test_normalize_slide_rejects_non_object_until_attempts_exhausted():
    backend = FakeBackend(["[]", "[]"])

    with pytest.raises(SlideNormalizationError, match="after 2 attempts"):
        normalize_slide(
            backend,
            number=1,
            source_text="Visible source text",
            extraction_method="text",
            attempts=2,
        )


def test_normalize_slide_rejects_empty_title_and_content():
    backend = FakeBackend(
        [response(title="", content=[], knowledge_statements=[], brief_explanation="")]
    )

    with pytest.raises(SlideNormalizationError, match="title or content"):
        normalize_slide(
            backend,
            number=1,
            source_text="Visible source text",
            extraction_method="text",
            attempts=1,
        )


def test_document_explicit_metadata_overrides_qwen_candidates():
    backend = FakeBackend([response()])
    pages = (SimpleNamespace(number=1, text="Page text", method="text"),)

    module = normalize_document(
        backend,
        pages,
        source_file="module.pdf",
        course_code="CPE0021",
        module_number="01",
        module_title="Explicit Title",
        attempts=1,
    )

    assert module.course_code == "CPE0021"
    assert module.module_number == "01"
    assert module.module_title == "Explicit Title"


def test_document_uses_first_slide_metadata_candidates_when_not_explicit():
    backend = FakeBackend([response(module_number="4", module_title="Networks")])
    pages = (SimpleNamespace(number=1, text="Page text", method="text"),)

    module = normalize_document(
        backend,
        pages,
        source_file="module.pdf",
        course_code=None,
        module_number=None,
        module_title=None,
        attempts=1,
    )

    assert module.course_code == "Not Specified"
    assert module.module_number == "4"
    assert module.module_title == "Networks"
