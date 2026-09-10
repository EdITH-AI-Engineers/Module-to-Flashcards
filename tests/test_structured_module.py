import pytest

from structured_module import (
    Definition,
    StructuredModule,
    StructuredSlide,
    extract_lesson_facts,
    graph_ready_text,
    parse_module_metadata,
    render_structured_module,
)


def make_module(*, content=("A processor executes instructions.",)):
    return StructuredModule(
        course_code="CPE0021",
        module_number="01",
        module_title="Processor Fundamentals",
        source_file="module.pdf",
        slides=(
            StructuredSlide(
                number=1,
                extraction_method="text",
                title="Instruction Cycle",
                content=content,
                visual_text=("Not Specified",),
                definitions=(
                    Definition("Instruction cycle", "The steps used to execute an instruction."),
                ),
                knowledge_statements=(
                    "The processor fetches an instruction before decoding it.",
                ),
                brief_explanation="The slide presents the ordered instruction cycle.",
            ),
        ),
    )


def test_render_and_parse_structured_module_round_trip_metadata():
    rendered = render_structured_module(make_module())

    assert rendered.startswith("[MODULE]\nformat_version: 1\n")
    assert "[SLIDE 1]\nextraction_method: text\n" in rendered
    assert "- Instruction cycle :: The steps used to execute an instruction." in rendered
    assert rendered.endswith("[/SLIDE]\n")
    assert parse_module_metadata(rendered) == {
        "format_version": "1",
        "course_code": "CPE0021",
        "module_number": "01",
        "module_title": "Processor Fundamentals",
        "source_file": "module.pdf",
    }


def test_render_neutralizes_control_tag_lines_inside_content():
    rendered = render_structured_module(make_module(content=("[SLIDE 99]", "Safe fact.")))

    assert rendered.count("[SLIDE 99]") == 0
    assert "Safe fact." in rendered


def test_graph_ready_text_excludes_structure_but_preserves_learning_content():
    projected = graph_ready_text(render_structured_module(make_module()))

    assert "[MODULE]" not in projected
    assert projected.startswith("Slide 1\n")
    assert "format_version" not in projected
    assert "source_file" not in projected
    assert "Instruction Cycle" in projected
    assert "A processor executes instructions." in projected
    assert "Instruction cycle: The steps used to execute an instruction." in projected
    assert "The processor fetches an instruction before decoding it." in projected


def test_extract_lesson_facts_keeps_context_and_filters_presentation_noise():
    module = StructuredModule(
        course_code="CPE0021",
        module_number="01",
        module_title="Architecture",
        source_file="slides.pdf",
        slides=(
            StructuredSlide(
                number=3,
                extraction_method="text",
                title="Instruction Cycle",
                content=("Fetch, decode, and execute are processor stages.",),
                visual_text=("Not Specified",),
                definitions=(
                    Definition("Fetch", "Retrieve an instruction from memory."),
                ),
                knowledge_statements=(
                    "The slide introduces the instruction cycle.",
                    "The processor fetches an instruction before decoding it.",
                ),
                brief_explanation="This slide presents the instruction cycle.",
            ),
        ),
    )

    facts = extract_lesson_facts(render_structured_module(module))

    assert facts == (
        {
            "id": "f1",
            "statement": "Fetch: Retrieve an instruction from memory.",
            "slides": [3],
            "kind": "definition",
            "topic": "Instruction Cycle",
        },
        {
            "id": "f2",
            "statement": "The processor fetches an instruction before decoding it.",
            "slides": [3],
            "kind": "knowledge_statement",
            "topic": "Instruction Cycle",
        },
    )


def test_extract_lesson_facts_uses_content_when_slide_has_no_normalized_facts():
    module = make_module(content=("A processor executes instructions.",))
    slide = module.slides[0]
    without_facts = StructuredModule(
        course_code=module.course_code,
        module_number=module.module_number,
        module_title=module.module_title,
        source_file=module.source_file,
        slides=(
            StructuredSlide(
                number=slide.number,
                extraction_method=slide.extraction_method,
                title=slide.title,
                content=slide.content,
                visual_text=slide.visual_text,
                definitions=(),
                knowledge_statements=(),
                brief_explanation=slide.brief_explanation,
            ),
        ),
    )

    assert extract_lesson_facts(render_structured_module(without_facts))[0][
        "statement"
    ] == "A processor executes instructions."


def test_module_requires_at_least_one_slide():
    with pytest.raises(ValueError, match="at least one slide"):
        StructuredModule(
            course_code="CPE0021",
            module_number="1",
            module_title="Empty",
            source_file="module.pdf",
            slides=(),
        )


def test_module_rejects_document_when_every_slide_is_unreadable():
    slide = StructuredSlide(
        number=1,
        extraction_method="ocr",
        title="Not Specified",
        content=("[Unreadable Text]",),
        visual_text=("[Unreadable Text]",),
        definitions=(),
        knowledge_statements=(),
        brief_explanation="Not Specified",
    )

    with pytest.raises(ValueError, match="no readable learning content"):
        StructuredModule(
            course_code="CPE0021",
            module_number="1",
            module_title="Unreadable",
            source_file="module.pdf",
            slides=(slide,),
        )
