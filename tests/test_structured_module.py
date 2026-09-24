import pytest

from structured_module import (
    Definition,
    StructuredModule,
    StructuredSlide,
    deduplicate_lesson_fact_records,
    extract_lesson_facts,
    filter_lesson_fact_records,
    graph_ready_text,
    parse_module_metadata,
    render_structured_module,
)


def test_deduplicate_lesson_facts_keeps_complete_cumulative_list():
    rules = (
        "strive for consistency",
        "enable frequent users to use shortcuts",
        "offer informative feedback",
        "design dialogs to yield closure",
        "prevent errors",
        "permit easy reversal of actions",
        "support internal locus of control",
        "reduce short-term memory load",
    )
    prefix = "Shneiderman's eight golden rules are: "
    records = (
        {
            "id": "f72",
            "statement": prefix + "; ".join(rules[1:]),
            "slides": [20],
        },
        {
            "id": "f73",
            "statement": prefix + "; ".join(item for item in rules if item != rules[1]),
            "slides": [21],
        },
        {
            "id": "f75",
            "statement": prefix + "; ".join(rules),
            "slides": [22],
        },
    )

    facts = deduplicate_lesson_fact_records(records)

    assert len(facts) == 1
    assert facts[0]["id"] == "f75"
    assert facts[0]["statement"] == records[2]["statement"]
    assert facts[0]["slides"] == [20, 21, 22]


def test_deduplicate_lesson_facts_merges_metric_paraphrases():
    records = (
        {
            "id": "f60",
            "statement": (
                "Learnability metrics include percentage of functions learned, "
                "time to learn, and ease-of-learning ratings."
            ),
            "slides": [12],
        },
        {
            "id": "f64",
            "statement": (
                "Learnability metrics include the percentage of functions learned, "
                "time to learn, and ease-of-learning ratings."
            ),
            "slides": [13],
        },
    )

    facts = deduplicate_lesson_fact_records(records)

    assert len(facts) == 1
    assert facts[0]["id"] == "f60"
    assert facts[0]["slides"] == [12, 13]


def test_deduplicate_lesson_facts_keeps_parallel_concepts_distinct():
    records = (
        {
            "id": "f40",
            "statement": "A system is useful if it provides the functions users need.",
            "slides": [8],
        },
        {
            "id": "f41",
            "statement": "A system is usable if users can operate its functions effectively.",
            "slides": [8],
        },
    )

    facts = deduplicate_lesson_fact_records(records)

    assert [fact["id"] for fact in facts] == ["f40", "f41"]


@pytest.mark.parametrize(
    ("records", "canonical_id", "expected_slides"),
    [
        (
            (
                {
                    "id": "f53",
                    "statement": (
                        "The ISO 9241 standard outlines traditional usability "
                        "categories with specific measures such as effectiveness, "
                        "efficiency, and satisfaction."
                    ),
                    "slides": [25],
                },
                {
                    "id": "f59",
                    "statement": (
                        "The ISO 9241 standard addresses ergonomics in human-system "
                        "interaction and outlines traditional usability categories "
                        "with specific measures."
                    ),
                    "slides": [26, 27],
                },
            ),
            "f53",
            [25, 26, 27],
        ),
        (
            (
                {
                    "id": "f60",
                    "statement": (
                        "Usability in HCI includes aspects such as effectiveness, "
                        "efficiency, and satisfaction, which are evaluated through "
                        "metrics like percentage of time to complete tasks and "
                        "rating scales for user satisfaction."
                    ),
                    "slides": [26],
                },
                {
                    "id": "f63",
                    "statement": (
                        "Usability is categorized into effectiveness, efficiency, "
                        "and satisfaction, each with metrics such as percentage of "
                        "time to complete tasks and rating scales for satisfaction."
                    ),
                    "slides": [27],
                },
            ),
            "f60",
            [26, 27],
        ),
        (
            (
                {
                    "id": "f57",
                    "statement": (
                        "Learnability is determined by the percentage of time to "
                        "learn functions."
                    ),
                    "slides": [25],
                },
                {
                    "id": "f61",
                    "statement": (
                        "Learnability is assessed using metrics such as the "
                        "percentage of time to learn functions and rating scales "
                        "for ease of learning."
                    ),
                    "slides": [26],
                },
                {
                    "id": "f64",
                    "statement": (
                        "Learnability is measured by the percentage of time to "
                        "learn functions and the ease of learning for users."
                    ),
                    "slides": [27],
                },
            ),
            "f61",
            [25, 26, 27],
        ),
        (
            (
                {
                    "id": "f58",
                    "statement": (
                        "Error tolerance is measured by the percentage of time "
                        "spent on correcting errors."
                    ),
                    "slides": [25],
                },
                {
                    "id": "f62",
                    "statement": (
                        "Error tolerance in HCI involves measuring the percentage "
                        "of time spent on correcting errors and evaluating error "
                        "handling effectiveness through rating scales."
                    ),
                    "slides": [26],
                },
                {
                    "id": "f65",
                    "statement": (
                        "Error tolerance involves metrics like the percentage of "
                        "time spent on correcting errors and successful error "
                        "handling."
                    ),
                    "slides": [27],
                },
            ),
            "f62",
            [25, 26, 27],
        ),
    ],
)
def test_deduplicate_lesson_facts_collapses_same_subject_paraphrase_families(
    records,
    canonical_id,
    expected_slides,
):
    facts = deduplicate_lesson_fact_records(records)

    assert len(facts) == 1
    assert facts[0]["id"] == canonical_id
    assert facts[0]["slides"] == expected_slides


def test_deduplicate_lesson_facts_keeps_same_subject_different_claims():
    records = (
        {
            "id": "f1",
            "statement": "Photosynthesis occurs in chloroplasts within plant cells.",
            "slides": [1],
        },
        {
            "id": "f2",
            "statement": (
                "Photosynthesis converts light energy into stored chemical energy."
            ),
            "slides": [2],
        },
    )

    facts = deduplicate_lesson_fact_records(records)

    assert [fact["id"] for fact in facts] == ["f1", "f2"]


def test_filter_lesson_facts_removes_generic_presentation_artifacts():
    records = (
        {
            "id": "f1",
            "statement": "Cell Structure",
            "kind": "content",
            "topic": "Cell Structure",
        },
        {
            "id": "f2",
            "statement": "The learner should be able to identify cell organelles.",
            "kind": "knowledge_statement",
            "topic": "Objectives",
        },
        {
            "id": "f3",
            "statement": "https://example.edu/reference/cell-structure",
            "kind": "content",
            "topic": "References",
        },
        {
            "id": "f4",
            "statement": "that regulates entry into the cell",
            "kind": "content",
            "topic": "Cell Membrane",
        },
        {
            "id": "f5",
            "statement": "The cell membrane regulates entry into the cell.",
            "kind": "knowledge_statement",
            "topic": "Cell Membrane",
        },
    )

    facts = filter_lesson_fact_records(records)

    assert [fact["id"] for fact in facts] == ["f5"]


def test_filter_lesson_facts_keeps_a_sparse_short_substantive_claim():
    records = (
        {
            "id": "f1",
            "statement": "Ice melts.",
            "kind": "content",
            "topic": "Phase Changes",
        },
        {
            "id": "f2",
            "statement": "IT systems improve communication.",
            "kind": "content",
            "topic": "Information Technology",
        },
    )

    assert filter_lesson_fact_records(records) == records


def test_filter_lesson_facts_removes_numbered_presentation_metadata_generically():
    records = (
        {
            "id": "f1",
            "statement": "Module 3 Title: Programming Fundamentals.",
            "kind": "knowledge_statement",
        },
        {
            "id": "f2",
            "statement": "Module 3 is titled Programming Fundamentals.",
            "kind": "knowledge_statement",
        },
        {
            "id": "f3",
            "statement": "The title of Module 3 is Programming Fundamentals.",
            "kind": "knowledge_statement",
        },
        {
            "id": "f4",
            "statement": "Module 3 focuses on programming fundamentals.",
            "kind": "knowledge_statement",
        },
        {
            "id": "f5",
            "statement": "Module 3 | title | Programming Fundamentals",
            "kind": "knowledge_statement",
        },
        {
            "id": "f6",
            "statement": "A software module exposes a public interface.",
            "kind": "knowledge_statement",
        },
        {
            "id": "f7",
            "statement": "A document title identifies a work.",
            "kind": "knowledge_statement",
        },
    )

    facts = filter_lesson_fact_records(records)

    assert [fact["id"] for fact in facts] == ["f6", "f7"]


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
                    "The module introduces HCI principles and concepts.",
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


def test_extract_lesson_facts_falls_back_when_normalized_items_are_only_noise():
    module = StructuredModule(
        course_code="BIO101",
        module_number="01",
        module_title="Cells",
        source_file="cells.pdf",
        slides=(
            StructuredSlide(
                number=1,
                extraction_method="text",
                title="Cell Membrane",
                content=("The cell membrane regulates entry into the cell.",),
                visual_text=("Not Specified",),
                definitions=(),
                knowledge_statements=(
                    "The learner should be able to identify cell structures.",
                ),
                brief_explanation="The slide presents the cell membrane.",
            ),
        ),
    )

    facts = extract_lesson_facts(render_structured_module(module))

    assert [fact["statement"] for fact in facts] == [
        "The cell membrane regulates entry into the cell."
    ]


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
