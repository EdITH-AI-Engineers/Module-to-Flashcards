from collections import Counter
from dataclasses import replace

import pytest

import api_server
import batch_pipeline
import pipeline
import text_extractor
from batch_pipeline import BatchItem
from tests.batch_helpers import counting_dependencies, structured_content
from structured_module import (
    extract_lesson_facts,
    graph_ready_text,
    parse_slide_report_metadata,
)


REPORT = """Module #: Not Specified
Module Title: BASIC ELECTRICAL ENGINEERING

---

Slide 1:
{
Title:
BASIC ELECTRICAL ENGINEERING

Content:
Course: BASICEE
Tagline: *The Innovation College*

Image/Diagram Description:
A printed circuit board.
}

Brief Explanation:
This slide introduces the course.

---

Slide 2:
{
Title:
MODULE 9

Content:
Not Specified

Image/Diagram Description:
Not Specified
}

Brief Explanation:
This slide introduces Module 9.

---

Slide 4:
{
Title:
DEFINITION OF TERMS

Content:
- Frequency is the number of cycles per second, measured in hertz.
- Period is the time taken to complete one cycle.

Image/Diagram Description:
Not Specified
}

Brief Explanation:
This slide defines alternating current terms.
"""


def test_slide_report_stages_original_text_and_uses_title_module_number(tmp_path):
    source = tmp_path / "EDITH-2503-0522-5807-75B3.txt"
    source.write_text(REPORT, encoding="utf-8")
    destination = tmp_path / "stage" / "structured_module.txt"

    assert api_server.structured_module_number(source, "COE0041") == "9"
    pipeline.stage_structured_module(
        source,
        destination,
        course_code="COE0041",
        module_number="09",
    )

    assert destination.read_bytes() == source.read_bytes()
    assert destination.read_text(encoding="utf-8").startswith("Module #:")


def test_slide_report_uses_requested_course_id_not_cover_abbreviation(tmp_path):
    source = tmp_path / "module.txt"
    source.write_text(REPORT, encoding="utf-8")

    assert api_server.structured_module_number(source, "COE0041") == "9"
    assert "course_code" not in parse_slide_report_metadata(REPORT)


def test_slide_report_payload_course_overrides_header_course_code(tmp_path):
    report = REPORT.replace(
        "Module Title: BASIC ELECTRICAL ENGINEERING",
        "Course Code: BASICEE\nModule Title: BASIC ELECTRICAL ENGINEERING",
        1,
    )
    source = tmp_path / "module.txt"
    source.write_text(report, encoding="utf-8")

    assert api_server.structured_module_number(source, "COE0041") == "9"
    _, metadata = text_extractor.prepare_input_text(
        report, course_code="COE0041", module_number="9"
    )
    assert metadata["course_code"] == "COE0041"


def test_slide_report_graph_input_preserves_course_and_slide_provenance():
    prepared, metadata = text_extractor.prepare_input_text(
        REPORT,
        course_code="COE0041",
        module_number="9",
        source_file="EDITH-2503-0522-5807-75B3.txt",
    )
    facts = extract_lesson_facts(REPORT)

    assert metadata == {
        "module_title": "BASIC ELECTRICAL ENGINEERING",
        "module_number": "9",
        "course_code": "COE0041",
        "source_file": "EDITH-2503-0522-5807-75B3.txt",
    }
    assert "Slide 4\nDEFINITION OF TERMS" in prepared
    assert "Slide 1" not in prepared
    assert "Slide 2" not in prepared
    assert "Frequency is the number of cycles per second" in prepared
    assert "Module #: Not Specified" not in prepared
    assert "Tagline:" not in prepared
    assert [(fact["statement"], fact["slides"]) for fact in facts] == [
        ("Frequency is the number of cycles per second, measured in hertz.", [4]),
        ("Period is the time taken to complete one cycle.", [4]),
    ]


def test_slide_report_requires_discoverable_module_number(tmp_path):
    without_module_title = REPORT.replace("MODULE 9", "OVERVIEW")
    source = tmp_path / "report.txt"
    source.write_text(without_module_title, encoding="utf-8")

    assert "module_number" not in parse_slide_report_metadata(without_module_title)
    with pytest.raises(ValueError, match="module number"):
        api_server.structured_module_number(source, "BASICEE")


@pytest.mark.parametrize("content", (REPORT, structured_content("9")))
def test_filename_identity_skips_conflicting_report_identity_during_ingest(
    tmp_path, monkeypatch, content
):
    source = tmp_path / "CS0003_M2.txt"
    source.write_text(content, encoding="utf-8")
    monkeypatch.setattr(api_server, "OUTPUT_ROOT", tmp_path / "output")
    args = api_server.pipeline_args(
        source,
        "CS0003",
        "2",
        identity_from_filename=True,
    )
    paths = pipeline.pipeline_paths(
        source,
        args.output_root,
        args.course_code,
        args.module_number,
    )
    item = BatchItem(source.name, args, paths)

    staged = batch_pipeline._ingest_stage(item, object())

    assert staged.read_bytes() == source.read_bytes()


def test_filename_identity_can_reuse_staged_report_without_module_label(
    tmp_path, monkeypatch
):
    source = tmp_path / "CS0003_M2.txt"
    source.write_text(REPORT.replace("MODULE 9", "OVERVIEW"), encoding="utf-8")
    monkeypatch.setattr(api_server, "OUTPUT_ROOT", tmp_path / "output")
    args = api_server.pipeline_args(
        source, "CS0003", "2", identity_from_filename=True
    )
    paths = pipeline.pipeline_paths(source, args.output_root, "CS0003", "2")
    item = BatchItem(source.name, args, paths)
    batch_pipeline._ingest_stage(item, object())
    batch_pipeline._write_manifest(item)

    state = batch_pipeline._make_state(item, None)

    assert not state.needs_ingest
    dependencies = replace(
        counting_dependencies(Counter()),
        ingest_stage=batch_pipeline._ingest_stage,
    )
    result = batch_pipeline.run_batch((item,), dependencies=dependencies)
    assert result.errors == ()
    assert result.outputs == paths.flashcard_parts
    assert batch_pipeline._make_state(item, None).stage == "complete"


def test_filename_report_passes_batch_ingest_validation(tmp_path, monkeypatch):
    source = tmp_path / "CS0003_M2.txt"
    source.write_text(REPORT.replace("MODULE 9", "OVERVIEW"), encoding="utf-8")
    monkeypatch.setattr(api_server, "OUTPUT_ROOT", tmp_path / "output")
    args = api_server.pipeline_args(
        source, "CS0003", "2", identity_from_filename=True
    )
    paths = pipeline.pipeline_paths(source, args.output_root, "CS0003", "2")
    item = BatchItem(source.name, args, paths)
    dependencies = replace(
        counting_dependencies(Counter()),
        ingest_stage=batch_pipeline._ingest_stage,
    )

    result = batch_pipeline.run_batch((item,), dependencies=dependencies)

    assert result.errors == ()
    assert result.outputs == paths.flashcard_parts


def test_slide_report_uses_substantive_explanation_when_content_is_missing():
    report = """Module #: 9
Module Title: AC circuits

Slide 12:
{
Title:
Time domain to phasor domain
Content:
Not Specified
Image/Diagram Description:
Not Specified
}
Brief Explanation:
This slide demonstrates a conversion. Phasors represent sinusoidal signals as complex numbers. Phasor representation simplifies AC circuit calculations.
"""

    assert [fact["statement"] for fact in extract_lesson_facts(report)] == [
        "Phasors represent sinusoidal signals as complex numbers.",
        "Phasor representation simplifies AC circuit calculations.",
    ]


def test_slide_report_does_not_treat_exercise_prompts_as_facts():
    report = """Module #: 9
Module Title: AC circuits

Slide 10:
{
Title:
Try this
Content:
Task: Find the RMS value of the waveform.
What will the average value be.
Ans: 3.1831 A and 5 A.
The RMS value of a sinusoidal current equals its peak value divided by square root of two.
Image/Diagram Description:
Not Specified
}
Brief Explanation:
This slide presents a calculation exercise.
"""

    assert [fact["statement"] for fact in extract_lesson_facts(report)] == [
        "The RMS value of a sinusoidal current equals its peak value divided by square root of two."
    ]
    prepared, _ = text_extractor.prepare_input_text(report, course_code="BASICEE")
    assert "Task: Find" not in prepared
    assert "Ans: 3.1831" not in prepared


def test_slide_report_uses_explanation_on_topic_slide_with_branding_labels():
    report = """Module #: 9
Module Title: BASIC ELECTRICAL ENGINEERING

Slide 29:
{
Title:
IMPEDANCE COMBINATION
Content:
Course: BASIC ELECTRICAL ENGINEERING
Institution: FEU TECH
Image/Diagram Description:
Not Specified
}
Brief Explanation:
This slide introduces impedance combination. Impedances in series add to form an equivalent impedance.
"""

    assert [fact["statement"] for fact in extract_lesson_facts(report)] == [
        "Impedances in series add to form an equivalent impedance."
    ]


def test_slide_report_excludes_qa_closing_slide_from_facts(tmp_path):
    report = """Module #: 9
Module Title: BASIC ELECTRICAL ENGINEERING

Slide 45:
{
Title:
Q&A SESSION
Content:
FEU ALABANG
FEU DILIMAN
Image/Diagram Description:
Colorful question-mark speech bubbles.
}
Brief Explanation:
The illustration of speech bubbles with question marks invites discussion. The institutions hosted the event.
"""

    assert extract_lesson_facts(report) == ()
    assert graph_ready_text(report) == ""
    source = tmp_path / "qa-only.txt"
    source.write_text(report, encoding="utf-8")
    with pytest.raises(ValueError, match="no readable learning content"):
        api_server.structured_module_number(source, "BASICEE")
