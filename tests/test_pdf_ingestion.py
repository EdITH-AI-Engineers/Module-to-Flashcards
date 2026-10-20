import asyncio
import io
import json

import httpx
import pymupdf
import pytest

import api_server
import batch_pipeline
from batch_pipeline import BatchResult
from module_sources import course_folder_pdfs, pdf_identity
import pdf_ingestion as pdf
import pipeline
from structured_module import extract_lesson_facts


def module_bytes():
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((45, 40), "DECORATIVE ACADEMY LOGO", fontsize=12)
        page.insert_text((45, 160), "Voltage equals current multiplied by resistance.\nResistance opposes current flow in an electrical circuit.\nOhm's law is expressed as V = I R.", fontsize=15)
        return document.tobytes()


class FilterBackend:
    def __init__(self):
        self.calls = []
    def complete(self, system, user, **kwargs):
        values = json.loads(user)["blocks"]
        self.calls.append(values)
        return json.dumps({"remove": ["DECORATIVE" in block["text"] for block in values],
            "study_groups": [[index] for index, block in enumerate(values) if "DECORATIVE" not in block["text"]]})


def test_selectable_pdf_filters_only_decorations_and_preserves_source(tmp_path, monkeypatch):
    source, target = tmp_path / "Module 1.pdf", tmp_path / "output" / "structured_module.txt"
    original = module_bytes()
    source.write_bytes(original)
    monkeypatch.setattr(pdf, "_ocr_blocks", lambda *a, **k: pytest.fail("Readable text needs no OCR"))
    backend = FilterBackend()
    pdf.convert_pdf(source, target, course_code="EE101", module_number="1", backend=backend)
    text = target.read_text(encoding="utf-8")
    assert "DECORATIVE ACADEMY" not in text
    assert "V = I R" in text and "Voltage equals current multiplied by resistance." in text
    assert "Resistance opposes current flow" in text
    assert source.read_bytes() == original
    assert extract_lesson_facts(text)
    audit = json.loads(target.with_name("pdf_extraction.json").read_text(encoding="utf-8"))
    assert audit[0]["method"] == "text"
    assert any(block["removed"] and "DECORATIVE" in block["text"] for block in audit[0]["blocks"])
    assert not list(target.parent.glob("*.tmp"))


def test_real_tesseract_reads_scanned_page(tmp_path):
    try:
        pdf.find_tesseract()
    except RuntimeError:
        pytest.skip("Tesseract is not installed")
    with pymupdf.open(stream=module_bytes(), filetype="pdf") as text_document:
        image = text_document[0].get_pixmap(dpi=160).tobytes("png")
    source = tmp_path / "scan.pdf"
    with pymupdf.open() as scanned:
        page = scanned.new_page()
        page.insert_image(page.rect, stream=image)
        scanned.save(source)
    target = tmp_path / "result" / "structured_module.txt"
    pdf.convert_pdf(source, target, course_code="EE101", module_number="2", backend=FilterBackend())
    text = target.read_text(encoding="utf-8")
    assert "extraction_method: ocr" in text
    assert "Resistance opposes current flow" in text
    assert "DECORATIVE ACADEMY" not in text


def test_mixed_page_with_native_title_and_scanned_body_still_runs_ocr(monkeypatch):
    calls = []
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((40, 40), "A sufficiently long native title " * 5)
        from PIL import Image
        image = io.BytesIO()
        Image.new("RGB", (500, 500), "white").save(image, format="PNG")
        page.insert_image(pymupdf.Rect(40, 100, 500, 600), stream=image.getvalue())
        monkeypatch.setattr(pdf, "_ocr_blocks", lambda *a, **k: calls.append(True) or [pdf.TextBlock("Scanned body teaches Ohm's law.", (0.1, .2, .9, .8))])
        blocks, method = pdf.extract_page(page)
    assert method == "ocr" and calls
    assert any("Scanned body" in block.text for block in blocks)
    assert any("native title" in block.text for block in blocks)


def test_filter_retries_invalid_decisions_and_cannot_rewrite_retained_text():
    responses = iter(['{"remove":[]}', '{"remove":[false,true]}'])
    class Backend:
        def complete(self, *args, **kwargs):
            return next(responses)
    blocks = [pdf.TextBlock("E = mc^2 (do not simplify)", (0, .2, 1, .3)), pdf.TextBlock("Company logo", (0, 0, .2, .1))]
    assert pdf.filter_blocks(blocks, Backend(), page_number=1) == [False, True]


def test_page_25_conflicting_remove_and_study_selection_preserves_lesson_block():
    blocks = [pdf.TextBlock(f"Civil engineering service {index}", (0, 0, 1, 1)) for index in range(21)]
    blocks[18] = pdf.TextBlock("Materials process design", (0, 0, 1, 1))
    decisions = [False] * len(blocks)
    decisions[18] = True
    calls = []
    class Backend:
        def complete(self, system, user, **kwargs):
            calls.append(user)
            return json.dumps({"remove": decisions, "study_groups": [list(range(21))]})
    quotes, warnings = [], []
    result = pdf.filter_blocks(blocks, Backend(), page_number=25, study_quotes=quotes, warnings=warnings)
    assert result == [False] * len(blocks)
    assert len(calls) == 1 and "Materials process design" in quotes[0]
    assert warnings == ["Kept 1 block(s) marked as both decorative and lesson content."]


def test_study_groups_can_reuse_headings_and_normalize_duplicate_source_indexes():
    blocks = [pdf.TextBlock(text, (0, 0, 1, 1)) for text in (
        "Special engineering services:", "Computer modeling.", "Safety engineering.")]
    class Backend:
        def complete(self, *args, **kwargs):
            return '{"remove":[false,false,false],"study_groups":[[0,2],[1,0,0],[0,1]]}'
    quotes = []
    assert pdf.filter_blocks(blocks, Backend(), page_number=1, study_quotes=quotes) == [False] * 3
    assert quotes == ["Special engineering services: Safety engineering.",
                      "Special engineering services: Computer modeling."]


@pytest.mark.parametrize("bad_response", ["not json", "[]", '{"remove":[0],"study_groups":[]}',
    '{"remove":[true],"study_groups":[[99]]}'])
def test_unrecoverable_cleanup_preserves_source_with_review_note(bad_response):
    source = "Voltage equals current multiplied by resistance."
    calls = []
    class Backend:
        def complete(self, system, user, **kwargs):
            calls.append(system)
            return bad_response
    quotes, warnings = [], []
    result = pdf.filter_blocks([pdf.TextBlock(source, (0, 0, 1, 1))], Backend(),
        page_number=25, attempts=2, study_quotes=quotes, warnings=warnings)
    assert result == [False] and quotes == [source]
    assert len(calls) == 2 and "previous response failed validation" in calls[1]
    assert "kept the original text" in warnings[0]


def test_cut_short_cleanup_increases_bounded_budget_and_sends_repair_feedback():
    from flashcard_types import CompletionTruncatedError
    budgets, prompts = [], []
    class Backend:
        def complete(self, system, user, **kwargs):
            budgets.append(kwargs["max_tokens"])
            prompts.append(system)
            if len(budgets) < 3:
                raise CompletionTruncatedError("Output cut short", partial_content='{"remove":[')
            return '{"remove":[false],"study_groups":[[0]]}'
    quotes = []
    source = "Resistance opposes current flow."
    assert pdf.filter_blocks([pdf.TextBlock(source, (0, 0, 1, 1))], Backend(),
        page_number=25, study_quotes=quotes) == [False]
    assert budgets[0] < budgets[1] < budgets[2] <= 1024
    assert "response was cut short" in prompts[1] and quotes == [source]


def test_provider_failure_is_not_hidden_by_cleanup_fallback():
    from mistral_backend import MistralRequestError
    calls = []
    class Backend:
        def complete(self, *args, **kwargs):
            calls.append(True)
            raise MistralRequestError("Model access denied", category="access_denied", status=403)
    quotes, warnings = [], []
    with pytest.raises(MistralRequestError, match="Model access denied"):
        pdf.filter_blocks([pdf.TextBlock("Voltage equals current times resistance.", (0, 0, 1, 1))],
            Backend(), page_number=25, study_quotes=quotes, warnings=warnings)
    assert calls == [True] and not quotes and not warnings


def test_invalid_cleanup_on_page_25_keeps_pdf_and_continues_to_page_26(tmp_path, caplog):
    source, target = tmp_path / "Module 2.pdf", tmp_path / "result" / "structured_module.txt"
    with pymupdf.open() as document:
        for number in range(1, 27):
            page = document.new_page()
            page.insert_text((45, 40), "DECORATIVE ACADEMY LOGO", fontsize=12)
            page.insert_text((45, 160), f"Page {number}: Voltage equals current multiplied by resistance.\n"
                "Resistance opposes current flow in an electrical circuit.", fontsize=15)
        original = document.tobytes()
        source.write_bytes(original)
    class Backend(FilterBackend):
        def complete(self, system, user, **kwargs):
            if json.loads(user)["page"] == 25:
                return '{"remove":[]}'
            return super().complete(system, user, **kwargs)
    progress = []
    pdf.convert_pdf(source, target, course_code="CE0055", module_number="2",
        backend=Backend(), attempts=2, progress=progress.append)
    text = target.read_text(encoding="utf-8")
    audit = json.loads(target.with_name("pdf_extraction.json").read_text(encoding="utf-8"))
    assert source.read_bytes() == original
    assert "Page 25: Voltage" in text and "Page 26: Voltage" in text
    assert len(audit) == 26 and audit[24]["filter_status"] == "review_needed"
    assert audit[24]["filter_warnings"] and not any(block["removed"] for block in audit[24]["blocks"])
    assert audit[25]["filter_status"] == "complete" and not audit[25]["filter_warnings"]
    assert "page 25" in caplog.text and any("Page 25/26:" in message for message in progress)


def test_unusable_pdf_does_not_publish_structured_text(tmp_path):
    source, target = tmp_path / "module.pdf", tmp_path / "result.txt"
    source.write_bytes(module_bytes())
    class RemoveAll:
        def complete(self, system, user, **kwargs):
            return json.dumps({"remove": [True] * len(json.loads(user)["blocks"]), "study_groups": []})
    with pytest.raises(RuntimeError, match="No readable lesson content"):
        pdf.convert_pdf(source, target, course_code="EE", module_number="1", backend=RemoveAll())
    assert not target.exists()
    assert target.with_name("pdf_extraction.json").exists()


def test_invalid_pdf_and_missing_tesseract_have_focused_errors(tmp_path, monkeypatch):
    source = tmp_path / "broken.pdf"
    source.write_bytes(b"not a PDF")
    with pytest.raises(ValueError, match="Could not open this PDF"):
        pdf.validate_pdf(source)
    monkeypatch.setattr(pdf.Path, "is_file", lambda *a: False)
    monkeypatch.setattr(pdf.shutil, "which", lambda *a: None)
    with pytest.raises(RuntimeError, match="Tesseract OCR was not found"):
        pdf.find_tesseract()


@pytest.mark.parametrize("filename,expected", [
    ("Courses/CS0003/Module 01 - Basics.pdf", ("CS0003", "01")),
    ("CS0003/M2.PDF", ("CS0003", "2")),
    ("CS0003-M3.pdf", ("CS0003", "3")),
    ("EE101/04 - circuits.pdf", ("EE101", "04")),
    ("EE101\\Module_5.pdf", ("EE101", "5")),
])
def test_folder_and_module_identity(filename, expected):
    assert pdf_identity(filename) == expected


@pytest.mark.parametrize("filename", ["../outside/M1.pdf", "/absolute/M1.pdf", "CS0003/unknown.pdf"])
def test_ambiguous_or_unsafe_pdf_names_are_rejected(filename):
    with pytest.raises(ValueError):
        pdf_identity(filename)


def test_api_folder_upload_keeps_courses_and_pdf_bytes_separate(tmp_path, monkeypatch):
    monkeypatch.setattr(api_server, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(api_server, "OUTPUT_ROOT", tmp_path / "output")
    captured = []
    monkeypatch.setattr(api_server, "run_batch", lambda items, **kwargs: captured.extend(items) or BatchResult((), ()))
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api_server.app), base_url="http://test") as client:
            result = await client.post("/process", data={"courseCode": ""}, files=[
                ("files", ("Courses/CS0003/Module 1.pdf", module_bytes(), "application/pdf")),
                ("files", ("Courses/EE101/Module 1.pdf", module_bytes(), "application/pdf"))])
            assert result.status_code == 200 and not result.json()["errors"]
    asyncio.run(scenario())
    assert [item.args.course_code for item in captured] == ["CS0003", "EE101"]
    assert all(item.args.input.suffix == ".pdf" and item.args.input.read_bytes().startswith(b"%PDF") for item in captured)
    assert len({item.paths.workspace for item in captured}) == 2


def test_folder_cli_and_ingestion_adapter_use_mistral(tmp_path, monkeypatch):
    root = tmp_path / "courses"
    for course in ["CS0003", "EE101"]:
        (root / course).mkdir(parents=True)
        for number in [2, 10]:
            (root / course / f"Module {number}.pdf").write_bytes(module_bytes())
    assert [path.name for path in course_folder_pdfs(root)][:2] == ["Module 2.pdf", "Module 10.pdf"]
    args = pipeline.parse_args([str(root), "--output-root", str(tmp_path / "output")])
    captured = []
    monkeypatch.setattr(batch_pipeline, "run_batch", lambda items, **kwargs: captured.extend(items) or BatchResult((), ()))
    assert not pipeline.run_folder(args).errors
    assert len(captured) == 4 and {item.args.course_code for item in captured} == {"CS0003", "EE101"}
    backend = FilterBackend()
    backend.close = lambda: None
    monkeypatch.setattr(batch_pipeline, "backend_from_args", lambda args: backend)
    batch_pipeline._ingest_stage(captured[0], None)
    assert captured[0].paths.structured_text.is_file()
    assert "DECORATIVE" not in captured[0].paths.structured_text.read_text(encoding="utf-8")


def test_single_pdf_cli_infers_course_and_module_before_batching(tmp_path, monkeypatch):
    folder = tmp_path / "EE101"
    folder.mkdir()
    source = folder / "Module 9.pdf"
    source.write_bytes(module_bytes())
    args = pipeline.parse_args([str(source), "--output-root", str(tmp_path / "out")])
    captured = []
    monkeypatch.setattr(batch_pipeline, "run_batch", lambda items, **kwargs: captured.extend(items) or BatchResult((), ()))
    result = pipeline.run(args)
    assert captured[0].args.course_code == "EE101"
    assert captured[0].args.module_number == "9"
    assert result == captured[0].paths


def test_ocr_settings_change_pdf_cache_identity(tmp_path):
    source = tmp_path / "Module 1.pdf"
    source.write_bytes(module_bytes())
    args = pipeline.parse_args([str(source), "--course-code", "EE101", "--module-number", "1"])
    original = batch_pipeline._manifest_settings(args)
    args.ocr_mode = "always"
    assert original != batch_pipeline._manifest_settings(args)


def test_bullet_quotes_are_grounded_and_invented_statements_are_retried():
    source = "Voltage\nElectrical potential difference\nUnit: volt (V)\nV = I R"
    responses = iter([
        {"remove": [False], "study_groups": [[99]]},
        {"remove": [False], "study_groups": [[0]]}])
    class Backend:
        def complete(self, *args, **kwargs):
            return json.dumps(next(responses))
    quotes = []
    assert pdf.filter_blocks([pdf.TextBlock(source, (0, 0, 1, 1))], Backend(),
        page_number=1, study_quotes=quotes) == [False]
    assert quotes == ["Voltage Electrical potential difference Unit: volt (V) V = I R"]


def test_page_cleanup_uses_compact_selections_and_a_small_output_budget():
    source = "Voltage equals current multiplied by resistance. " * 40
    captured = []
    class Backend:
        def complete(self, system, user, **kwargs):
            captured.append(kwargs)
            return '{"remove":[false],"study_groups":[[0]]}'
    quotes = []
    pdf.filter_blocks([pdf.TextBlock(source, (0, 0, 1, 1))], Backend(), page_number=1, study_quotes=quotes)
    assert quotes == [" ".join(source.split())]
    assert captured[0]["max_tokens"] <= 512
    assert "quotes" not in captured[0]["schema"]["properties"]
