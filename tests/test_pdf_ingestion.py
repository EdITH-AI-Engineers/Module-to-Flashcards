from pathlib import Path
import os
import sys
from types import SimpleNamespace

import pytest

from pdf_ingestion import (
    PdfExtractionError,
    _configure_tesseract,
    _open_document,
    extract_pdf_pages,
)


class FakePage:
    def __init__(self, text):
        self.text = text

    def get_text(self, kind, *, sort):
        assert kind == "text"
        assert sort is True
        return self.text


class FakeDocument:
    def __init__(self, texts):
        self.pages = [FakePage(text) for text in texts]
        self.closed = False

    @property
    def page_count(self):
        return len(self.pages)

    def load_page(self, index):
        return self.pages[index]

    def close(self):
        self.closed = True


def pdf_file(tmp_path: Path) -> Path:
    path = tmp_path / "module.pdf"
    path.write_bytes(b"fake pdf for injected document")
    return path


def test_extracts_ordered_text_pages_without_calling_ocr(tmp_path):
    source = pdf_file(tmp_path)
    document = FakeDocument(
        [
            "First page contains enough meaningful alphanumeric text for direct extraction.",
            "Second page also contains sufficient content to remain in source order.",
        ]
    )

    pages = extract_pdf_pages(
        source,
        document_factory=lambda path: document,
        ocr=lambda page, dpi: pytest.fail("OCR must not run for readable text"),
    )

    assert [page.number for page in pages] == [1, 2]
    assert [page.method for page in pages] == ["text", "text"]
    assert pages[0].text.startswith("First page")
    assert document.closed is True
    assert source.is_file()


def test_sparse_page_uses_ocr_at_requested_dpi(tmp_path):
    source = pdf_file(tmp_path)
    document = FakeDocument(["tiny"])
    calls = []

    def ocr(page, dpi):
        calls.append((page, dpi))
        return "OCR recovered a complete statement from the rendered slide."

    pages = extract_pdf_pages(
        source,
        min_chars=40,
        dpi=240,
        document_factory=lambda path: document,
        ocr=ocr,
    )

    assert pages[0].method == "ocr"
    assert pages[0].text.startswith("OCR recovered")
    assert calls == [(document.pages[0], 240)]


def test_empty_ocr_page_is_retained_as_unreadable_when_other_page_is_readable(tmp_path):
    source = pdf_file(tmp_path)
    document = FakeDocument(
        ["", "This other page contains enough readable educational content for processing."]
    )

    pages = extract_pdf_pages(
        source,
        document_factory=lambda path: document,
        ocr=lambda page, dpi: "",
    )

    assert pages[0].text == "[Unreadable Text]"
    assert pages[0].method == "ocr"
    assert pages[1].method == "text"


def test_all_unreadable_pages_fail(tmp_path):
    source = pdf_file(tmp_path)

    with pytest.raises(PdfExtractionError, match="no readable text"):
        extract_pdf_pages(
            source,
            document_factory=lambda path: FakeDocument(["", ""]),
            ocr=lambda page, dpi: "",
        )


def test_missing_file_fails_before_opening_document(tmp_path):
    missing = tmp_path / "missing.pdf"

    with pytest.raises(PdfExtractionError, match="not found"):
        extract_pdf_pages(
            missing,
            document_factory=lambda path: pytest.fail("must not open missing PDF"),
        )


def test_non_pdf_input_is_rejected(tmp_path):
    source = tmp_path / "module.txt"
    source.write_text("text", encoding="utf-8")

    with pytest.raises(PdfExtractionError, match="PDF"):
        extract_pdf_pages(source, document_factory=lambda path: FakeDocument([]))


def test_default_opener_uses_the_current_pymupdf_import_name(tmp_path, monkeypatch):
    source = pdf_file(tmp_path)
    sentinel = object()
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "pymupdf",
        SimpleNamespace(open=lambda value: calls.append(value) or sentinel),
    )
    monkeypatch.setitem(sys.modules, "fitz", None)

    assert _open_document(source) is sentinel
    assert calls == [str(source)]


def test_tesseract_uses_standard_install_when_executable_is_not_on_path(tmp_path):
    executable = tmp_path / "Tesseract-OCR" / "tesseract.exe"
    executable.parent.mkdir()
    executable.write_bytes(b"exe")
    module = SimpleNamespace(
        pytesseract=SimpleNamespace(tesseract_cmd="tesseract")
    )

    _configure_tesseract(
        module,
        candidates=(executable,),
        path_lookup=lambda name: None,
    )

    assert module.pytesseract.tesseract_cmd == str(executable)


def test_tesseract_uses_explicit_portable_runtime_before_path(tmp_path, monkeypatch):
    executable = tmp_path / "tesseract" / "tesseract.exe"
    tessdata = tmp_path / "tesseract" / "tessdata"
    executable.parent.mkdir()
    executable.write_bytes(b"exe")
    tessdata.mkdir()
    module = SimpleNamespace(pytesseract=SimpleNamespace(tesseract_cmd="tesseract"))
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)

    _configure_tesseract(
        module,
        portable_executable=executable,
        tessdata_dir=tessdata,
        path_lookup=lambda name: "C:/system/tesseract.exe",
    )

    assert module.pytesseract.tesseract_cmd == str(executable)
    assert os.environ["TESSDATA_PREFIX"] == str(tessdata)
