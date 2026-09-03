from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable

from structured_module import UNREADABLE


class PdfExtractionError(RuntimeError):
    """Raised when a PDF cannot yield usable local text."""


@dataclass(frozen=True)
class ExtractedPage:
    number: int
    text: str
    method: str


def _clean_text(value: object) -> str:
    text = str(value or "").replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _meaningful_count(text: str) -> int:
    return sum(character.isalnum() for character in text)


def _open_document(path: Path) -> Any:
    try:
        import pymupdf
    except ImportError as exc:
        raise PdfExtractionError(
            "PyMuPDF is not installed; run python -m pip install -r requirements.txt"
        ) from exc
    try:
        return pymupdf.open(str(path))
    except Exception as exc:
        raise PdfExtractionError(f"could not open PDF {path}: {exc}") from exc


def _configure_tesseract(
    pytesseract_module: Any,
    *,
    candidates: tuple[Path, ...] | None = None,
    path_lookup: Callable[[str], str | None] = shutil.which,
) -> None:
    if path_lookup("tesseract"):
        return
    if candidates is None:
        program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        local_app_data = Path(os.environ.get("LOCALAPPDATA", ""))
        candidates = (
            program_files / "Tesseract-OCR" / "tesseract.exe",
            local_app_data / "Programs" / "Tesseract-OCR" / "tesseract.exe",
        )
    for candidate in candidates:
        if candidate.is_file():
            pytesseract_module.pytesseract.tesseract_cmd = str(candidate)
            return


def _ocr_page(page: Any, dpi: int) -> str:
    try:
        from PIL import Image
        import pytesseract
    except ImportError as exc:
        raise PdfExtractionError(
            "OCR dependencies are missing; run python -m pip install -r requirements.txt"
        ) from exc

    _configure_tesseract(pytesseract)
    try:
        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
        image = Image.open(BytesIO(pixmap.tobytes("png")))
        return str(pytesseract.image_to_string(image))
    except Exception as exc:
        if exc.__class__.__name__ == "TesseractNotFoundError" or "tesseract" in str(exc).casefold():
            raise PdfExtractionError(
                "Tesseract OCR is not installed or is not on PATH; install Tesseract and retry"
            ) from exc
        raise PdfExtractionError(f"local OCR failed: {exc}") from exc


def extract_pdf_pages(
    path: Path,
    *,
    min_chars: int = 40,
    dpi: int = 200,
    document_factory: Callable[[Path], Any] | None = None,
    ocr: Callable[[Any, int], str] | None = None,
) -> tuple[ExtractedPage, ...]:
    source = Path(path)
    if not source.is_file():
        raise PdfExtractionError(f"PDF not found: {source}")
    if source.suffix.casefold() != ".pdf":
        raise PdfExtractionError(f"input must be a PDF file: {source}")
    if min_chars < 1:
        raise ValueError("min_chars must be at least 1")
    if dpi < 72:
        raise ValueError("dpi must be at least 72")

    opener = document_factory or _open_document
    ocr_reader = ocr or _ocr_page
    document = opener(source)
    pages: list[ExtractedPage] = []
    try:
        page_count = int(getattr(document, "page_count", 0))
        for index in range(page_count):
            page = document.load_page(index)
            direct_text = _clean_text(page.get_text("text", sort=True))
            if _meaningful_count(direct_text) >= min_chars:
                pages.append(ExtractedPage(index + 1, direct_text, "text"))
                continue

            ocr_text = _clean_text(ocr_reader(page, dpi))
            if _meaningful_count(ocr_text) > 0:
                pages.append(ExtractedPage(index + 1, ocr_text, "ocr"))
            elif _meaningful_count(direct_text) > 0:
                pages.append(ExtractedPage(index + 1, direct_text, "text"))
            else:
                pages.append(ExtractedPage(index + 1, UNREADABLE, "ocr"))
    finally:
        close = getattr(document, "close", None)
        if callable(close):
            close()

    if not pages:
        raise PdfExtractionError(f"PDF contains no pages: {source}")
    if not any(page.text != UNREADABLE for page in pages):
        raise PdfExtractionError(f"PDF contains no readable text after local OCR: {source}")
    return tuple(pages)
