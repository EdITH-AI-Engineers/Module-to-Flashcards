from pathlib import Path

import pytest

import slides_pdf_to_txt
from pdf_ingestion import ExtractedPage, PdfExtractionError


def test_parse_args_defaults_to_8k_context():
    assert slides_pdf_to_txt.parse_args(["module.pdf"]).n_ctx == 8192


def test_parse_args_preserves_pdf_identity_and_model_options():
    args = slides_pdf_to_txt.parse_args(
        [
            "module.pdf",
            "--course-code",
            "CPE0021",
            "--module-number",
            "01",
            "--module-title",
            "Architecture",
            "--n-gpu-layers",
            "0",
            "--ocr-dpi",
            "240",
        ]
    )

    assert args.pdf == Path("module.pdf")
    assert args.course_code == "CPE0021"
    assert args.module_number == "01"
    assert args.module_title == "Architecture"
    assert args.n_gpu_layers == 0
    assert args.ocr_dpi == 240


def test_run_extracts_before_loading_qwen_and_writes_default_output(
    tmp_path, monkeypatch
):
    source = tmp_path / "module.pdf"
    source.write_bytes(b"pdf")
    monkeypatch.chdir(tmp_path)
    events = []
    pages = (ExtractedPage(1, "Readable page text", "text"),)

    def extract(path, **kwargs):
        events.append("extract")
        return pages

    def ensure(path):
        events.append("model")
        return tmp_path / "model.gguf"

    class Backend:
        def __init__(self, model_path, **kwargs):
            events.append("backend")
            assert model_path == tmp_path / "model.gguf"
            assert kwargs["n_gpu_layers"] == 0

    sentinel = object()

    def normalize(backend, page_values, **kwargs):
        events.append("normalize")
        assert page_values == pages
        assert kwargs["module_number"] == "01"
        return sentinel

    monkeypatch.setattr(slides_pdf_to_txt, "extract_pdf_pages", extract)
    monkeypatch.setattr(slides_pdf_to_txt, "ensure_model", ensure)
    monkeypatch.setattr(slides_pdf_to_txt, "LocalQwenBackend", Backend)
    monkeypatch.setattr(slides_pdf_to_txt, "normalize_document", normalize)
    monkeypatch.setattr(
        slides_pdf_to_txt,
        "render_structured_module",
        lambda module: "[MODULE]\n[/MODULE]\n",
    )
    args = slides_pdf_to_txt.parse_args(
        [str(source), "--module-number", "01", "--n-gpu-layers", "0"]
    )

    output = slides_pdf_to_txt.run(args)

    assert output == Path("structured_text") / "module.txt"
    assert output.read_text(encoding="utf-8") == "[MODULE]\n[/MODULE]\n"
    assert events == ["extract", "model", "backend", "normalize"]
    assert source.is_file()
    assert not list(output.parent.glob("*.tmp"))


def test_run_uses_explicit_output_and_backend_configuration(tmp_path, monkeypatch):
    source = tmp_path / "module.pdf"
    source.write_bytes(b"pdf")
    output = tmp_path / "nested" / "structured.txt"
    captured = {}
    monkeypatch.setattr(
        slides_pdf_to_txt,
        "extract_pdf_pages",
        lambda *args, **kwargs: (ExtractedPage(1, "Page text", "text"),),
    )
    monkeypatch.setattr(
        slides_pdf_to_txt, "ensure_model", lambda path: tmp_path / "qwen.gguf"
    )

    class Backend:
        def __init__(self, model_path, **kwargs):
            captured.update(model_path=model_path, **kwargs)

    monkeypatch.setattr(slides_pdf_to_txt, "LocalQwenBackend", Backend)
    monkeypatch.setattr(slides_pdf_to_txt, "normalize_document", lambda *a, **k: object())
    monkeypatch.setattr(slides_pdf_to_txt, "render_structured_module", lambda value: "ok\n")
    args = slides_pdf_to_txt.parse_args(
        [
            str(source),
            "--output",
            str(output),
            "--n-ctx",
            "8192",
            "--seed",
            "9",
            "--n-gpu-layers",
            "4",
        ]
    )

    assert slides_pdf_to_txt.run(args) == output
    assert output.read_text(encoding="utf-8") == "ok\n"
    assert captured == {
        "model_path": tmp_path / "qwen.gguf",
        "n_ctx": 8192,
        "n_gpu_layers": 4,
        "seed": 9,
    }


def test_invalid_pdf_fails_before_model_download(tmp_path, monkeypatch):
    source = tmp_path / "bad.pdf"
    source.write_bytes(b"bad")
    monkeypatch.setattr(
        slides_pdf_to_txt,
        "extract_pdf_pages",
        lambda *args, **kwargs: (_ for _ in ()).throw(PdfExtractionError("bad PDF")),
    )
    monkeypatch.setattr(
        slides_pdf_to_txt,
        "ensure_model",
        lambda path: pytest.fail("model must not load for an invalid PDF"),
    )

    with pytest.raises(PdfExtractionError, match="bad PDF"):
        slides_pdf_to_txt.run(slides_pdf_to_txt.parse_args([str(source)]))


def test_main_reports_focused_error(capsys):
    exit_code = slides_pdf_to_txt.main(["missing.pdf"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.startswith("Error: PDF not found:")
