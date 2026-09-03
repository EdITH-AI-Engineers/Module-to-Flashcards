import importlib.util
from pathlib import Path

from structured_module import (
    StructuredModule,
    StructuredSlide,
    render_structured_module,
)


def load_extractor():
    path = Path(__file__).parents[1] / "text-extractor.py"
    spec = importlib.util.spec_from_file_location("text_extractor_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def structured_text():
    module = StructuredModule(
        course_code="CPE0021",
        module_number="01",
        module_title="Architecture",
        source_file="slides.pdf",
        slides=(
            StructuredSlide(
                number=1,
                extraction_method="text",
                title="Processor",
                content=("A processor executes instructions.",),
                visual_text=("Not Specified",),
                definitions=(),
                knowledge_statements=(
                    "A processor contains an arithmetic logic unit.",
                ),
                brief_explanation="The slide identifies a processor component.",
            ),
        ),
    )
    return render_structured_module(module)


def test_prepare_input_projects_structured_content_and_metadata():
    extractor = load_extractor()

    prepared, metadata = extractor.prepare_input_text(structured_text())

    assert "[MODULE]" not in prepared
    assert "[CONTENT]" not in prepared
    assert "A processor executes instructions." in prepared
    assert metadata["course_code"] == "CPE0021"
    assert metadata["module_number"] == "01"
    assert metadata["module_title"] == "Architecture"
    assert metadata["source_file"] == "slides.pdf"
    assert metadata["format_version"] == "1"


def test_build_graph_propagates_structured_module_metadata(tmp_path):
    extractor = load_extractor()
    triples = [
        {
            "subject": "processor",
            "relation": "contains",
            "object": "ALU",
            "evidence": [],
        }
    ]

    graph = extractor.build_graph(
        triples,
        tmp_path / "structured_module.txt",
        "Babelscape/rebel-large",
        module_metadata={
            "format_version": "1",
            "course_code": "CPE0021",
            "module_number": "01",
            "module_title": "Architecture",
            "source_file": "slides.pdf",
        },
    )

    assert graph["metadata"]["course_code"] == "CPE0021"
    assert graph["metadata"]["module_number"] == "01"
    assert graph["metadata"]["module_title"] == "Architecture"
    assert graph["metadata"]["source_file"] == "slides.pdf"


def test_plain_text_input_remains_supported():
    extractor = load_extractor()
    plain = "Processor\r\n\r\nA processor executes instructions."

    prepared, metadata = extractor.prepare_input_text(plain)

    assert prepared == "Processor\n\nA processor executes instructions."
    assert metadata == {}
