import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import text_extractor
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


def test_local_rebel_load_never_uses_external_model_cache(tmp_path, monkeypatch):
    calls = []

    class FakeModel:
        def to(self, device):
            calls.append(("to", device))
            return self

        def eval(self):
            calls.append(("eval",))
            return self

    class FakeLoader:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            calls.append((cls.__name__, model, kwargs))
            return FakeModel() if cls is FakeModelLoader else object()

    class FakeTokenizerLoader(FakeLoader):
        pass

    class FakeModelLoader(FakeLoader):
        pass

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    fake_transformers = SimpleNamespace(
        AutoTokenizer=FakeTokenizerLoader,
        AutoModelForSeq2SeqLM=FakeModelLoader,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    model_dir = tmp_path / "rebel-large"

    runtime = text_extractor.load_runtime(
        model_dir,
        "cpu",
        local_files_only=True,
    )

    assert runtime.device == "cpu"
    assert calls == [
        ("FakeTokenizerLoader", model_dir, {"local_files_only": True}),
        ("FakeModelLoader", model_dir, {"local_files_only": True}),
        ("to", "cpu"),
        ("eval",),
    ]


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


def test_build_graph_includes_normalized_lesson_facts(tmp_path):
    extractor = load_extractor()
    lesson_facts = (
        {
            "id": "f1",
            "statement": "A processor executes instructions.",
            "slides": [1],
            "kind": "knowledge_statement",
            "topic": "Processor",
        },
    )

    graph = extractor.build_graph(
        [],
        tmp_path / "structured_module.txt",
        "Babelscape/rebel-large",
        lesson_facts=lesson_facts,
    )

    assert graph["facts"] == list(lesson_facts)
    assert graph["metadata"]["fact_count"] == 1


def test_save_outputs_serializes_portable_model_path(tmp_path):
    extractor = load_extractor()
    model_path = tmp_path / "models" / "rebel-large"
    graph = extractor.build_graph(
        [],
        tmp_path / "structured_module.txt",
        model_path,
    )
    output_dir = tmp_path / "graph"

    extractor.save_outputs(graph, output_dir)

    saved = json.loads((output_dir / "knowledge_graph.json").read_text("utf-8"))
    assert saved["metadata"]["model"] == str(model_path)


def test_plain_text_input_remains_supported():
    extractor = load_extractor()
    plain = "Processor\r\n\r\nA processor executes instructions."

    prepared, metadata = extractor.prepare_input_text(plain)

    assert prepared == "Processor\n\nA processor executes instructions."
    assert metadata == {}


def test_run_reuses_injected_runtime(tmp_path, monkeypatch):
    """Catches run loading or releasing a runtime supplied by its caller."""
    source = tmp_path / "module.txt"
    source.write_text(structured_text(), encoding="utf-8")
    output_dir = tmp_path / "graph"

    class FakeTokenizer:
        def encode(self, text, *, add_special_tokens):
            return [1]

        def decode(self, ids, *, skip_special_tokens):
            return "processor contains ALU"

    tokenizer = FakeTokenizer()
    runtime = text_extractor.RebelRuntime(tokenizer, object(), "cpu")
    monkeypatch.setattr(
        text_extractor,
        "load_runtime",
        lambda *a, **k: pytest.fail("injected runtime must be reused"),
    )
    monkeypatch.setattr(
        text_extractor,
        "release_runtime",
        lambda *a, **k: pytest.fail("injected runtime remains caller-owned"),
    )
    triples = [
        {
            "subject": "processor",
            "relation": "contains",
            "object": "ALU",
            "evidence": [],
        }
    ]
    monkeypatch.setattr(text_extractor, "extract_relations", lambda *a, **k: triples)
    args = text_extractor.parse_args(
        [
            str(source),
            "--output-dir",
            str(output_dir),
            "--batch-size",
            "1",
            "--num-beams",
            "1",
        ]
    )

    json_path, csv_path = text_extractor.run(args, runtime=runtime)

    assert json_path.is_file()
    assert csv_path.is_file()
    saved_graph = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved_graph["metadata"]["fact_count"] == 1
    assert saved_graph["facts"][0]["statement"] == (
        "A processor contains an arithmetic logic unit."
    )
    assert args.batch_size == 1
    assert args.num_beams == 1


@pytest.mark.parametrize("save_error", (None, OSError("output failed")))
def test_run_releases_an_internally_owned_runtime_on_success_and_output_failure(
    tmp_path, monkeypatch, save_error
):
    source = tmp_path / "module.txt"
    source.write_text(structured_text(), encoding="utf-8")
    runtime = text_extractor.RebelRuntime(object(), object(), "cpu")
    released = []
    args = text_extractor.parse_args(
        [str(source), "--output-dir", str(tmp_path / "graph"), "--batch-size", "1"]
    )

    monkeypatch.setattr(text_extractor, "load_runtime", lambda *a, **k: runtime)
    monkeypatch.setattr(text_extractor, "release_runtime", released.append)
    monkeypatch.setattr(text_extractor, "make_chunks", lambda *a, **k: [{"id": 1}])
    monkeypatch.setattr(text_extractor, "extract_relations", lambda *a, **k: [])
    if save_error is not None:
        monkeypatch.setattr(
            text_extractor,
            "save_outputs",
            lambda *a, **k: (_ for _ in ()).throw(save_error),
        )
        with pytest.raises(OSError, match="output failed"):
            text_extractor.run(args)
    else:
        monkeypatch.setattr(text_extractor, "save_outputs", lambda *a, **k: None)
        assert text_extractor.run(args) == (
            args.output_dir / "knowledge_graph.json",
            args.output_dir / "triples.csv",
        )

    assert released == [runtime]
