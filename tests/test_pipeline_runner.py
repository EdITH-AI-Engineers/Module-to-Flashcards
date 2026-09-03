import json
from pathlib import Path
import subprocess

import pytest

import pipeline
from structured_module import StructuredModule, StructuredSlide, render_structured_module


def test_parse_args_defaults_to_8k_context(tmp_path):
    assert pipeline.parse_args(
        [str(tmp_path / "module.pdf"), "--course-code", "CPE0021", "--module-number", "1"]
    ).n_ctx == 8192


def test_timeout_defaults_to_disabled(tmp_path):
    assert make_args(tmp_path).timeout == 0


def make_args(tmp_path, *extra):
    source = tmp_path / "Module One.pdf"
    source.write_bytes(b"pdf")
    return pipeline.parse_args(
        [
            str(source),
            "--course-code",
            "CPE0021",
            "--module-number",
            "01",
            "--output-root",
            str(tmp_path / "pipeline output"),
            *extra,
        ]
    )


def structured_content():
    return render_structured_module(
        StructuredModule(
            course_code="CPE0021",
            module_number="01",
            module_title="Architecture",
            source_file="Module One.pdf",
            slides=(
                StructuredSlide(
                    number=1,
                    extraction_method="text",
                    title="Processor",
                    content=("A processor executes instructions.",),
                    visual_text=("Not Specified",),
                    definitions=(),
                    knowledge_statements=("A processor contains an ALU.",),
                    brief_explanation="The page describes a processor.",
                ),
            ),
        )
    )


def graph_content():
    return json.dumps(
        {
            "metadata": {
                "course_code": "CPE0021",
                "module_number": "01",
            },
            "nodes": [],
            "edges": [
                {
                    "id": "e1",
                    "subject": "processor",
                    "relation": "contains",
                    "object": "ALU",
                }
            ],
        }
    )


def materialize(stage, paths):
    if stage == "pdf-to-text":
        paths.structured_text.parent.mkdir(parents=True, exist_ok=True)
        paths.structured_text.write_text(structured_content(), encoding="utf-8")
    elif stage == "knowledge-graph":
        paths.graph_dir.mkdir(parents=True, exist_ok=True)
        paths.graph_json.write_text(graph_content(), encoding="utf-8")
        paths.triples_csv.write_text("subject,relation,object\n", encoding="utf-8")
    elif stage == "flashcards":
        paths.flashcards.parent.mkdir(parents=True, exist_ok=True)
        paths.flashcards.write_text("Module 01.1\nModule 01.2\n", encoding="utf-8")


def test_pipeline_paths_use_a_sanitized_per_pdf_workspace(tmp_path):
    source = tmp_path / "Module: 1?.pdf"
    paths = pipeline.pipeline_paths(source, tmp_path / "outputs")

    assert paths.workspace == tmp_path / "outputs" / "Module_1"
    assert paths.structured_text == paths.workspace / "structured_module.txt"
    assert paths.graph_json == paths.workspace / "knowledge_graph" / "knowledge_graph.json"
    assert paths.flashcards == paths.workspace / "flashcards.txt"


def test_build_stage_commands_use_current_python_and_absolute_artifacts(tmp_path):
    args = make_args(
        tmp_path,
        "--n-gpu-layers",
        "0",
        "--kg-device",
        "cpu",
        "--kg-batch-size",
        "1",
        "--kg-num-beams",
        "1",
    )
    paths = pipeline.pipeline_paths(args.pdf, args.output_root)

    commands = pipeline.build_stage_commands(args, paths)

    assert [item.name for item in commands] == [
        "pdf-to-text",
        "knowledge-graph",
        "flashcards",
    ]
    assert commands[0].command[0] == pipeline.sys.executable
    assert Path(commands[0].command[1]).name == "slides_pdf_to_txt.py"
    assert str(paths.structured_text.resolve()) in commands[0].command
    assert "--course-code" in commands[0].command
    assert "CPE0021" in commands[0].command
    assert Path(commands[1].command[1]).name == "text-extractor.py"
    assert "cpu" in commands[1].command
    assert "--batch-size" in commands[1].command
    assert commands[1].command[commands[1].command.index("--batch-size") + 1] == "1"
    assert "--num-beams" in commands[1].command
    assert commands[1].command[commands[1].command.index("--num-beams") + 1] == "1"
    assert Path(commands[2].command[1]).name == "main.py"
    assert str(paths.graph_json.resolve()) in commands[2].command


def test_run_executes_all_stages_in_order_when_artifacts_are_missing(tmp_path):
    args = make_args(tmp_path)
    paths = pipeline.pipeline_paths(args.pdf, args.output_root)
    calls = []

    def command_runner(command, **kwargs):
        stage = {
            "slides_pdf_to_txt.py": "pdf-to-text",
            "text-extractor.py": "knowledge-graph",
            "main.py": "flashcards",
        }[Path(command[1]).name]
        calls.append(stage)
        materialize(stage, paths)
        return subprocess.CompletedProcess(command, 0)

    result = pipeline.run(args, command_runner=command_runner)

    assert result == paths
    assert calls == ["pdf-to-text", "knowledge-graph", "flashcards"]


def test_zero_timeout_does_not_pass_subprocess_timeout(tmp_path):
    args = make_args(tmp_path)
    paths = pipeline.pipeline_paths(args.pdf, args.output_root)
    calls = []

    def runner(command, **kwargs):
        calls.append(kwargs)
        stage = {
            "slides_pdf_to_txt.py": "pdf-to-text",
            "text-extractor.py": "knowledge-graph",
            "main.py": "flashcards",
        }[Path(command[1]).name]
        materialize(stage, paths)
        return subprocess.CompletedProcess(command, 0)

    pipeline.run(args, command_runner=runner)

    assert all("timeout" not in kwargs for kwargs in calls)


def test_negative_timeout_fails_before_running_stages(tmp_path):
    args = make_args(tmp_path, "--timeout", "-1")

    with pytest.raises(pipeline.PipelineRunError, match="must not be negative"):
        pipeline.run(args, command_runner=lambda *a, **k: pytest.fail("must not run"))


def test_run_reuses_all_valid_artifacts_without_subprocesses(tmp_path):
    args = make_args(tmp_path)
    paths = pipeline.pipeline_paths(args.pdf, args.output_root)
    for stage in ("pdf-to-text", "knowledge-graph", "flashcards"):
        materialize(stage, paths)

    result = pipeline.run(
        args,
        command_runner=lambda *a, **k: pytest.fail("valid artifacts must be reused"),
    )

    assert result == paths


def test_force_recomputes_every_stage(tmp_path):
    args = make_args(tmp_path, "--force")
    paths = pipeline.pipeline_paths(args.pdf, args.output_root)
    for stage in ("pdf-to-text", "knowledge-graph", "flashcards"):
        materialize(stage, paths)
    calls = []

    def command_runner(command, **kwargs):
        stage = {
            "slides_pdf_to_txt.py": "pdf-to-text",
            "text-extractor.py": "knowledge-graph",
            "main.py": "flashcards",
        }[Path(command[1]).name]
        calls.append(stage)
        materialize(stage, paths)
        return subprocess.CompletedProcess(command, 0)

    pipeline.run(args, command_runner=command_runner)

    assert calls == ["pdf-to-text", "knowledge-graph", "flashcards"]


def test_stage_failure_stops_the_sequence(tmp_path):
    args = make_args(tmp_path)
    calls = []

    def command_runner(command, **kwargs):
        calls.append(Path(command[1]).name)
        raise subprocess.CalledProcessError(1, command)

    with pytest.raises(pipeline.PipelineRunError, match="pdf-to-text"):
        pipeline.run(args, command_runner=command_runner)

    assert calls == ["slides_pdf_to_txt.py"]


def test_help_documents_artifacts_resume_and_force(capsys):
    with pytest.raises(SystemExit) as exc_info:
        pipeline.parse_args(["--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "source slide PDF" in help_text
    assert "--course-code" in help_text
    assert "--module-number" in help_text
    assert "structured_module.txt" in help_text
    assert "knowledge_graph.json" in help_text
    assert "flashcards.txt" in help_text
    assert "resume" in help_text.casefold()
    assert "--force" in help_text
