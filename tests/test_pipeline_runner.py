import json
from pathlib import Path
import subprocess

import pytest

import pipeline
from flashcard_csv import render_module, render_module_parts, write_module_parts
from flashcard_types import ModuleIdentity
from structured_module import StructuredModule, StructuredSlide, render_structured_module
from tests.batch_helpers import flashcard_content
from tests.factories import valid_clusters


def test_parse_args_defaults_to_12k_context(tmp_path):
    assert pipeline.parse_args(
        [str(tmp_path / "module.txt"), "--course-code", "CPE0021", "--module-number", "1"]
    ).n_ctx == 12288


def test_timeout_defaults_to_disabled(tmp_path):
    assert make_args(tmp_path).timeout == 0


def make_args(tmp_path, *extra):
    source = tmp_path / "Module One.txt"
    source.write_text(structured_content(), encoding="utf-8")
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
            source_file="Module One.txt",
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


def checked_graph_content():
    graph = json.loads(graph_content())
    graph["metadata"]["graph_checker"] = {"status": "checked", "model": "test"}
    return json.dumps(graph)


def materialize(stage, paths):
    if stage == "knowledge-graph":
        paths.unchecked_graph_dir.mkdir(parents=True, exist_ok=True)
        paths.unchecked_graph_json.write_text(graph_content(), encoding="utf-8")
        paths.unchecked_triples_csv.write_text(
            "subject,relation,object\n", encoding="utf-8"
        )
    elif stage == "flashcards":
        paths.graph_dir.mkdir(parents=True, exist_ok=True)
        paths.graph_json.write_text(checked_graph_content(), encoding="utf-8")
        identity = ModuleIdentity("CPE0021", "01")
        write_module_parts(
            paths.flashcards,
            render_module_parts(identity, valid_clusters()),
            identity,
        )


def test_pipeline_paths_use_a_sanitized_per_module_workspace(tmp_path):
    source = tmp_path / "Module: 1?.txt"
    paths = pipeline.pipeline_paths(source, tmp_path / "outputs", "CPE0021", "01")

    assert paths.workspace == tmp_path / "outputs" / "CPE0021" / "Module_1"
    assert paths.structured_text == paths.workspace / "structured_module.txt"
    assert paths.graph_json == paths.workspace / "knowledge_graph" / "knowledge_graph.json"
    assert paths.flashcards == (
        tmp_path / "flashcards" / "CPE0021" / "CPE0021_M1.csv"
    )
    assert paths.flashcard_parts == (
        tmp_path / "flashcards" / "CPE0021" / "CPE0021_M1-1.csv",
        tmp_path / "flashcards" / "CPE0021" / "CPE0021_M1-2.csv",
    )
    assert paths.flashcard_receipt.name == "CPE0021_M1.parts.json"


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
    paths = pipeline.pipeline_paths(
        args.input, args.output_root, args.course_code, args.module_number
    )

    commands = pipeline.build_stage_commands(args, paths)

    assert [item.name for item in commands] == [
        "knowledge-graph",
        "flashcards",
    ]
    assert commands[0].command[0] == pipeline.sys.executable
    assert Path(commands[0].command[1]).name == "text-extractor.py"
    assert str(paths.structured_text.resolve()) in commands[0].command
    assert str(paths.unchecked_graph_dir.resolve()) in commands[0].command
    assert commands[0].command[
        commands[0].command.index("--course-code") + 1
    ] == "CPE0021"
    assert commands[0].command[
        commands[0].command.index("--module-number") + 1
    ] == "01"
    assert "cpu" in commands[0].command
    assert "--batch-size" in commands[0].command
    assert commands[0].command[commands[0].command.index("--batch-size") + 1] == "1"
    assert "--num-beams" in commands[0].command
    assert commands[0].command[commands[0].command.index("--num-beams") + 1] == "1"
    assert Path(commands[1].command[1]).name == "main.py"
    assert str(paths.graph_json.resolve()) in commands[1].command
    unchecked_index = commands[1].command.index("--unchecked-graph") + 1
    assert commands[1].command[unchecked_index] == str(
        paths.unchecked_graph_json.resolve()
    )
    corpus_index = commands[1].command.index("--course-corpus") + 1
    assert commands[1].command[corpus_index] == str(
        (paths.flashcards.parent / "course_corpus.json").resolve()
    )


def test_run_executes_all_stages_in_order_when_artifacts_are_missing(tmp_path):
    args = make_args(tmp_path)
    paths = pipeline.pipeline_paths(
        args.input, args.output_root, args.course_code, args.module_number
    )
    calls = []

    def command_runner(command, **kwargs):
        stage = {
            "text-extractor.py": "knowledge-graph",
            "main.py": "flashcards",
        }[Path(command[1]).name]
        calls.append(stage)
        materialize(stage, paths)
        return subprocess.CompletedProcess(command, 0)

    result = pipeline.run(args, command_runner=command_runner)

    assert result == paths
    assert calls == ["knowledge-graph", "flashcards"]
    assert paths.structured_text.read_text(encoding="utf-8") == structured_content()


def test_zero_timeout_does_not_pass_subprocess_timeout(tmp_path):
    args = make_args(tmp_path)
    paths = pipeline.pipeline_paths(
        args.input, args.output_root, args.course_code, args.module_number
    )
    calls = []

    def runner(command, **kwargs):
        calls.append(kwargs)
        stage = {
            "text-extractor.py": "knowledge-graph",
            "main.py": "flashcards",
        }[Path(command[1]).name]
        materialize(stage, paths)
        return subprocess.CompletedProcess(command, 0)

    pipeline.run(args, command_runner=runner)

    assert all("timeout" not in kwargs for kwargs in calls)


def test_positive_timeout_passes_the_remaining_budget_to_each_subprocess(
    tmp_path, monkeypatch
):
    args = make_args(tmp_path, "--timeout", "5")
    paths = pipeline.pipeline_paths(
        args.input, args.output_root, args.course_code, args.module_number
    )
    clock_values = iter((100.0, 101.0, 102.0))
    timeouts = []

    def runner(command, **kwargs):
        timeouts.append(kwargs["timeout"])
        stage = {
            "text-extractor.py": "knowledge-graph",
            "main.py": "flashcards",
        }[Path(command[1]).name]
        materialize(stage, paths)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(pipeline.time, "monotonic", lambda: next(clock_values))

    pipeline.run(args, command_runner=runner)

    assert timeouts == [4.0, 3.0]


def test_subprocess_timeout_keeps_the_stage_specific_error(tmp_path):
    args = make_args(tmp_path, "--timeout", "5")

    def runner(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    with pytest.raises(
        pipeline.PipelineRunError,
        match=r"knowledge-graph stage exceeded the 5-second timeout",
    ):
        pipeline.run(args, command_runner=runner)


def test_negative_timeout_fails_before_running_stages(tmp_path):
    args = make_args(tmp_path, "--timeout", "-1")

    with pytest.raises(pipeline.PipelineRunError, match="must not be negative"):
        pipeline.run(args, command_runner=lambda *a, **k: pytest.fail("must not run"))


def test_run_rejects_non_txt_input_before_running_stages(tmp_path):
    source = tmp_path / "module.pdf"
    source.write_text(structured_content(), encoding="utf-8")
    args = pipeline.parse_args(
        [
            str(source),
            "--course-code",
            "CPE0021",
            "--module-number",
            "01",
        ]
    )

    with pytest.raises(pipeline.PipelineRunError, match=r"must be a \.txt file"):
        pipeline.run(args, command_runner=lambda *a, **k: pytest.fail("must not run"))


def test_run_reuses_all_valid_artifacts_without_subprocesses(tmp_path):
    args = make_args(tmp_path)
    paths = pipeline.pipeline_paths(
        args.input, args.output_root, args.course_code, args.module_number
    )
    for stage in ("knowledge-graph", "flashcards"):
        materialize(stage, paths)
    pipeline.stage_structured_module(
        args.input, paths.structured_text,
        course_code=args.course_code, module_number=args.module_number,
    )
    pipeline._write_pipeline_manifest(
        paths, pipeline._pipeline_manifest_contents(args, args.input)
    )

    result = pipeline.run(
        args,
        command_runner=lambda *a, **k: pytest.fail("valid artifacts must be reused"),
    )

    assert result == paths


def test_unchanged_legacy_output_converts_without_subprocess(tmp_path):
    args = make_args(tmp_path)
    paths = pipeline.pipeline_paths(
        args.input, args.output_root, args.course_code, args.module_number
    )
    pipeline.stage_structured_module(
        args.input, paths.structured_text,
        course_code=args.course_code, module_number=args.module_number,
    )
    materialize("knowledge-graph", paths)
    materialize("flashcards", paths)
    for path in (*paths.flashcard_parts, paths.flashcard_receipt):
        path.unlink()
    legacy = render_module(ModuleIdentity("CPE0021", "01"), valid_clusters())
    paths.flashcards.write_text(legacy, encoding="utf-8-sig")

    pipeline.run(
        args,
        command_runner=lambda *a, **k: pytest.fail("unchanged legacy artifact must convert"),
    )

    assert pipeline._valid_flashcards(paths.flashcards, "01", "CPE0021")
    assert paths.flashcards.read_text(encoding="utf-8-sig") == legacy


def test_legacy_migration_rejects_changed_source(tmp_path):
    args = make_args(tmp_path)
    paths = pipeline.pipeline_paths(
        args.input, args.output_root, args.course_code, args.module_number
    )
    pipeline.stage_structured_module(
        args.input, paths.structured_text,
        course_code=args.course_code, module_number=args.module_number,
    )
    materialize("knowledge-graph", paths)
    materialize("flashcards", paths)
    for path in (*paths.flashcard_parts, paths.flashcard_receipt):
        path.unlink()
    paths.flashcards.write_text(
        render_module(ModuleIdentity("CPE0021", "01"), valid_clusters()),
        encoding="utf-8-sig",
    )
    args.input.write_text(args.input.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    calls = []

    def runner(command, **kwargs):
        stage = Path(command[1]).name
        calls.append(stage)
        materialize("knowledge-graph" if stage == "text-extractor.py" else "flashcards", paths)
        return subprocess.CompletedProcess(command, 0)

    pipeline.run(args, command_runner=runner)

    assert calls == ["text-extractor.py", "main.py"]


@pytest.mark.parametrize("change", ["source", "seed"])
def test_changed_source_or_settings_recomputes_existing_pair(tmp_path, change):
    args = make_args(tmp_path)
    paths = pipeline.pipeline_paths(
        args.input, args.output_root, args.course_code, args.module_number
    )
    materialize("knowledge-graph", paths)
    materialize("flashcards", paths)
    pipeline.stage_structured_module(
        args.input, paths.structured_text,
        course_code=args.course_code, module_number=args.module_number,
    )
    pipeline._write_pipeline_manifest(
        paths, pipeline._pipeline_manifest_contents(args, args.input)
    )
    pipeline.run(args, command_runner=lambda *a, **k: pytest.fail("initial artifacts must reuse"))
    if change == "source":
        args.input.write_text(args.input.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    else:
        args.seed = 99
    calls = []

    def runner(command, **kwargs):
        stage = Path(command[1]).name
        calls.append(stage)
        materialize("knowledge-graph" if stage == "text-extractor.py" else "flashcards", paths)
        return subprocess.CompletedProcess(command, 0)

    pipeline.run(args, command_runner=runner)

    assert calls == ["text-extractor.py", "main.py"]


def test_retained_legacy_backup_never_replaces_newer_subprocess_pair(tmp_path):
    args = make_args(tmp_path)
    paths = pipeline.pipeline_paths(
        args.input, args.output_root, args.course_code, args.module_number
    )
    pipeline.stage_structured_module(
        args.input, paths.structured_text,
        course_code=args.course_code, module_number=args.module_number,
    )
    materialize("knowledge-graph", paths)
    materialize("flashcards", paths)
    for path in (*paths.flashcard_parts, paths.flashcard_receipt):
        path.unlink()
    paths.flashcards.write_text(
        render_module(ModuleIdentity("CPE0021", "01"), valid_clusters()),
        encoding="utf-8-sig",
    )
    pipeline.run(args, command_runner=lambda *a, **k: pytest.fail("convert A"))
    args.input.write_text(args.input.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    def runner(command, **kwargs):
        stage = Path(command[1]).name
        materialize("knowledge-graph" if stage == "text-extractor.py" else "flashcards", paths)
        return subprocess.CompletedProcess(command, 0)

    pipeline.run(args, command_runner=runner)
    paths.flashcard_parts[1].unlink()
    calls = []

    def rerun(command, **kwargs):
        calls.append(Path(command[1]).name)
        materialize("flashcards", paths)
        return subprocess.CompletedProcess(command, 0)

    pipeline.run(args, command_runner=rerun)

    assert calls == ["main.py"]


def test_force_recomputes_every_stage(tmp_path):
    args = make_args(tmp_path, "--force")
    paths = pipeline.pipeline_paths(
        args.input, args.output_root, args.course_code, args.module_number
    )
    for stage in ("knowledge-graph", "flashcards"):
        materialize(stage, paths)
    calls = []

    def command_runner(command, **kwargs):
        stage = {
            "text-extractor.py": "knowledge-graph",
            "main.py": "flashcards",
        }[Path(command[1]).name]
        calls.append(stage)
        materialize(stage, paths)
        return subprocess.CompletedProcess(command, 0)

    pipeline.run(args, command_runner=command_runner)

    assert calls == ["knowledge-graph", "flashcards"]


@pytest.mark.parametrize("force", (False, True), ids=("normal", "force"))
def test_partial_upstream_failure_invalidates_stale_downstream_artifacts(
    tmp_path, force
):
    args = make_args(tmp_path, *("--force",) if force else ())
    paths = pipeline.pipeline_paths(
        args.input, args.output_root, args.course_code, args.module_number
    )
    for stage in ("knowledge-graph", "flashcards"):
        materialize(stage, paths)
    if not force:
        paths.unchecked_graph_json.write_text("invalid", encoding="utf-8")
        paths.graph_json.write_text("invalid", encoding="utf-8")
    manifest = paths.workspace / "batch_reuse_manifest.json"
    manifest.write_text("old manifest", encoding="utf-8")

    def first_runner(command, **kwargs):
        stage = {
            "text-extractor.py": "knowledge-graph",
            "main.py": "flashcards",
        }[Path(command[1]).name]
        raise subprocess.CalledProcessError(1, command)

    with pytest.raises(pipeline.PipelineRunError, match="knowledge-graph"):
        pipeline.run(args, command_runner=first_runner)

    assert not paths.graph_json.exists()
    assert not paths.triples_csv.exists()
    assert not paths.flashcards.exists()
    assert not manifest.exists()

    args.force = False
    resumed = []

    def resumed_runner(command, **kwargs):
        stage = {
            "text-extractor.py": "knowledge-graph",
            "main.py": "flashcards",
        }[Path(command[1]).name]
        resumed.append(stage)
        materialize(stage, paths)
        return subprocess.CompletedProcess(command, 0)

    pipeline.run(args, command_runner=resumed_runner)

    assert resumed == ["knowledge-graph", "flashcards"]


def test_stage_failure_stops_the_sequence(tmp_path):
    args = make_args(tmp_path)
    calls = []

    def command_runner(command, **kwargs):
        calls.append(Path(command[1]).name)
        raise subprocess.CalledProcessError(1, command)

    with pytest.raises(pipeline.PipelineRunError, match="knowledge-graph"):
        pipeline.run(args, command_runner=command_runner)

    assert calls == ["text-extractor.py"]


def test_valid_flashcard_reuse_artifact_requires_complete_rendered_structure(tmp_path):
    artifact = tmp_path / "flashcards.txt"
    identity = ModuleIdentity("CPE0021", "01")
    write_module_parts(
        artifact,
        render_module_parts(identity, valid_clusters()),
        identity,
    )

    assert pipeline._valid_flashcards(artifact, "01", "CPE0021")
    assert not pipeline._valid_flashcards(artifact, "01", "CPE0099")


@pytest.mark.parametrize(
    "mutate",
    (
        lambda text: text.split("\n\nModule 01.2\n", 1)[0],
        lambda text: "Module 01.1\nModule 01.2\n",
        lambda text: text.replace("01", "02"),
        lambda text: text.replace("Type,Question", "Question,Type", 1),
        lambda text: text.replace(
            "00000000-0000-4000-8000-000000000001", "not-a-uuid", 1
        ),
        lambda text: text.replace(
            "00000000-0000-4000-8000-000000000002",
            "00000000-0000-4000-8000-000000000001",
        ),
        lambda text: text.replace(
            "00000000-0000-4000-8000-000000000002",
            "00000000-0000-4000-8000-000000000021",
            1,
        ),
        lambda text: "\n".join(
            line
            for index, line in enumerate(text.splitlines())
            if index != 2
        )
        + "\n",
    ),
    ids=(
        "truncated",
        "heading-only",
        "wrong-module",
        "malformed-header",
        "invalid-uuid",
        "duplicate-cluster",
        "excess-cluster",
        "wrong-card-count",
    ),
)
def test_invalid_flashcard_reuse_artifacts_are_rejected(tmp_path, mutate):
    artifact = tmp_path / "flashcards.txt"
    artifact.write_text(mutate(flashcard_content()), encoding="utf-8")

    assert not pipeline._valid_flashcards(artifact, "01")


def test_help_documents_artifacts_resume_and_force(capsys):
    with pytest.raises(SystemExit) as exc_info:
        pipeline.parse_args(["--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "UTF-8 TXT module report" in help_text
    assert "--course-code" in help_text
    assert "--module-number" in help_text
    assert "structured_module.txt" in help_text
    assert "knowledge_graph.json" in help_text
    assert "<course>_M<module>-1.csv" in help_text
    assert "resume" in help_text.casefold()
    assert "--force" in help_text
