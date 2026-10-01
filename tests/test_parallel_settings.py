from pathlib import Path

import api_server
from batch_pipeline import _manifest_settings
import main
import pipeline


def test_twelve_cluster_workers_reach_flashcard_stage(tmp_path):
    args = pipeline.parse_args(
        [
            str(tmp_path / "module.txt"),
            "--course-code", "CPE0021",
            "--module-number", "1",
            "--cluster-workers", "12",
        ]
    )
    paths = pipeline.pipeline_paths(
        args.input, args.output_root, args.course_code, args.module_number
    )
    flashcard_command = pipeline.build_stage_commands(args, paths)[-1].command

    assert flashcard_command[flashcard_command.index("--cluster-workers") + 1] == "12"
    assert main.parse_args(["graph.json", "--cluster-workers", "12"]).cluster_workers == 12


def test_api_cluster_worker_setting_is_passed_to_batch(monkeypatch):
    monkeypatch.setenv("MODULE_FLASHCARDS_CLUSTER_WORKERS", "12")

    args = api_server.pipeline_args(Path("module.txt"), "CPE0021", "1")

    assert args.cluster_workers == 12


def test_auto_worker_policy_is_default_and_stored_in_manifest(tmp_path, monkeypatch):
    monkeypatch.delenv("MODULE_FLASHCARDS_CLUSTER_WORKERS", raising=False)
    base = [
        str(tmp_path / "module.txt"),
        "--course-code", "CPE0021",
        "--module-number", "1",
    ]
    default = pipeline.parse_args(base)
    parallel = pipeline.parse_args([*base, "--cluster-workers", "12"])

    assert default.cluster_workers == "auto"
    assert main.parse_args(["graph.json"]).cluster_workers == "auto"
    assert api_server.pipeline_args(Path("module.txt"), "CPE0021", "1").cluster_workers == "auto"
    assert _manifest_settings(default)["cluster_workers"] == "auto"
    assert _manifest_settings(parallel)["cluster_workers"] == 12
