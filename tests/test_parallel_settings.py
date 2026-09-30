from pathlib import Path

import api_server
from batch_pipeline import _manifest_settings
import main
import pipeline


def test_five_cluster_workers_reach_flashcard_stage(tmp_path):
    args = pipeline.parse_args(
        [
            str(tmp_path / "module.txt"),
            "--course-code", "CPE0021",
            "--module-number", "1",
            "--cluster-workers", "5",
        ]
    )
    paths = pipeline.pipeline_paths(
        args.input, args.output_root, args.course_code, args.module_number
    )
    flashcard_command = pipeline.build_stage_commands(args, paths)[-1].command

    assert flashcard_command[flashcard_command.index("--cluster-workers") + 1] == "5"
    assert main.parse_args(["graph.json", "--cluster-workers", "5"]).cluster_workers == 5


def test_api_cluster_worker_setting_is_passed_to_batch(monkeypatch):
    monkeypatch.setenv("MODULE_FLASHCARDS_CLUSTER_WORKERS", "5")

    args = api_server.pipeline_args(Path("module.txt"), "CPE0021", "1")

    assert args.cluster_workers == 5


def test_default_worker_setting_keeps_existing_manifest_identity(tmp_path):
    base = [
        str(tmp_path / "module.txt"),
        "--course-code", "CPE0021",
        "--module-number", "1",
    ]
    default = pipeline.parse_args(base)
    parallel = pipeline.parse_args([*base, "--cluster-workers", "5"])

    assert "cluster_workers" not in _manifest_settings(default)
    assert _manifest_settings(parallel)["cluster_workers"] == 5
