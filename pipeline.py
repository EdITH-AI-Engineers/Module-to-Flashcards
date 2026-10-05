from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Callable, Sequence

from artifact_paths import (
    flashcard_output_path,
    flashcard_part_paths,
    flashcard_receipt_path,
    module_file_label,
    safe_path_component,
)
from flashcard_csv import migrate_legacy_module, valid_written_parts
from flashcard_types import ModuleIdentity
from graph_input import GraphInputError, extract_graph_facts, load_graph
from knowledge_graph_checker import is_checked_graph
from local_model import DEFAULT_N_CTX
from worker_budget import parse_cluster_workers
from structured_module import (
    graph_ready_text,
    parse_module_metadata,
    parse_slide_report_metadata,
    validate_slide_report,
)


PROJECT_DIR = Path(__file__).resolve().parent
_BATCH_REUSE_MANIFEST_NAME = "batch_reuse_manifest.json"
_PIPELINE_REUSE_MANIFEST_NAME = "pipeline_reuse_manifest.json"


class PipelineRunError(RuntimeError):
    """Raised when an end-to-end stage fails or omits its promised artifact."""


@dataclass(frozen=True)
class PipelinePaths:
    workspace: Path
    structured_text: Path
    unchecked_graph_dir: Path
    unchecked_graph_json: Path
    unchecked_triples_csv: Path
    graph_dir: Path
    graph_json: Path
    triples_csv: Path
    flashcards: Path

    @property
    def flashcard_parts(self) -> tuple[Path, Path]:
        return flashcard_part_paths(self.flashcards)

    @property
    def flashcard_receipt(self) -> Path:
        return flashcard_receipt_path(self.flashcards)


@dataclass(frozen=True)
class StageCommand:
    name: str
    command: tuple[str, ...]
    expected_output: Path


def _safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem.strip()).strip("._-")
    return stem or "module"


def pipeline_paths(
    source: Path,
    output_root: Path,
    course_code: str | None = None,
    module_number: str | None = None,
) -> PipelinePaths:
    output_root = Path(output_root)
    workspace_root = output_root
    if course_code is not None:
        workspace_root /= safe_path_component(course_code, fallback="course")
    workspace = workspace_root / _safe_stem(Path(source))
    unchecked_graph_dir = workspace / "unchecked_knowledge_graph"
    graph_dir = workspace / "knowledge_graph"
    flashcards = workspace / "flashcards.csv"
    if course_code is not None and module_number is not None:
        flashcards = flashcard_output_path(
            output_root.parent / "flashcards",
            course_code,
            module_number,
        )
    return PipelinePaths(
        workspace=workspace,
        structured_text=workspace / "structured_module.txt",
        unchecked_graph_dir=unchecked_graph_dir,
        unchecked_graph_json=unchecked_graph_dir / "knowledge_graph.json",
        unchecked_triples_csv=unchecked_graph_dir / "triples.csv",
        graph_dir=graph_dir,
        graph_json=graph_dir / "knowledge_graph.json",
        triples_csv=graph_dir / "triples.csv",
        flashcards=flashcards,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a TXT module report -> knowledge graph -> validated "
            "flashcards as a resumable local sequence."
        ),
        epilog=(
            "Each per-module workspace contains structured_module.txt and the "
            "checked knowledge_graph/knowledge_graph.json. CSV output is written to "
            "flashcards/<course>/<course>_M<module>-1.csv and -2.csv. "
            "Valid completed stages are reused when the sequence resumes."
        ),
    )
    parser.add_argument("input", type=Path, help="UTF-8 TXT module report")
    parser.add_argument("--course-code", required=True, help="exact course code")
    parser.add_argument("--module-number", required=True, help="exact module number")
    parser.add_argument(
        "--identity-from-filename",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("pipeline_output"),
        help="root for per-module stage artifacts (default: pipeline_output)",
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--n-ctx", type=int, default=DEFAULT_N_CTX)
    parser.add_argument(
        "--cluster-workers",
        type=parse_cluster_workers,
        default="auto",
        help="parallel model contexts: auto GPU-memory budget or 1–20 (default: auto)",
    )
    parser.add_argument(
        "--clusters-per-call", type=int, choices=(1, 2, 5), default=1,
        help="experimental concepts per model call (default: 1)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0,
        help="maximum seconds for one module (default: disabled)",
    )
    parser.add_argument(
        "--kg-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="REBEL device (default: auto)",
    )
    parser.add_argument(
        "--kg-batch-size",
        type=int,
        default=4,
        help="REBEL chunks per inference batch (default: 4)",
    )
    parser.add_argument(
        "--kg-num-beams",
        type=int,
        default=3,
        help="REBEL beams per generated chunk (default: 3)",
    )
    parser.add_argument(
        "--skip-final-review",
        action="store_true",
        help="skip the local model's final flashcard review calls",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute all stages instead of resuming from valid artifacts",
    )
    return parser.parse_args(argv)


def build_stage_commands(
    args: argparse.Namespace,
    paths: PipelinePaths,
) -> tuple[StageCommand, ...]:
    model_dir = str(Path(args.model_dir).resolve())
    stage_two = [
        sys.executable,
        str((PROJECT_DIR / "text-extractor.py").resolve()),
        str(paths.structured_text.resolve()),
        "--output-dir",
        str(paths.unchecked_graph_dir.resolve()),
        "--course-code",
        args.course_code,
        "--module-number",
        args.module_number,
        "--device",
        args.kg_device,
        "--batch-size",
        str(args.kg_batch_size),
        "--num-beams",
        str(args.kg_num_beams),
    ]
    stage_three = [
        sys.executable,
        str((PROJECT_DIR / "main.py").resolve()),
        str(paths.graph_json.resolve()),
        "--unchecked-graph",
        str(paths.unchecked_graph_json.resolve()),
        "--course-code",
        args.course_code,
        "--module-number",
        args.module_number,
        "--output",
        str(paths.flashcards.resolve()),
        "--course-corpus",
        str((paths.flashcards.parent / "course_corpus.json").resolve()),
        "--model-dir",
        model_dir,
        "--max-retries",
        str(args.attempts),
        "--seed",
        str(args.seed),
        "--n-gpu-layers",
        str(args.n_gpu_layers),
        "--n-ctx",
        str(args.n_ctx),
        "--cluster-workers",
        str(args.cluster_workers),
        "--clusters-per-call",
        str(args.clusters_per_call),
    ]
    if getattr(args, "identity_from_filename", False):
        stage_two.append("--identity-from-filename")
    if args.skip_final_review:
        stage_three.append("--skip-final-review")

    return (
        StageCommand(
            "knowledge-graph", tuple(stage_two), paths.unchecked_graph_json
        ),
        StageCommand("flashcards", tuple(stage_three), paths.flashcard_parts[0]),
    )


def _valid_structured_text(
    path: Path, *, identity_from_filename: bool = False
) -> bool:
    if not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        return False
    metadata = parse_module_metadata(text)
    if metadata:
        return metadata.get("format_version") == "1" and bool(
            graph_ready_text(text).strip()
        )
    report_metadata = parse_slide_report_metadata(text)
    return bool(
        report_metadata
        and (identity_from_filename or report_metadata.get("module_number"))
        and graph_ready_text(text).strip()
    )


def stage_structured_module(
    source: Path,
    destination: Path,
    *,
    course_code: str | None = None,
    module_number: str | None = None,
    identity_from_filename: bool = False,
) -> Path:
    """Validate and atomically stage a supported UTF-8 module unchanged."""
    source = Path(source)
    try:
        source_bytes = source.read_bytes()
        content = source_bytes.decode("utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise PipelineRunError(f"could not read structured module {source}: {exc}") from exc
    metadata = parse_module_metadata(content)
    if metadata:
        if metadata.get("format_version") != "1" or not graph_ready_text(content).strip():
            raise PipelineRunError(
                f"input is not a valid format_version 1 structured module: {source}"
            )
        if (
            not identity_from_filename
            and course_code is not None
            and metadata.get("course_code") != str(course_code).strip()
        ):
            raise PipelineRunError(
                "structured module course_code does not match --course-code"
            )
        if (
            not identity_from_filename
            and module_number is not None
            and module_file_label(metadata.get("module_number", ""))
            != module_file_label(module_number)
        ):
            raise PipelineRunError(
                "structured module module_number does not match --module-number"
            )
    else:
        if course_code is None:
            raise PipelineRunError("course code is required for slide-report input")
        try:
            validate_slide_report(
                content,
                course_code=course_code,
                module_number=module_number,
                identity_from_filename=identity_from_filename,
            )
        except ValueError as exc:
            raise PipelineRunError(f"invalid slide-report input {source}: {exc}") from exc

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(source_bytes)
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return destination


def _valid_graph(path: Path) -> bool:
    try:
        graph = load_graph(path)
        extract_graph_facts(graph)
    except (GraphInputError, OSError, ValueError):
        return False
    return True


def _valid_checked_graph(path: Path) -> bool:
    try:
        graph = load_graph(path)
        extract_graph_facts(graph)
    except (GraphInputError, OSError, ValueError):
        return False
    return is_checked_graph(graph)


def _valid_flashcards(
    path: Path,
    module_number: str,
    course_code: str | None = None,
) -> bool:
    return valid_written_parts(
        path,
        ModuleIdentity(str(course_code or ""), str(module_number)),
        validate_course_code=course_code is not None,
    )


def _pipeline_manifest_contents(args: argparse.Namespace, source: Path) -> dict[str, object]:
    return {
        "output_format": "split-csv-v1",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "course_code": str(args.course_code),
        "module_number": str(args.module_number),
        "settings": {
            **(
                {"identity_from_filename": True}
                if getattr(args, "identity_from_filename", False) else {}
            ),
            "model_dir": str(Path(args.model_dir).resolve()),
            "attempts": args.attempts,
            "seed": args.seed,
            "n_gpu_layers": args.n_gpu_layers,
            "n_ctx": args.n_ctx,
            "cluster_workers": args.cluster_workers,
            "clusters_per_call": args.clusters_per_call,
            "kg_device": args.kg_device,
            "kg_batch_size": args.kg_batch_size,
            "kg_num_beams": args.kg_num_beams,
            "skip_final_review": args.skip_final_review,
        },
    }


def _pipeline_manifest_path(paths: PipelinePaths) -> Path:
    return paths.workspace / _PIPELINE_REUSE_MANIFEST_NAME


def _pipeline_manifest_matches(
    paths: PipelinePaths, identity: dict[str, object]
) -> bool:
    try:
        return json.loads(_pipeline_manifest_path(paths).read_text(encoding="utf-8")) == identity
    except (OSError, ValueError, TypeError):
        return False


def _write_pipeline_manifest(
    paths: PipelinePaths, identity: dict[str, object]
) -> None:
    path = _pipeline_manifest_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False, dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp",
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(identity, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _invalidate_artifacts(paths: Sequence[Path]) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise PipelineRunError(
                f"could not invalidate stale artifact {path}: {exc}"
            ) from exc


def _invalidate_downstream_for_stage(stage_name: str, paths: PipelinePaths) -> None:
    manifest = paths.workspace / _BATCH_REUSE_MANIFEST_NAME
    pipeline_manifest = _pipeline_manifest_path(paths)
    if stage_name == "knowledge-graph":
        _invalidate_artifacts(
            (
                paths.unchecked_graph_json,
                paths.unchecked_triples_csv,
                paths.graph_json,
                paths.triples_csv,
                *paths.flashcard_parts,
                paths.flashcard_receipt,
                manifest,
                pipeline_manifest,
            )
        )
    elif stage_name == "flashcards":
        _invalidate_artifacts(
            (*paths.flashcard_parts, paths.flashcard_receipt, manifest, pipeline_manifest)
        )


def run(
    args: argparse.Namespace,
    *,
    command_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    timeout_seconds: float | None = None,
) -> PipelinePaths:
    source = Path(args.input)
    if not source.is_file():
        raise PipelineRunError(f"module report not found: {source}")
    if source.suffix.casefold() != ".txt":
        raise PipelineRunError(f"input must be a .txt file: {source}")
    if args.attempts < 1:
        raise PipelineRunError("--attempts must be at least 1")

    effective_timeout = (
        getattr(args, "timeout", None)
        if timeout_seconds is None
        else timeout_seconds
    )
    if effective_timeout is not None and effective_timeout < 0:
        raise PipelineRunError("module timeout must not be negative")
    if effective_timeout == 0:
        effective_timeout = None
    paths = pipeline_paths(
        source,
        args.output_root,
        args.course_code,
        args.module_number,
    )
    current_identity = _pipeline_manifest_contents(args, source)
    manifest_exists = _pipeline_manifest_path(paths).is_file()
    reusable_identity = _pipeline_manifest_matches(paths, current_identity)
    try:
        unchanged_staged_source = (
            paths.structured_text.read_bytes() == source.read_bytes()
        )
    except OSError:
        unchanged_staged_source = False
    paths.workspace.mkdir(parents=True, exist_ok=True)
    stage_structured_module(
        source,
        paths.structured_text,
        course_code=args.course_code,
        module_number=args.module_number,
        identity_from_filename=getattr(args, "identity_from_filename", False),
    )
    commands = build_stage_commands(args, paths)
    if (
        not args.force
        and not manifest_exists
        and unchanged_staged_source
        and _valid_checked_graph(paths.graph_json)
        and not _valid_flashcards(paths.flashcards, args.module_number, args.course_code)
    ):
        migrated = migrate_legacy_module(
            paths.flashcards,
            ModuleIdentity(args.course_code, args.module_number),
        )
        if migrated is not None:
            _write_pipeline_manifest(paths, current_identity)
            reusable_identity = True
    validators = {
        "knowledge-graph": lambda path: (
            _valid_checked_graph(paths.graph_json) or _valid_graph(path)
        ),
        "flashcards": lambda path: (
            _valid_checked_graph(paths.graph_json)
            and _valid_flashcards(paths.flashcards, args.module_number, args.course_code)
        ),
    }
    deadline = (
        time.monotonic() + effective_timeout
        if effective_timeout is not None
        else None
    )
    upstream_recomputed = args.force or not reusable_identity
    for stage in commands:
        is_valid = validators[stage.name](stage.expected_output)
        if not args.force and not upstream_recomputed and is_valid:
            print(f"Reusing {stage.name}: {stage.expected_output.resolve()}")
            continue

        print(f"Running {stage.name}...", flush=True)
        try:
            _invalidate_downstream_for_stage(stage.name, paths)
            command_kwargs = {"check": True, "cwd": str(PROJECT_DIR)}
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise PipelineRunError(
                        f"module exceeded {effective_timeout:g}-second timeout"
                    )
                command_kwargs["timeout"] = remaining
            command_runner(list(stage.command), **command_kwargs)
        except subprocess.TimeoutExpired as exc:
            limit = (
                f"{effective_timeout:g}-second"
                if effective_timeout is not None
                else "stage"
            )
            raise PipelineRunError(
                f"{stage.name} stage exceeded the {limit} timeout"
            ) from exc
        except (subprocess.CalledProcessError, OSError) as exc:
            raise PipelineRunError(f"{stage.name} stage failed: {exc}") from exc
        if not validators[stage.name](stage.expected_output):
            raise PipelineRunError(
                f"{stage.name} stage did not create a valid artifact: "
                f"{stage.expected_output}"
            )
        upstream_recomputed = True

    _write_pipeline_manifest(paths, current_identity)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    try:
        paths = run(parse_args(argv))
    except (PipelineRunError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Structured TXT: {paths.structured_text.resolve()}")
    print(f"Knowledge graph: {paths.graph_json.resolve()}")
    for path in paths.flashcard_parts:
        print(f"Flashcards: {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
